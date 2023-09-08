import argparse
import time
import torch
import torch.optim as optim
from collections import namedtuple
from datasets import load_dataset
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig, PeftModel
from pathlib import Path
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from transformers import (
    BitsAndBytesConfig, 
    LlamaForCausalLM,
    LlamaTokenizer,
    default_data_collator,
)

DatasetItem = namedtuple("DatasetItem", ["input_field", "output_field"])

# //// SETUP ////

"""
Steps for running the finetuning script:
1. Configure the hyperparameters in the CONFIG dictionary
2. Setup the datasets for training and evaluation
3. Configure the prompt formatter for the dataset
"""


# //// CONFIG ////

CONFIG = {
    "model_name": "NousResearch/Llama-2-7b-hf",
    "lora_output_path": "./training_output/",
    "finetuned_model_name": "new-model-name",
    "seed": 111111,
    "gamma": 0.85,
    "epochs": 3,
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-4,
    "training_batch_size": 4,
    "eval_batch_size": 1,
    "num_workers_dataloader": 1,
    "should_run_validation": True,
    "should_save_model": True,
}


LORA_CONFIG = LoraConfig(
    r = 8,
    lora_alpha = 32,
    target_modules = ["q_proj", "v_proj"],
    bias = "none",
    task_type = "CAUSAL_LM",
    lora_dropout = 0.05,
    inference_mode = False,
)


BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=False,
)


def get_training_dataset(tokenizer):
    return get_tokenized_dataset(tokenizer, split="train", source="huggingface", dataset_name="ChrisHayduk/Llama-2-SQL-Dataset")
    # return get_dataset(None, split="train", source="local", local_file="data/train.csv")  


def get_eval_dataset(tokenizer):
    return get_tokenized_dataset(tokenizer, split="eval", source="huggingface", dataset_name="ChrisHayduk/Llama-2-SQL-Dataset")
    # return get_dataset(None, split="eval", source="local", local_file="data/train.csv")  


def default_prompt_formatter(item: DatasetItem):
    return f"Query: {item['input']}\nResult: {item['output']}"


# //// DATALOADER UTILS ////

def get_tokenized_dataset(tokenizer, split="train", source="huggingface", dataset_name=None, local_file=None, prompt_formatter=default_prompt_formatter):
    if source == "huggingface":
        return TokenizedDatasetHF(tokenizer, dataset_name, split, prompt_formatter)
    elif source == "local":
        return TokenizedDataset(tokenizer, load_from_csv, local_file, split, prompt_formatter)
    else:
        raise ValueError(f"Invalid source: {source}")


def load_from_csv(file_path, split="train"):
    print(f"Loading dataset from csv file: {file_path}")
    data =  load_dataset("csv", data_files={split: [file_path]}, delimiter=",")
    for item in data:
        yield DatasetItem(item["col1"], item["col2"])


def load_from_jsonl(file_path, split="train"):
    print(f"Loading dataset from jsonl file: {file_path}")
    data =  load_dataset("json", data_files={split: [file_path]})
    for item in data:
        yield DatasetItem(item["input"], item["output"])


def tokenize_item(tokenizer, item, prompt_formatter, max_length=1024):
    prompt = prompt_formatter(item)
    # print(prompt)
    features = tokenizer(prompt)
    # pad the input ids and attention mask, fixes issue with mismatch sizes
    source_ids = features["input_ids"]
    padded_input_ids = source_ids + [tokenizer.pad_token_id] * (max_length - len(source_ids))
    attention_mask = features["attention_mask"] + [0] * (max_length - len(source_ids))
    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(padded_input_ids, dtype=torch.long),
    }

class TokenizedDatasetHF(Dataset):
    def __init__(self, tokenizer, dataset_name, split, prompt_formatter, max_length=512):
        self.tokenizer = tokenizer
        self.dataset = load_dataset(dataset_name, split=split).select(range(10))
        self.prompt_formatter = prompt_formatter
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return tokenize_item(self.tokenizer, self.dataset[idx], self.prompt_formatter, self.max_length)


class TokenizedDataset(Dataset):
    def __init__(self, tokenizer, data_source_func, source_path, split, prompt_formatter, max_length=512):
        self.tokenizer = tokenizer
        self.data_gen = data_source_func(source_path, split)
        self.prompt_formatter = prompt_formatter
        self.max_length = max_length
        self.len_gen = data_source_func(source_path, split)
        self._length = 0

    def __len__(self):
        if self._length == 0:
            self._length = sum(1 for _ in self.len_gen) # this is expensive
        return self._length

    def __getitem__(self, idx):
        return tokenize_item(self.tokenizer, next(self.data_gen), self.prompt_formatter, self.max_length)


# //// MAIN DRIVER ////

def init_tokenizer():
    tokenizer = LlamaTokenizer.from_pretrained(CONFIG["model_name"])
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    return tokenizer


def init_model(tokenizer):
    model = LlamaForCausalLM.from_pretrained(
        CONFIG["model_name"],
        quantization_config=BNB_CONFIG,
        device_map={"": 0},
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # print model size
    print(f"--> Model {CONFIG['model_name']}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--> {CONFIG['model_name']} has {total_params / 1e6} Million params\n")

    # quantize and prepare the peft model
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()
    model.resize_token_embeddings(len(tokenizer))
    return model


def configure_optimizer(model):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=0.0,
    )
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=CONFIG["gamma"])
    return optimizer, lr_scheduler


def get_dataloader(tokenizer, split, batch_size):
    return torch.utils.data.DataLoader(
        get_training_dataset(tokenizer=tokenizer) if split == "train" else get_eval_dataset(tokenizer=tokenizer),
        batch_size=batch_size,
        num_workers=CONFIG["num_workers_dataloader"],
        pin_memory=True,
        drop_last=True,
        collate_fn=default_data_collator,
    )


def train(
    model,
    tokenizer,
    optimizer,
    lr_scheduler,
    train_dataloader,
    eval_dataloader,
):
    epoch_times = []
    gradient_accumulation_steps = CONFIG["gradient_accumulation_steps"]

    # perplexity and loss
    train_loss = []
    train_perp = []

    # evaluation
    best_val_loss = float("inf")
    eval_losses = []
    eval_perps = []
    
    # checkpointing
    ckpt_times = []

    for epoch in range(CONFIG["epochs"]):
        start_time = time.perf_counter()
        # set to train mode
        model.train()

        total_loss = 0.0
        total_len = len(train_dataloader) // gradient_accumulation_steps

        pbar = tqdm(colour="green", desc=f"Training epoch: {epoch}", total=total_len)
        for step, batch in enumerate(train_dataloader):
            for key in batch.keys():
                batch[key] = batch[key].to("cuda:0")
            loss = model(**batch).loss
            loss = loss / gradient_accumulation_steps
            total_loss += loss.detach().float()

            loss.backward()
            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                optimizer.zero_grad()
                pbar.update(step//gradient_accumulation_steps)
            pbar.set_description(f"Training epoch: {epoch}/{CONFIG['epochs']}, step: {step}/{len(train_dataloader)} completed - loss: {loss.detach().float()}")

        epoch_end_time = time.perf_counter()-start_time
        epoch_times.append(epoch_end_time)

        train_epoch_loss = total_loss / len(train_dataloader)
        train_perplexity = torch.exp(train_epoch_loss)
        train_perp.append(train_perplexity)
        train_loss.append(train_epoch_loss)

        # update the lr
        lr_scheduler.step()

        # evaluate
        if CONFIG["should_run_validation"]:
            eval_perp, eval_loss = evaluate(model, tokenizer, eval_dataloader)

            ckpt_start_time = time.perf_counter()
            if CONFIG["should_save_model"] and eval_loss < best_val_loss:
                model.save_pretrained(CONFIG['lora_output_path'])
            ckpt_end_time = time.perf_counter() - ckpt_start_time
            ckpt_times.append(ckpt_end_time)

            if eval_loss < best_val_loss:
                best_val_loss = eval_loss
                print(f"New best validation loss: {best_val_loss} on epoch: {epoch}")
            eval_losses.append(best_val_loss)
            eval_perps.append(eval_perp)
        
        print(f"Epoch: {epoch+1}, train_perplexity: {train_perplexity:.4f}, train_epoch_loss: {train_epoch_loss:.4f}, epoch_time: {epoch_end_time}s")

    # final stats
    results = {
        "avg_epoch_time": sum(epoch_times) / len(epoch_times),
        "avg_ckpt_time": sum(ckpt_times) / len(ckpt_times) if len(ckpt_times) > 0 else 0,
        "avg_train_perp": sum(train_perp) / len(train_perp),
        "avg_train_loss": sum(train_loss) / len(train_loss),
    }
    if CONFIG["should_run_validation"]:
        results["avg_eval_perp"] = sum(eval_perps) / len(eval_perps)
        results["avg_eval_loss"] = sum(eval_losses) / len(eval_losses)

    return results


def evaluate(model, tokenizer, eval_dataloader):
    # set to eval mode
    model.eval()
    eval_preds = []
    eval_loss = 0.0

    for _, batch in enumerate(tqdm(eval_dataloader, colour="blue", desc="Evaluating")):
        for key in batch.keys():
            batch[key] = batch[key].to("cuda:0")
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
        # decode pred logits 
        logits = torch.argmax(outputs.logits, -1).detach().cpu().numpy()
        eval_preds.append(
            tokenizer.batch_decode(logits, skip_special_tokens=True)
        )
    
    # compute average loss and perplexity
    eval_loss = eval_loss / len(eval_dataloader)
    eval_perp = torch.exp(eval_loss)

    print(f"Eval loss: {eval_loss}, Eval perplexity: {eval_perp}")
    return eval_perp, eval_loss


def main_pipeline():
    torch.cuda.manual_seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])

    tokenizer = init_tokenizer()
    model = init_model(tokenizer)
    optimizer, lr_scheduler = configure_optimizer(model)

    train_dataloader = get_dataloader(tokenizer, "train", CONFIG["training_batch_size"])
    eval_dataloader = get_dataloader(tokenizer, "eval", CONFIG["eval_batch_size"]) if CONFIG["should_run_validation"] else None

    results = train(
        model,
        tokenizer,
        optimizer,
        lr_scheduler,
        train_dataloader,
        eval_dataloader,
    )

    # print results
    [print(f"{key}: {value}") for key, value in results.items()]


def load_and_merge_lora_to_model():
    base_model = LlamaForCausalLM.from_pretrained(
        CONFIG["model_name"],
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map={"": 0}
    )
    model = PeftModel.from_pretrained(base_model, CONFIG["lora_output_path"])
    model = model.merge_and_unload()
    tokenizer = init_tokenizer()

    # save the model and tokenizer to the final model name
    model.save_pretrained(CONFIG["finetuned_model_name"])
    tokenizer.save_pretrained(CONFIG["finetuned_model_name"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the finetuning script or merge lora to the final model")
    parser.add_argument("--mode", type=str, choices=["finetune", "merge"], required=True, help="The mode to run the script in, either 'finetune' or 'merge'")
    args = parser.parse_args()

    if args.mode == "finetune":
        main_pipeline()
    elif args.mode == "merge":
        load_and_merge_lora_to_model()

