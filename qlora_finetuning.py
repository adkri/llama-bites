import torch
from tqdm import tqdm
from transformers import (
    BitsAndBytesConfig, 
    LlamaForCausalLM,
    LlamaTokenizer,
    default_data_collator,
)
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig, PeftModel
from datasets import load_dataset
from pathlib import Path

from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import time


# //// SETUP ////

"""
Steps for running the finetuning script:
1. Configure the hyperparameters in the CONFIG dictionary
2. Setup the datasets for training and evaluation
"""


# //// CONFIG ////

CONFIG = {
    "model_name": "NousResearch/Llama-2-7b-hf",
    "output_path": "./training_output/",
    "merged_model_name": "my-merged-name",
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


# //// DATALOADER ////

def get_dataset(tokenizer, split="train"):
    # load the dataset from huggingface 
    hf_dataset_name = "ChrisHayduk/Llama-2-SQL-Dataset"
    print("Loading dataset from huggingface: ", hf_dataset_name)
    data = load_dataset(hf_dataset_name, split=split)
    return FinetuningDataset(tokenizer, data, split=split)

    # load the dataset from local files
    if split == "train":
        train_path = Path.cwd() / "data" / "train.csv"
        print("Loading train dataset from: ", train_path)
        return FinetuningDataset(tokenizer, local_file=train_path, split="train")
    else: # eval
        eval_path = Path.cwd() / "data" / "eval.csv",
        print("Loading eval dataset from: ", eval_path)
        return FinetuningDataset(tokenizer, local_file=eval_path, split="eval")


class FinetuningDataset(Dataset):
    def __init__(self, tokenizer, dataset, local_file=None, split="train"):
        self.tokenizer = tokenizer
        self.split = "train" if split == "train" else "eval"
        self.is_local = local_file is not None
        self.max_length = 1024
        if dataset is not None:
            self.dataset = dataset
        elif local_file is not None:
            self.dataset = load_dataset(
                "csv",
                data_files={
                    split: [local_file]
                },
                delimiter=",",
            )

    def __len__(self):
        if self.is_local:
            return self.dataset[self.split].shape[0]
        return self.dataset.__len__()

    def get_item_features(self, item):
        # create prompt if needed
        prompt = f"Query: {item['input']} \n Result: {item['output']}"
        # print(prompt)
        sample = self.tokenizer(prompt)
        return sample
    
    def __getitem__(self, idx):
        if self.is_local:
            item = {
                "input": self.dataset[self.split].iloc[idx]["input"],
                "output": self.dataset[self.split].iloc[idx]["output"],
            }
        else:
            item = self.dataset.__getitem__(idx)

        features = self.get_item_features(item)

        # pad the input ids and attention mask, fixes issue with mismatch sizes
        source_ids = features["input_ids"]
        padded_input_ids = source_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(source_ids))
        attention_mask = features["attention_mask"] + [0] * (self.max_length - len(source_ids))

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_input_ids, dtype=torch.long),
        }


# //// MAIN DRIVER ////

def main():
    torch.cuda.manual_seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])

    model = init_model()
    tokenizer = LlamaTokenizer.from_pretrained(CONFIG["model_name"])
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    model.resize_token_embeddings(len(tokenizer))

    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=0.0,
    )
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=CONFIG["gamma"])

    # get dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        get_dataset(tokenizer, split="train"),
        batch_size=CONFIG["training_batch_size"],
        num_workers=CONFIG["num_workers_dataloader"],
        pin_memory=True,
        drop_last=True,
        collate_fn=default_data_collator,
    )
    eval_dataloader = None
    if CONFIG["should_run_validation"]:
        eval_dataloader = torch.utils.data.DataLoader(
            get_dataset(tokenizer, split="eval"),
            batch_size=CONFIG["eval_batch_size"],
            num_workers=CONFIG["num_workers_dataloader"],
            pin_memory=True,
            drop_last=True,
            collate_fn=default_data_collator,
        )

    # run the training loop
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


def init_model():
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
    return model


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
    eval_loss = []
    eval_perp = []
    
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
                model.save_pretrained(CONFIG['output_path'])
            ckpt_end_time = time.perf_counter() - ckpt_start_time
            ckpt_times.append(ckpt_end_time)

            if eval_loss < best_val_loss:
                best_val_loss = eval_loss
                print(f"New best validation loss: {best_val_loss} on epoch: {epoch}")
            eval_loss.append(best_val_loss)
            eval_perp.append(eval_perp)
        
        print(f"Epoch: {epoch+1}, train_perplexity: {train_perplexity:.4f}, train_epoch_loss: {train_epoch_loss:.4f}, epoch_time: {epoch_end_time}s")

    # final stats
    results = {
        "avg_epoch_time": sum(epoch_times) / len(epoch_times),
        "avg_ckpt_time": sum(ckpt_times) / len(ckpt_times) if len(ckpt_times) > 0 else 0,
        "avg_train_perp": sum(train_perp) / len(train_perp),
        "avg_train_loss": sum(train_loss) / len(train_loss),
    }
    if CONFIG["should_run_validation"]:
        results["avg_eval_perp"] = sum(eval_perp) / len(eval_perp)
        results["avg_eval_loss"] = sum(eval_loss) / len(eval_loss)

    return results


def evaluate(model, tokenizer, eval_dataloader):
    # set to eval mode
    model.eval()
    eval_preds = []
    eval_loss = 0.0

    for step, batch in enumerate(tqdm(eval_dataloader, colour="blue", desc="Evaluating")):
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


def merge_result_output_to_final_model(tokenizer):
    base_model = LlamaForCausalLM.from_pretrained(
        CONFIG["model_name"],
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map={"": 0}
    )
    model = PeftModel.from_pretrained(base_model, CONFIG["output_path"])
    model = model.merge_and_unload()

    # load the tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(CONFIG["model_name"])
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    # save the model and tokenizer to the final model name
    model.save_pretrained(CONFIG["merged_model_name"])
    tokenizer.save_pretrained(CONFIG["merged_model_name"])


if __name__ == "__main__":
    main()

    # --- merge the output to the final model ---
    # remove comment and run this after the training is done in new process to avoid memory issues
    # merge_result_output_to_final_model(tokenizer)

