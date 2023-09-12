import torch
import pandas as pd
import json
import gc
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from torch.optim.lr_scheduler import StepLR
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, LlamaForCausalLM


BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=False,
)

LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
    inference_mode=False,
)


def get_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        "NousResearch/Llama-2-7b-hf", trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        "NousResearch/Llama-2-7b-hf",
        device_map={"": 0},  # load on gpu 0
        quantization_config=BNB_CONFIG,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    return model, tokenizer


def configure_optimizer(model):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.0,
    )
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.85)
    return optimizer, lr_scheduler


def get_dataset():
    feedback_data = pd.read_csv("test_feedback.csv")
    feedback_data["Chosen"] = feedback_data["Prompt"] + " \n" + feedback_data["Choosen"]
    feedback_data["Rejected"] = (
        feedback_data["Prompt"] + " \n" + feedback_data["Rejected"]
    )

    feedback_data = feedback_data[["Chosen", "Rejected"]]
    return feedback_data


def tokenize_item(tokenizer, prompt, max_length=1024):
    features = tokenizer(prompt)
    # pad the input ids and attention mask, fixes issue with mismatch sizes
    source_ids = features["input_ids"]
    padded_input_ids = source_ids + [tokenizer.pad_token_id] * (
        max_length - len(source_ids)
    )
    attention_mask = features["attention_mask"] + [0] * (max_length - len(source_ids))
    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(padded_input_ids, dtype=torch.long),
    }


def forward_and_compute_loss(tokenizer, policy_model, reference_model, batch):
    chosen = [tokenize_item(tokenizer, item) for item in batch["Chosen"]]
    rejected = [tokenize_item(tokenizer, item) for item in batch["Rejected"]]
    # chosen : {input_ids, attention_mask, labels} + rejected {input_ids, attention_mask, labels}
    combined = chosen + rejected

    concatenated = {k: torch.stack([d[k] for d in combined]) for k in combined[0]}
    # concatenated is {input_ids: [....], attention_mask: [...], labels: [....]}

    for key in concatenated.keys():
        concatenated[key] = concatenated[key].to("cuda:0")
        print(concatenated[key])

    # forward pass through policy model
    policy_logits = policy_model(**concatenated).logits.to(torch.float32)
    policy_logps = _get_batch_logps(
        tokenizer, policy_logits, concatenated["labels"], average_log_prob=True
    )

    policy_chosen_logps, policy_rejected_logps = (
        policy_logps[: len(chosen)],
        policy_logps[len(chosen) :],
    )

    # forward pass through reference model
    with torch.no_grad():
        reference_logits = reference_model(**concatenated).logits.to(torch.float32)
    reference_logps = _get_batch_logps(
        tokenizer, reference_logits, concatenated["labels"], average_log_prob=True
    )

    reference_chosen_logps, reference_rejected_logps = (
        reference_logps[: len(chosen)],
        reference_logps[len(chosen) :],
    )

    # compute the loss
    losses, choosen_reward, rejected_reward = compute_dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    )

    # cleanup gpu memory
    policy_chosen_logps.detach().cpu()
    policy_rejected_logps.detach().cpu()
    reference_chosen_logps.detach().cpu()
    reference_rejected_logps.detach().cpu()
    for key in concatenated.keys():
        concatenated[key] = concatenated[key].cpu()

    return losses.mean(), choosen_reward.cpu(), rejected_reward.cpu()


def compute_dpo_loss(
    policy_chosen,
    policy_rejected,
    reference_choosen,
    reference_rejected,
    without_reference=False,
):
    policy_ratio = policy_chosen - policy_rejected
    reference_ratio = 0

    if not without_reference:
        reference_ratio = reference_choosen - reference_rejected

    logits = policy_ratio - reference_ratio

    losses = -F.logsigmoid(logits)
    choosen_reward = policy_chosen
    rejected_reward = policy_rejected

    return losses, choosen_reward, rejected_reward


def _get_batch_logps(
    tokenizer,
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False,
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of tokenizer.pad_token_id are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != tokenizer.pad_token_id

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == tokenizer.pad_token_id] = 0

    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


if __name__ == "__main__":
    feedback_data = get_dataset()
    dataloader = torch.utils.data.DataLoader(
        feedback_data.to_dict(orient="records"), batch_size=2
    )

    policy_model, tokenizer = get_model_and_tokenizer()
    optimizer, lr_scheduler = configure_optimizer(policy_model)

    reference_model, tokenizer = get_model_and_tokenizer()
    policy_model.train()
    reference_model.eval()

    # start training
    for step, batch in enumerate(dataloader):
        loss, choosen_reward, rejected_reward = forward_and_compute_loss(
            tokenizer, policy_model, reference_model, batch
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(loss)
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()

    # save the model
    policy_model.save_pretrained("dpo_model")
