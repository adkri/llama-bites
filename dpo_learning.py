import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import time


feedback_data = pd.read_csv("test_feedback.csv")
feedback_data["Chosen"] = feedback_data["Prompt"] + " \n" + feedback_data["Choosen"]
feedback_data["Rejected"] = feedback_data["Prompt"] + " \n" + feedback_data["Rejected"]

feedback_data = feedback_data[["Chosen", "Rejected"]]
feedback_data

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "NousResearch/Llama-2-7b-hf", trust_remote_code=True
)


def tokenize_item(prompt, max_length=1024):
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


import torch.nn.functional as F


def forward_and_compute_loss(policy_model, reference_model, batch):
    print(batch)
    chosen = [tokenize_item(item) for item in batch["Chosen"]]
    rejected = [tokenize_item(item) for item in batch["Rejected"]]

    print(chosen)
    print("chosen length", len(chosen))

    print(rejected)
    print("rejected length", len(rejected))

    combined = chosen + rejected

    # chosen : {input_ids, attention_mask, labels} + rejectedc {input_ids, attention_mask, labels}

    # {input_ids: [....], attention_mask: [...], labels: [....]}
    # each item in combined is a dictionary with input_ids, attention_mask, and labels
    # torch stack them
    concatenated = {k: torch.stack([d[k] for d in combined]) for k in combined[0]}

    for key in concatenated.keys():
        concatenated[key] = concatenated[key].to("cuda:0")

    # forward pass through both models
    data = policy_model(**concatenated)

    for key in concatenated.keys():
        concatenated[key] = concatenated[key].detach().cpu()

    loss = data.loss.detach().cpu()
    all_logits = data.logits.to(torch.float32)
    all_logps = _get_batch_logps(
        all_logits.detach().cpu(), concatenated["labels"], average_log_prob=True
    )

    policy_chosen_logps, policy_rejected_logps = (
        all_logps[: len(chosen)],
        all_logps[len(chosen) :],
    )
    # reference_logits = reference_model(**batch)

    # reference_chosen_logits, reference_rejected_logits = torch.split(reference_logits, len(chosen))

    # compute the loss
    loss, choosen_reward, rejected_reward = compute_dpo_loss(
        policy_chosen_logps.detach().cpu(),
        policy_rejected_logps.detach().cpu(),
        None,
        None,
        without_reference=True,
    )

    del all_logits
    del all_logps
    del policy_chosen_logps
    del policy_rejected_logps

    return loss, choosen_reward, rejected_reward


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
    choosen_reward = (policy_chosen).detach()
    rejected_reward = (policy_rejected).detach()

    return losses, choosen_reward, rejected_reward


def _get_batch_logps(
    logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != -100

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, LlamaForCausalLM
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig, PeftModel


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

model = LlamaForCausalLM.from_pretrained(
    "NousResearch/Llama-2-7b-hf",
    device_map={"": 0},  # load on gpu 0
    quantization_config=BNB_CONFIG,
)
model.config.use_cache = False
model.config.pretraining_tp = 1
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LORA_CONFIG)
model.print_trainable_parameters()


dataloader = torch.utils.data.DataLoader(
    feedback_data.to_dict(orient="records"), batch_size=2
)


if __name__ == "__main__":
    model.train()
    # run it through the model

    for step, batch in enumerate(dataloader):
        loss, choosen_reward, rejected_reward = forward_and_compute_loss(
            model, model, batch
        )
        loss.detach().cpu()
        choosen_reward.detach().cpu()
        rejected_reward.detach().cpu()
        print(loss)
        # torch.cuda.empty_cache()
        break

    while True:
        time.sleep(2)
        import gc

        gc.collect()
