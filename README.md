# Llama Bites

scripts to work with llama on a single 3090

### Features
- qLoRA finetuning (adapted from https://github.com/facebookresearch/llama-recipes/blob/main/src/llama_recipes/finetuning.py)
  - This is further supervised learning on top of Llama pretrained model with collected dataset

- Direct Preference Optimization(DPO) for further finetuning by reward modeling ([paper](https://arxiv.org/pdf/2305.18290.pdf))
  - Adapted parts of code from https://github.com/eric-mitchell/direct-preference-optimization
  - This is further finetuning with rewards based on a chosen and rejected pair of prompts
  - Dataset is in format of (prompt, chosen response, rejected response)


