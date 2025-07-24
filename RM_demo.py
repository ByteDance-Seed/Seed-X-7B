import logging
from typing import List

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, MistralForCausalLM
from safetensors.torch import load_file

class RewardModel:
    def __init__(self, model_dir) -> None:
        config = AutoConfig.from_pretrained(model_dir)
        # config._attn_implementation = "flash_attention_2"
        self.device = torch.device('cuda')
        self.model = MistralForCausalLM(config)
        self.model.lm_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        state_dict = load_file(f"{model_dir}/model.safetensors")
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(dtype=torch.bfloat16)
        self.model.to(device=self.device)
        self.model.eval()
        logging.info("Load model completed.")

    @torch.no_grad()
    def score(self, prompts, chosens) -> List[float]:
        # Concat prompt and chosen, append eos_id
        input_ids_list = [self.tokenizer.encode(prompt) + self.tokenizer.encode(chosen) + [self.tokenizer.eos_token_id] for prompt, chosen in zip(prompts, chosens)]

        # Pad sequences to the maximum length
        max_length = max(len(ids) for ids in input_ids_list)
        padded_input_ids = [ids + [self.tokenizer.pad_token_id or self.tokenizer.eos_token_id] * (max_length - len(ids)) for ids in input_ids_list]

        # Forward pass
        input_ids = torch.tensor(padded_input_ids).to(device=self.device)
        logits = self.model(input_ids).logits

        # Extract logits corresponding to eos_token_id positions
        scores = []
        for i, input_ids in enumerate(input_ids_list):
            eos_position = input_ids.index(self.tokenizer.eos_token_id)
            eos_logit = logits[i, eos_position, :].squeeze().item()
            scores.append(eos_logit)

        return scores

if __name__ == '__main__':

    local_model_dir = "Your local model dir"
    model_dir = f"{local_model_dir}/Seed-X-RM-7B"  
    prompt = ["Translate the following English sentence into Chinese:\nMay the force be with you <zh>"]
    candidate = ["愿原力与你同在","愿力量与你同在"]
    model = RewardModel(model_dir)
    scores = model.score(prompt, candidate)    # output [1.46875, -0.376953125]
    print(scores)
