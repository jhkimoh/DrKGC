from pathlib import Path
import numpy as np
import torch
from torch import nn
from transformers import GenerationConfig, Seq2SeqTrainer

__all__ = ["DrKGC", "DrKGC_extract", "CustomTrainer"]

from transformers import Seq2SeqTrainer

class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_loss_dict = {
            "lm_loss": 0.0, 
            "label_loss": 0.0, 
            "reconstruction_loss": 0.0, 
            "kgc_loss": 0.0
        }
        self.custom_loss_steps = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        if self.model.training and isinstance(outputs, dict):
            for key in self.custom_loss_dict.keys():
                val = outputs.get(key)
                if val is not None:
                    # 텐서면 .item()으로 스칼라 변환, 파이썬 숫자면 그대로 누적
                    self.custom_loss_dict[key] += val.item() if hasattr(val, "item") else val
            self.custom_loss_steps += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict):
        if self.custom_loss_steps > 0:
            for key in self.custom_loss_dict:
                logs[key] = self.custom_loss_dict[key] / self.custom_loss_steps
                self.custom_loss_dict[key] = 0.0
            self.custom_loss_steps = 0
        super().log(logs)

class DrKGC(nn.Module):
    def __init__(self, tokenizer, llm_model, graph_model):
        super().__init__()
        self.tokenizer = tokenizer
        self.llm_model = llm_model
        self.graph_model = graph_model
        self.query_token_id = self.tokenizer.convert_tokens_to_ids(['[QUERY]'])[0]
        self.entity_token_id = self.tokenizer.convert_tokens_to_ids(['[ENTITY]'])[0]


    def _replace_placeholders(self, input_ids: torch.Tensor, query_ids: torch.Tensor, entity_ids: torch.Tensor, subgraph=None):
        query_embeds, entity_embeds = self.graph_model(query_ids, entity_ids, subgraph)

        clean_ids = input_ids.clone()
        clean_ids[clean_ids == self.query_token_id] = self.tokenizer.pad_token_id
        clean_ids[clean_ids == self.entity_token_id] = self.tokenizer.pad_token_id
        inputs_embeds = self.llm_model.model.model.embed_tokens(clean_ids).clone()

        query_pos = torch.nonzero(input_ids == self.query_token_id)
        entity_pos = torch.nonzero(input_ids == self.entity_token_id)
        inputs_embeds[query_pos[:, 0], query_pos[:, 1]] = query_embeds
        inputs_embeds[entity_pos[:, 0], entity_pos[:, 1]] = entity_embeds
        return inputs_embeds


    def forward(self,input_ids, attention_mask, labels, query_ids, entity_ids, subgraph):
        inputs_embeds = self._replace_placeholders(input_ids, query_ids, entity_ids, subgraph)

        return self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
    
    def save_pretrained(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.llm_model.save_pretrained(save_dir)
        torch.save(self.graph_model.state_dict(), save_dir / "graph_model.bin")


    @torch.no_grad()
    def generate(
        self, input_ids, query_ids, entity_ids, subgraph = None, generation_config: GenerationConfig = None):
        inputs_embeds = self._replace_placeholders(input_ids, query_ids, entity_ids, subgraph)
        
        if generation_config is None:
            generation_config = GenerationConfig()
        
        return self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            generation_config=generation_config,
        )    

class DrKGC_extract(DrKGC):
    def __init__(self, tokenizer, llm_model, graph_model, extract_model, extract_loss_weight, use_attention):
        super().__init__(tokenizer, llm_model, graph_model)
        self.extract_model = extract_model
        self.extract_id = self.tokenizer.convert_tokens_to_ids(['<|extract_kg|>'])[0]
        self.extract_loss_weight = extract_loss_weight
        self.use_attention = use_attention

    def forward(self, input_ids, attention_mask, labels, query_ids, entity_ids, subgraph, triple_ids, is_predicted_tail):
        inputs_embeds = self._replace_placeholders(input_ids, query_ids, entity_ids, subgraph)
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask, 
            labels=labels, 
            output_hidden_states=True, 
            return_dict=True
        )
        last_hidden_state = outputs.hidden_states[-1]
        extract_pos = torch.nonzero(input_ids == self.extract_id, as_tuple=False)
        if extract_pos.numel() == 0:
            raise ValueError("No extract token '<|extract_kg|>' found in input_ids for DrKGC_extract.")
        if self.use_attention:
            max_pos = extract_pos[:, 1].max().item()
            x = last_hidden_state[:, :max_pos + 1, :]
            attn_mask = attention_mask[:, :max_pos + 1]
        else:
            x = last_hidden_state[extract_pos[:,0], extract_pos[:,1]]
            attn_mask = None
        extract_outputs = self.extract_model(x, extract_pos, attn_mask, query_ids, entity_ids, triple_ids, is_predicted_tail, subgraph)
        lm_loss = outputs.loss
        outputs.loss = lm_loss + (extract_outputs["total_loss"] * self.extract_loss_weight)
        outputs["lm_loss"] = lm_loss
        outputs["reconstruction_loss"] = extract_outputs["reconstruction_loss"]
        outputs["label_loss"] = extract_outputs["label_loss"]
        outputs["kgc_loss"] = extract_outputs["kgc_loss"]
        return outputs

    def save_pretrained(self, save_dir):
        super().save_pretrained(save_dir)
        save_dir = Path(save_dir)
        torch.save(self.extract_model.state_dict(), save_dir / "extract_model.bin")