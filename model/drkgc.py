from pathlib import Path
import numpy as np
import torch
from torch import nn
from transformers import GenerationConfig, Seq2SeqTrainer, LogitsProcessor, LogitsProcessorList
from collections import defaultdict

__all__ = ["DrKGC", "DrKGC_extract", "DrKGC_enhanced", "CustomTrainer"]

from transformers import Seq2SeqTrainer

class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_loss_dict = defaultdict(float)
        self.custom_loss_steps = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        if self.model.training and isinstance(outputs, dict):
            for key,val in outputs.items():
                if key.endswith("_loss") and key != "loss":
                    self.custom_loss_dict[key] += val.item() if hasattr(val, "item") else val
            self.custom_loss_steps += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict):
        if self.custom_loss_steps > 0:
            for key, total_loss in self.custom_loss_dict.items():
                logs[key] = total_loss / self.custom_loss_steps
            self.custom_loss_dict.clear()
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
    def __init__(self, tokenizer, llm_model, graph_model, extract_model, extract_loss_weight, use_attention, new_token):
        super().__init__(tokenizer, llm_model, graph_model)
        self.extract_model = extract_model
        self.new_token = new_token
        if self.new_token:
            self.extract_id = self.tokenizer.convert_tokens_to_ids(['<|extract_kg|>'])[0]
        self.extract_loss_weight = extract_loss_weight
        self.use_attention = use_attention

    def forward(self, input_ids, attention_mask, labels, query_ids, entity_ids, subgraph, triple_ids, is_predicted_tail, extract_positions):
        inputs_embeds = self._replace_placeholders(input_ids, query_ids, entity_ids, subgraph)
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask, 
            labels=labels, 
            output_hidden_states=True, 
            return_dict=True
        )
        #breakpoint()
        last_hidden_state = outputs.hidden_states[-1] # [8, 371, 4096]
        batch_size = last_hidden_state.size(0)
        assert self.new_token == False # self.new_token이 True인 경우 수정 안함. 

        batch_idx = torch.arange(batch_size, device=last_hidden_state.device) # [8]
        extract_pos = torch.stack([batch_idx, extract_positions], dim=1 ) # [8, 2]
        if self.use_attention:
            max_pos = extract_positions.max().item()
            x = last_hidden_state[:, :max_pos+1, :]
            seq_range = torch.arange(max_pos+1, device=last_hidden_state.device).unsqueeze(0) # (1,max_pos+1)
            strict_mask = (seq_range <= extract_positions.unsqueeze(1)).long() # (1,max_pos+1) <= (B,1)
            attn_mask = attention_mask[:, :max_pos+1] * strict_mask
        else:
            x = last_hidden_state[batch_idx, extract_positions]
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

class LateFusionLogitsProcessor(LogitsProcessor):
    def __init__(self, delta_logits):
        self.delta_logits = delta_logits # lm_head(S) 값
        self.step = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 질문을 다 읽고 "첫 번째" 정답 토큰을 생성할 때만 S의 영향을 더해줍니다!
        if self.step == 0:
            scores = scores + self.delta_logits
        self.step += 1
        return scores

class DrKGC_enhanced(DrKGC):
    def __init__(self, tokenizer, llm_model, graph_model, enhanced_model, kgc_loss_weight):
        super().__init__(tokenizer, llm_model, graph_model)
        self.enhanced_model = enhanced_model
        self.kgc_loss_weight = kgc_loss_weight

    def forward(self, input_ids, attention_mask, labels, query_ids, entity_ids, subgraph, triple_ids, is_predicted_tail, extract_positions):
        inputs_embeds = self._replace_placeholders(input_ids, query_ids, entity_ids, subgraph)
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask, 
            output_hidden_states=True, 
            return_dict=True
        )
        last_hidden_states = outputs.hidden_states[-1] # [8,371,4096]
        batch_size = last_hidden_states.size(0) # 8
        batch_indices = torch.arange(batch_size, device=last_hidden_states.device) # tensor([0,1,2,3,4,5,6,7])
        max_pos = extract_positions.max().item()
        lhs_cut = last_hidden_states[:, :max_pos+1, :]
        seq_range = torch.arange(max_pos+1, device=last_hidden_states.device).unsqueeze(0) # [1,366]
        strict_mask = (seq_range<=extract_positions.unsqueeze(1)).long() # [1,366] <= [8,1] -> [8,366]
        attn_mask = attention_mask[:, :max_pos+1] * strict_mask 
        structure_embedding, kgc_loss = self.enhanced_model(lhs_cut, attn_mask, triple_ids, entity_ids, is_predicted_tail)
        ## llm_loss 구하기
        kgc_loss = kgc_loss * self.kgc_loss_weight
        #breakpoint()
        fused_hidden_states = last_hidden_states.clone()
        fused_hidden_states[batch_indices, extract_positions] = fused_hidden_states[batch_indices, extract_positions] + structure_embedding # S 더해주기 
        new_logits = self.llm_model.lm_head(fused_hidden_states) # [8, 371, 128256]
        llm_loss = 0.0
        if labels is not None:
            shift_logits = new_logits[...,:-1,:].contiguous() # [8, 370, 128256]
            shift_labels = labels[...,1:].contiguous() # [8,370]
            loss_fct = nn.CrossEntropyLoss()
            llm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)) #([2960,128256],[2960])
        outputs["logits"] = new_logits 
        outputs["loss"] = llm_loss + kgc_loss 
        outputs["llm_loss"] = llm_loss
        outputs["kgc_loss"] = kgc_loss
        return outputs

    def save_pretrained(self, save_dir):
        super().save_pretrained(save_dir)
        save_dir = Path(save_dir)
        torch.save(self.enhanced_model.state_dict(), save_dir / "enhanced_model.bin")

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, query_ids, entity_ids, triple_ids, is_predicted_tail, subgraph=None, generation_config=None):
        # 1. 입력 임베딩 준비
        inputs_embeds = self._replace_placeholders(input_ids, query_ids, entity_ids, subgraph)
        # 2. LLM을 한 번 통과시켜서 H (last_hidden_states) 추출
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False # 순수하게 H만 뽑을 것이므로 캐시 끔
        ) # output['logits'].shape (1,L,Vocab) # output['hidden_states'][0].shape (1,L,H) -이게 최초 입력+레이어 통과 32개 = 총 33개 
        last_hidden_states = outputs.hidden_states[-1]
        #breakpoint()
        structure_embedding, kgc_loss = self.enhanced_model(last_hidden_states, attention_mask, triple_ids, entity_ids, is_predicted_tail)

        # 4. 수학적 마법: lm_head(S)를 미리 계산해 둡니다 (이게 바로 delta_logits)
        delta_logits = self.llm_model.lm_head(structure_embedding) # [1,4096] -> [1,128256]
        
        # 5. 첫 생성 토큰에만 delta_logits를 더해주는 프로세서 장착
        processor = LateFusionLogitsProcessor(delta_logits)
        logits_processor = LogitsProcessorList([processor])
        
        if generation_config is None:
            generation_config = GenerationConfig()
            
        # 6. 드디어 진짜 생성 (LogitsProcessor가 중간에 개입하여 점수를 조작함!)
        return self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            logits_processor=logits_processor
        )