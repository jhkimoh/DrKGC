import argparse
import json
import math
import os
import collections
import torch 
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import sys
import subprocess
from datetime import datetime 
from transformers import AutoTokenizer, LlamaForCausalLM, GenerationConfig
from torch.cuda.amp import autocast
from peft import PeftModel
from pathlib import Path 

class Scorer:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        kge_state_dict = torch.load(args.kge_checkpoint, map_location=self.device)
        if 'model_state_dict' in kge_state_dict:
            kge_state_dict = kge_state_dict['model_state_dict']
        self.entity_embedding = kge_state_dict['entity_embedding']
        self.relation_embedding = kge_state_dict['relation_embedding']
        self.kge_model_name = args.kge_model_name.lower()
        self.gamma = args.gamma
        self.epsilon = 2.0
        self.R_dim = self.relation_embedding.shape[-1]
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma + self.epsilon) / self.R_dim]), 
            requires_grad=False
        )

    def calculate_scores(self, triple_ids, entity_ids, llm_logits, is_predicted_tail, alpha, beta, tau):
        strategy_func_name = f"_score_{self.args.score_strategy}"
        score_func = getattr(self, strategy_func_name, None)
        if score_func is None:
            raise ValueError(f'Not implemented strategy_func_name {strategy_func_name}!')
        return score_func(triple_ids, entity_ids, llm_logits, is_predicted_tail, alpha, beta, tau)

    def _get_target_point(self, ent_emb, rel_emb, is_tail):
        """
        [Atomic 함수 1] 주어진 엔티티와 관계로 목표점(Target Point) 텐서를 반환합니다.
        - is_tail=True (Tail 예측): h + r  또는 h . r
        - is_tail=False (Head 예측): t - r 또는 t . r^{-1}
        """
        if self.kge_model_name.lower() == 'rotate':
            pi = 3.14159265358979323846
            phase_relation = rel_emb / (self.embedding_range.item() / pi)
            re_rel = torch.cos(phase_relation)
            im_rel = torch.sin(phase_relation)
            re_ent, im_ent = torch.chunk(ent_emb, 2, dim=-1)
            
            if is_tail: # h * r
                re_target = re_ent * re_rel - im_ent * im_rel
                im_target = re_ent * im_rel + im_ent * re_rel
            else: # t * r^{-1} (복소수 켤레 곱셈)
                re_target = re_ent * re_rel + im_ent * im_rel
                im_target = im_ent * re_rel - re_ent * im_rel
            return torch.cat([re_target, im_target], dim=-1)
            
        elif self.kge_model_name.lower() == 'transe': # TransE
            if is_tail:
                return ent_emb + rel_emb
            else:
                return ent_emb - rel_emb
        else:
            raise RuntimeError(f"[에러 감지] 지원하지 않는 kge_model_name이 들어왔습니다: '{self.kge_model_name}'")

    def _calc_distance(self, v1, v2, keepdim=False):
        """
        [Atomic 함수 2] 두 텐서 v1, v2 사이의 KGE 거리를 계산합니다.
        """
        if self.kge_model_name.lower() == 'rotate':
            re_v1, im_v1 = torch.chunk(v1, 2, dim=-1)
            re_v2, im_v2 = torch.chunk(v2, 2, dim=-1)
            # 실수부 차이, 허수부 차이를 구한 뒤 복소수 크기(Norm) 계산 후 차원 합산
            diff_re = re_v1 - re_v2
            diff_im = im_v1 - im_v2
            dist = torch.stack([diff_re, diff_im], dim=0).norm(dim=0).sum(dim=-1, keepdim=keepdim)
            return dist
        if self.kge_model_name.lower() == 'transe': # TransE
            return torch.norm(v1 - v2, p=1, dim=-1, keepdim=keepdim)
        else:
            raise RuntimeError(f"[에러 감지] 지원하지 않는 model_type이 들어왔습니다: '{self.kge_model_name}'")

    def _score_weighted_sum(self, triple_ids, entity_ids, llm_logits, is_predicted_tail, alpha, beta, tau):
        h, r, t = triple_ids[:,0], triple_ids[:,1], triple_ids[:,2]
        llm_probs = F.softmax(llm_logits/tau, dim=0)
        cand_ents = self.entity_embedding[entity_ids]
        query = torch.matmul(llm_probs.unsqueeze(0), cand_ents)
        rel_emb = self.relation_embedding[r].unsqueeze(0)
        is_tail = (is_predicted_tail == 'tail')
        fixed_ent_idx = h if is_tail else t 
        fixed_ent = self.entity_embedding[fixed_ent_idx].unsqueeze(0)
        temp = self._get_target_point(fixed_ent, rel_emb, is_tail)
        delta = self._calc_distance(query, temp, keepdim=True)
        if beta > 0:
            alpha_weight = torch.exp(- beta * delta)
            V_final = alpha_weight * query + (1 - alpha_weight) * temp 
        else:
            V_final = alpha * query + (1 - alpha) * temp 
        distances = self._calc_distance(V_final.unsqueeze(1), self.entity_embedding.unsqueeze(0))
        scores = -distances.squeeze(0)
        return scores

    def _score_only_llm(self, triple_ids, entity_ids, llm_logits, is_predicted_tail, alpha, beta, tau):
        llm_probs = F.softmax(llm_logits/tau, dim=0)
        num_entities = self.entity_embedding.size(0)
        scores = torch.full((num_entities,), float('-inf'), device=self.device)
        scores[entity_ids] = llm_probs
        return scores.unsqueeze(0)

    def _score_modify_query(self, triple_ids, entity_ids, llm_logits, is_predicted_tail, alpha, beta, tau):
        h, r, t = triple_ids[:,0], triple_ids[:,1], triple_ids[:,2]
        llm_probs = F.softmax(llm_logits/tau, dim=0)
        cand_ents = self.entity_embedding[entity_ids]
        rel_emb = self.relation_embedding[r]
        is_tail = (is_predicted_tail == 'tail')
        fixed_ent_idx = h if is_tail else t 
        fixed_ent = self.entity_embedding[fixed_ent_idx]
        temp = self._get_target_point(fixed_ent, rel_emb, is_tail)
        base_distances = self._calc_distance(temp.unsqueeze(1), self.entity_embedding.unsqueeze(0))
        scores = -base_distances.squeeze(0)
        delta = self._calc_distance(temp, cand_ents)
        kge_logits = -delta
        kge_probs = F.softmax(kge_logits, dim=0) # tau 적용? 
        #Delta = llm_probs - kge_probs ##
        Delta = llm_probs ##
        if beta == 0.0 and alpha==0.0:
            V_i = temp
        else:
            alpha_i = Delta * torch.exp(-beta * delta)
            alpha_i = alpha_i.unsqueeze(-1)
            #breakpoint()
            V_i = temp + alpha_i * (cand_ents - temp)
        cand_distances = self._calc_distance(V_i, cand_ents)
        cand_scores = -cand_distances
        scores[entity_ids] = cand_scores
        return scores.unsqueeze(0)

    # Chatgpt 사용
    def _score_candidate_order_kge(self, triple_ids, entity_ids, llm_logits, is_predicted_tail, alpha, beta, tau):
        """
        Candidate는 LLM top-n entity라고 가정.
        Candidate들은 LLM confidence 순서대로 top-n score를 부여하고,
        나머지 entity들은 KGE score를 그대로 사용한다.

        결과:
        - candidate entity는 항상 non-candidate보다 높은 순위
        - candidate 내부 순위는 llm_logits confidence 기준
        - candidate 밖 entity 내부 순위는 KGE score 기준
        """

        h, r, t = triple_ids[:, 0], triple_ids[:, 1], triple_ids[:, 2]

        entity_ids = entity_ids.view(-1).to(self.device)
        llm_logits = llm_logits.view(-1).to(self.device)

        if entity_ids.numel() != llm_logits.numel():
            raise ValueError(
                f"entity_ids 개수({entity_ids.numel()})와 llm_logits 개수({llm_logits.numel()})가 다릅니다."
            )

        # 1. 전체 entity에 대해 KGE score 계산
        rel_emb = self.relation_embedding[r]
        is_tail = (is_predicted_tail == 'tail')

        fixed_ent_idx = h if is_tail else t
        fixed_ent = self.entity_embedding[fixed_ent_idx]

        target_point = self._get_target_point(fixed_ent, rel_emb, is_tail)

        base_distances = self._calc_distance(
            target_point.unsqueeze(1),
            self.entity_embedding.unsqueeze(0)
        )

        scores = -base_distances.squeeze(0)

        # 2. candidate 내부 순위는 LLM confidence 기준
        # raw logits 기준으로 정렬해도 softmax와 순위는 동일함
        sorted_pos = torch.argsort(llm_logits, descending=True)
        sorted_entity_ids = entity_ids[sorted_pos]

        # 3. candidate들이 항상 non-candidate보다 위에 오도록 score 부여
        # KGE 최고점보다 candidate score를 더 크게 만든다.
        max_kge_score = torch.max(scores)

        n = sorted_entity_ids.numel()

        # 예: n=5이면 candidate score는 max_kge+5, +4, +3, +2, +1
        candidate_scores = max_kge_score + torch.arange(
            n,
            0,
            -1,
            device=self.device,
            dtype=scores.dtype
        )

        scores[sorted_entity_ids] = candidate_scores

        return scores.unsqueeze(0)

class Evaluator:
    def __init__(self, args):
        self.args = args
        self.output_dir = os.path.dirname(args.logits_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._prepare_data() # all_true_triple, hr2t, tr2h
        ## test.json 선언
        with open(self.args.test_json_path, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
        self.llm_logits_cache = torch.load(args.logits_path, map_location=self.device)
        self.scorer = Scorer(self.args)
        # 🌟 LLM 직접 로드 파트 추가
        if self.args.use_llm or self.args.divide_length:
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.add_tokens(['[QUERY]', '[ENTITY]', '[RELATION]'])
            if self.args.use_llm:
                from model import GraphEnhancer, DrKGC
                print("\n⏳ [use_llm=True] 분석을 위해 실제 LLM 모델을 로드합니다. (시간 소요 및 VRAM 15GB+ 필요)")
                llm_model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, low_cpu_mem_usage=True, device_map='auto')
                self.llm_model = PeftModel.from_pretrained(llm_model, args.checkpoint_dir)
                loaded_data = torch.load(args.kge_embedding_path, map_location='cpu')
                if isinstance(loaded_data, dict) and 'model_state_dict' in loaded_data:
                    kge_embedding = loaded_data['model_state_dict']['entity_embedding']
                else:
                    kge_embedding = loaded_data
                llm_config = self.llm_model.config
                embed_model = GraphEnhancer(kge_embedding, kge_embedding.shape[1], 4, 128, 1, 1024, llm_config.hidden_size, llm_config.hidden_act)
                ckpt_dir = Path(args.checkpoint_dir)  
                state = torch.load(ckpt_dir / "graph_model.bin", map_location="cpu")
                embed_model.load_state_dict(state)
                self.model = DrKGC(self.tokenizer, self.llm_model, embed_model)
                self.model = self.model.half()
                self.model.cuda()
                self.model.eval()

    def _prepare_data(self):
        # all_true_triple 만들기... 
        with open(os.path.join(self.args.data_path, 'entities.dict')) as fin:
            entity2id = dict()
            self.kge_id2entity = dict()
            for line in fin:
                eid, entity = line.strip().split('\t')
                entity2id[entity] = int(eid)
                self.kge_id2entity[int(eid)] = entity
        with open(os.path.join(self.args.data_path, 'relations.dict')) as fin:
            relation2id = dict()
            for line in fin:
                rid, relation = line.strip().split('\t')
                relation2id[relation] = int(rid)

        self.train_triples = self.read_triple(os.path.join(self.args.data_path, 'train.txt'), entity2id, relation2id)
        self.valid_triples = self.read_triple(os.path.join(self.args.data_path, 'valid.txt'), entity2id, relation2id)
        self.test_triples = self.read_triple(os.path.join(self.args.data_path, 'test.txt'), entity2id, relation2id)
        self.all_true_triples = self.train_triples + self.valid_triples + self.test_triples
        self.hr2t = collections.defaultdict(set)
        self.tr2h = collections.defaultdict(set)
        for h, r, t in self.all_true_triples:
            self.hr2t[(h, r)].add(t)
            self.tr2h[(t, r)].add(h)
    
    @torch.no_grad()
    def compute_llm_confidence(self, prompt, candidates, query_ids, entity_ids, subgraph):
        batch_size = len(candidates)
        prompt_ids = [self.tokenizer.bos_token_id] + self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_ids)
        cand_ids_list = [self.tokenizer.encode(cand, add_special_tokens=False) for cand in candidates]
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        eos_id = self.tokenizer.eos_token_id
        scores = []
        chunk_size = 5
        all_cand_tokens = []
        all_cand_probs = []
        for start_idx in range(0, batch_size, chunk_size):
            end_idx = min(start_idx+chunk_size, batch_size)
            chunk_cand_ids = cand_ids_list[start_idx:end_idx]
            chunk_len = len(chunk_cand_ids)
            input_ids_batch = []
            attention_masks_batch = []
            cand_lengths = []
            max_seq_len = max([prompt_len+len(c)+1 for c in chunk_cand_ids])
            for cand_ids in chunk_cand_ids:
                c_len = len(cand_ids)
                cand_lengths.append(c_len)
                seq = prompt_ids + cand_ids + [eos_id]
                seq_len = len(seq)
                pad_len = max_seq_len - seq_len
                padded_seq = seq + [pad_id] * pad_len
                mask = [1] * seq_len + [0] * pad_len
                input_ids_batch.append(padded_seq)
                attention_masks_batch.append(mask)
            input_ids = torch.tensor(input_ids_batch, dtype=torch.long).cuda()
            attention_masks = torch.tensor(attention_masks_batch, dtype=torch.long).cuda()
            b_query_ids = query_ids.repeat(chunk_len) if query_ids.dim() == 1 else query_ids.repeat(chunk_len, 1)
            b_entity_ids = entity_ids.repeat(chunk_len) if entity_ids.dim() == 1 else entity_ids.repeat(chunk_len, 1)
            b_subgraph = subgraph * chunk_len if subgraph is not None else None
            #breakpoint()
            inputs_embeds = self.model._replace_placeholders(input_ids, b_query_ids, b_entity_ids, b_subgraph)
            outputs = self.llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_masks)
            logits = outputs.logits # [20, seq_len, vocab_size]
            probs = F.softmax(logits, dim=-1)
            
            for i in range(chunk_len):
                c_len = cand_lengths[i]
                if c_len > 0:
                    cand_labels = input_ids[i, prompt_len: prompt_len + c_len + 1] # torch.Size([3])
                    cand_probs_tensor = probs[i, prompt_len-1: prompt_len + c_len]
                    idx_gpu = torch.arange(len(cand_labels), device=cand_labels.device)
                    token_probs = cand_probs_tensor[idx_gpu, cand_labels].tolist()
                    all_cand_tokens.append(cand_labels.tolist())
                    all_cand_probs.append(token_probs)
                else:
                    all_cand_tokens.append([])
                    all_cand_probs.append([])
            del outputs, logits, probs, inputs_embeds, input_ids, attention_masks ## OOM 해결 
        
        num_cands = len(all_cand_tokens)
        final_scores_dict = {i: 0.0 for i in range(num_cands)}
        def calculate_sub_cumulative(indices, depth):
            # 현재 depth에 토큰이 남아있는 후보만 필터링
            valid_indices = [i for i in indices if depth < len(all_cand_tokens[i])]
            if not valid_indices:
                return {i:0.0 for i in indices}
            # 토큰 기준으로 그룹화
            groups = {}
            for i in valid_indices:
                tok = all_cand_tokens[i][depth]
                if tok not in groups:
                    groups[tok] = []
                groups[tok].append(i)
                
            # 확률 오름차순 정렬
            group_list = []
            for tok, idxs in groups.items():
                prob = all_cand_probs[idxs[0]][depth]
                group_list.append((prob, idxs))
            group_list.sort(key=lambda x: x[0])
            
            res = {i: 0.0 for i in indices}
            smaller_sum = 0.0
            for prob, idxs in group_list:
                if len(idxs) == 1:
                    res[idxs[0]] = smaller_sum + prob
                else:
                    sub_cum = calculate_sub_cumulative(idxs, depth + 1)
                    for i in idxs:
                        res[i] = smaller_sum + prob * sub_cum[i]
                smaller_sum += prob
                    
            return res

        # Depth 0 그룹화
        groups_d0 = {}
        for i in range(num_cands):
            if len(all_cand_tokens[i]) > 0:
                tok = all_cand_tokens[i][0]
                if tok not in groups_d0:
                    groups_d0[tok] = []
                groups_d0[tok].append(i)
                
        group_list_d0 = []
        for tok, idxs in groups_d0.items():
            prob = all_cand_probs[idxs[0]][0]
            group_list_d0.append((prob, idxs))
        group_list_d0.sort(key=lambda x: x[0])

        prev_prob = 0.0
        for prob, idxs in group_list_d0:
            if len(idxs) == 1: # 🚀 첫 번째 토큰이 유니크하면 재귀 안 타고 즉시 확정 (속도 향상 및 깔끔함)
                final_scores_dict[idxs[0]] = prob
            else: # 겹치는 토큰들만 서브 누적 확률 계산 태우기
                sub_cum = calculate_sub_cumulative(idxs, 1)
                for i in idxs:
                    final_scores_dict[i] = prev_prob + sub_cum[i] * (prob - prev_prob)
            prev_prob = prob

        # 3. 외부 Scorer 연동용 Log 변환 및 반환
        log_scores = [math.log(max(final_scores_dict.get(i, 0.0), 1e-300)) for i in range(num_cands)]
        scores_tensor = torch.tensor(log_scores, dtype=torch.float32, device=entity_ids.device) 
        
        return scores_tensor

    @torch.no_grad()
    def save_llm_confidence(self, file_name):
        #breakpoint()
        self.model.eval()
        confidence_cache = {}
        for ex_idx, ex in enumerate(tqdm(self.dataset)):
            h, r, t = ex['triple_id']
            pred_type = ex.get('type').split('_')[-1] # 'head' | 'tail'
            query_id = h if pred_type=='tail' else t
            subgraph = [ex['subgraph']] if 'subgraph' in ex else None
            llm_conf_probs = self.compute_llm_confidence( 
                prompt=ex['input'], 
                candidates=ex['rank_entities'], 
                query_ids=torch.LongTensor([ex['query_entity_id']]).cuda(), 
                entity_ids=torch.LongTensor([ex['rank_entities_id']]).cuda(), 
                subgraph=subgraph, 
                # query_ids=torch.LongTensor([query_id]).cuda(), 
                # entity_ids=torch.LongTensor([ex['topk_id']]).cuda() 
            )
            query_key = f"{ex_idx}_{h}_{r}_{t}_{pred_type}"
            confidence_cache[query_key] = llm_conf_probs.cpu()
        save_path = os.path.join(self.output_dir, file_name)
        torch.save(confidence_cache, save_path)
        print(f"\n✅ [성공] LLM Confidence 캐시가 다음 경로에 저장되었습니다: {save_path}")

    def compare_greedy_vs_global(self):
        print("\n🔍 [분석] Global Score 1등 vs 실제 Greedy Decoding 예측 1등 비교 시작...")
        match_count = 0
        total_count = 0
        
        total_global_correct_id = 0
        total_global_correct_text = 0
        total_greedy_correct_text = 0
        
        # ranking_metrics와 동일한 조건(only_llm)으로 맞추기 위한 변수
        alpha = 0.0
        beta = 0.0
        tau = 1.0
        prediction_json_path = os.path.join(self.output_dir, 'prediction.json')
        print(f"📦 Greedy Decoding 결과가 담긴 JSON을 로드합니다: {prediction_json_path}")
        with open(prediction_json_path, 'r', encoding='utf-8') as f:
            pred_data = json.load(f).get("prediction", [])
            
        if len(pred_data) != len(self.dataset):
            print(f"⚠️ 경고: prediction.json의 데이터 개수({len(pred_data)})와 test.json의 데이터 개수({len(self.dataset)})가 다릅니다!")

        comparison_results = []
        for ex_idx, ex in enumerate(tqdm(self.dataset, desc="비교 중")):
            h, r, t = ex['triple_id']
            pred_type = ex.get('type').split('_')[-1]
            query_key = f"{ex_idx}_{h}_{r}_{t}_{pred_type}"
            
            # =================================================================
            # 1. Global Score 기준 1등 찾기 (ranking_metrics와 100% 동일한 로직)
            # =================================================================
            llm_logits = self.llm_logits_cache[query_key]
            
            # 단순 argmax가 아니라 scorer를 태워서 점수화
            output = self.scorer.calculate_scores(
                triple_ids=torch.LongTensor([ex['triplet_id']]).cuda(), 
                entity_ids=torch.LongTensor([ex['topk_id']]).cuda(), 
                llm_logits=llm_logits,
                is_predicted_tail=pred_type, 
                alpha=alpha, beta=beta, tau=tau
            )
            #breakpoint()
            h, r, t = ex['triplet_id']
            target_id = t if pred_type == 'tail' else h
            target_str = ex['output'] # 조작 없는 원본 정답 문자열
            
            # Filtered 세팅 적용 (타겟 이외의 맞는 정답지 페널티)
            target_score = output[0, target_id].clone()
            if pred_type == 'tail':
                true_tails = self.hr2t.get((h, r), set())
                for true_t in true_tails:
                    if true_t != t:
                        output[0, true_t] = target_score - 1.0
            else:
                true_heads = self.tr2h.get((t, r), set())
                for true_h in true_heads:
                    if true_h != h:
                        output[0, true_h] = target_score - 1.0
            
            # 필터링이 모두 끝난 후의 "진짜" 1등 엔티티 ID 추출
            global_top1_id = torch.argmax(output[0, :]).item()
            
            # WN18RR 숫자 ID 대신 자연어 텍스트 가져오기
            cand_ids = ex['topk_id']
            if global_top1_id in cand_ids:
                idx = cand_ids.index(global_top1_id)
                best_global_str = ex['rank_entities'][idx]
            else:
                if global_top1_id == target_id:
                    best_global_str = target_str
                else:
                    best_global_str = f"[Non-Candidate ID: {global_top1_id}]"
            
            global_is_correct_id = (global_top1_id == target_id)
            global_is_correct_text = (best_global_str == target_str)
            
            # =================================================================
            # 2. JSON에서 실제 LLM Greedy Decoding 결과 즉시 불러오기
            # =================================================================
            # LLM을 돌리지 않고 미리 생성된 json의 'pred' 항목을 사용합니다.
            greedy_str = pred_data[ex_idx]['pred'].strip()
            greedy_is_correct_text = (greedy_str == target_str)
            
            # =================================================================
            # 3. 통계 집계 및 출력
            # =================================================================
            is_match = (best_global_str == greedy_str)
            
            if is_match:
                match_count += 1
            if global_is_correct_id:
                total_global_correct_id += 1
            if global_is_correct_text:
                total_global_correct_text += 1
            if greedy_is_correct_text:
                total_greedy_correct_text += 1
            total_count += 1
            
            comparison_results.append({
                "ex_idx": ex_idx,
                "triplet_id": [h, r, t],
                "pred_type": pred_type,
                "target_entity": target_str,
                "global_score_top1": best_global_str,
                "greedy_decoding_top1": greedy_str,
                "is_match": is_match,
                "global_correct_id": global_is_correct_id,
                "global_correct_text": global_is_correct_text,
                "greedy_correct_text": greedy_is_correct_text
            })

            if ex_idx < 5:
                print(f"\n[Sample {ex_idx+1}]")
                print(f" - 정답(Target): '{target_str}'")
                print(f" - 🥇 Global : '{best_global_str}' (ID정답: {global_is_correct_id} | Text정답: {global_is_correct_text})")
                print(f" - 🤖 Greedy : '{greedy_str}' (Text정답: {greedy_is_correct_text})")
                
        # 🌟 결과 계산
        match_rate = (match_count / total_count) * 100
        global_hits1_id = (total_global_correct_id / total_count) * 100
        global_hits1_text = (total_global_correct_text / total_count) * 100
        greedy_hits1_text = (total_greedy_correct_text / total_count) * 100
        
        print("\n" + "="*60)
        print(f"📊 [초고속 채점 일치화 결과 확인 (JSON 로드)]")
        print("="*60)
        print(f"▶️ 전체 Global Score HITS@1 (ID 기준 엄격)  : {global_hits1_id:.2f}%")
        print(f"▶️ 전체 Global Score HITS@1 (Text 기준)     : {global_hits1_text:.2f}% 🔥")
        print(f"▶️ 전체 Greedy Decoding HITS@1 (JSON 기준)  : {greedy_hits1_text:.2f}%")
        print(f"▶️ 두 방식 간의 완벽 일치율 (Match Rate)    : {match_rate:.2f}%")
        print("="*60)
        
        save_filename = f'greedy_vs_global_analysis_{self.args.kge_model_name}.json'
        save_path = os.path.join(self.output_dir, save_filename)
        
        output_data = {
            "summary": {
                "total_count": total_count,
                "match_count": match_count,
                "match_rate": round(match_rate, 2),
                "global_hits1_id": round(global_hits1_id, 2),
                "global_hits1_text": round(global_hits1_text, 2),
                "greedy_hits1_text": round(greedy_hits1_text, 2),
                "kge_model": self.args.kge_model_name,
            },
            "details": comparison_results
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
            
        print(f"📁 상세 비교 분석 결과 및 정답 여부가 JSON으로 안전하게 저장되었습니다: {save_path}")
        
    
    def _apply_length_penalty(self, llm_logits, candidates, strategy="mean_length"):
        adjusted_logits = llm_logits.clone()
        lengths = []
        for cand in candidates:
            tokens = self.tokenizer.encode(cand, add_special_tokens=False)
            lengths.append(max(1, len(tokens)))
        for i,L in enumerate(lengths):
            if strategy == "log_length":
                penalty = math.log2(1+L)
            elif strategy == "mean_length":
                penalty = L
            else:
                raise ValueError("Not implemented length penalty strategy")
            adjusted_logits[i] = adjusted_logits[i] / penalty
        return adjusted_logits

    def ranking_metrics(self, alpha_beta, tau):
        logs = []
        alpha = alpha_beta['alpha']
        beta = alpha_beta['beta']
        #breakpoint()
        for ex_idx, ex in enumerate((self.dataset)):
            h, r, t = ex['triple_id']
            pred_type = ex.get('type').split('_')[-1] # 'head' | 'tail'
            query_key = f"{ex_idx}_{h}_{r}_{t}_{pred_type}"
            llm_logits = self.llm_logits_cache[query_key]
            if self.args.divide_length:
                llm_logits = self._apply_length_penalty(llm_logits=llm_logits, candidates=ex['rank_entities'])
            output = self.scorer.calculate_scores(
                triple_ids=torch.LongTensor([ex['triplet_id']]).cuda() , 
                entity_ids=torch.LongTensor([ex['topk_id']]).cuda(), 
                llm_logits=llm_logits,
                is_predicted_tail=pred_type, 
                alpha=alpha, 
                beta=beta, 
                tau=tau
            )
            h, r, t = ex['triplet_id']
            target_id = t if pred_type == 'tail' else h
            target_score = output[0, target_id].clone()
            if pred_type == 'tail':
                true_tails = self.hr2t.get((h, r), set())
                for true_t in true_tails:
                    if true_t != t: # 타겟 제외
                        output[0, true_t] = target_score - 1.0
            else:
                true_heads = self.tr2h.get((t, r), set())
                for true_h in true_heads:
                    if true_h != h: # 타겟 제외
                        output[0, true_h] = target_score - 1.0
            argsort = torch.argsort(output, dim=1, descending=True)
            ranking_tensor = (argsort[0, :] == target_id).nonzero()
            assert ranking_tensor.size(0) == 1
            ranking = 1 + ranking_tensor.item()
            logs.append({
                    'MRR': 1.0 / ranking,
                    'MR': float(ranking),
                    'HITS@1': 1.0 if ranking <= 1 else 0.0,
                    'HITS@3': 1.0 if ranking <= 3 else 0.0,
                    'HITS@10': 1.0 if ranking <= 10 else 0.0,
                })
        metrics = {}
        if len(logs) > 0:
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
        metrics = {k: round(v, 8) for k, v in metrics.items()}
        metrics = {
            'mrr': metrics['MRR'], 'mr': metrics['MR'],
            'hits1': metrics['HITS@1'], 'hits3': metrics['HITS@3'], 'hits10': metrics['HITS@10']
        }
        print(f"\n[{datetime.now()} {self.args.data_path}, {self.args.kge_model_name}, {self.args.score_strategy} | Alpha={alpha}, Beta={beta}, Tau={tau}] \n {metrics}")
        
        log_file_path = os.path.join(self.output_dir, f'metric_{self.args.kge_model_name}_a{alpha}_b{beta}_tau{tau}.txt')
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            log_line = f'ranking metrics: {metrics}\n'
            log_file.write(log_line)

    @staticmethod
    def read_triple(file_path, entity2id, relation2id):
        '''
        Read triples and map them into ids.
        '''
        triples = []
        with open(file_path) as fin:
            for line in fin:
                h, r, t = line.strip().split('\t')
                triples.append((entity2id[h], relation2id[r], entity2id[t]))
        return triples

def run_parallel_experiments(args):
    print("🚀 [Auto Run Mode] 모든 데이터셋과 KGE 모델 조합에 대해 실험을 자동 시작합니다!")
    datasets = ["wn18rr", "fb15k237"]
    #kge_models = ["TransE", "RotatE"]
    kge_models = ["RotatE"]
    score_strategy = "only_llm"
    score_strategy = "modify_query" # 적용하고 싶은 방법론 이름
    #score_strategy = "weighted_sum"
    available_gpus = [0,1,2,3]
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    num_gpus = len(available_gpus)
    processes = []
    open_files = []
    process_idx = 0
    for dataset in datasets:
        for kge_model in kge_models:
            
            # 1. 데이터셋에 따른 경로 맵핑 (FB15k-237의 폴더명 차이 대응)
            if dataset == "fb15k237":
                data_path = "KG_data/fb15k-237"
                test_json_path = "dataset_merged/fb15k237/test.json"
                ckpt_suffix = "FB15k-237_0"
                checkpoint_dir = "results/wn18rr/llama3/checkpoint-final"
                kge_embedding_path = "dataset/wn18rr/entity_embeddings.pt"
                gamma = 9.0
            else: # wn18rr
                data_path = "KG_data/wn18rr"
                test_json_path = "dataset_merged/wn18rr/test.json"
                ckpt_suffix = "wn18rr_0"
                checkpoint_dir = "results/fb15k237/llama3/checkpoint-final"
                kge_embedding_path = "dataset/fb15k237/entity_embeddings.pt"
                gamma = 6.0

            # 2. KGE 모델에 따른 체크포인트 경로 맵핑
            if kge_model == "TransE":
                kge_checkpoint = f"TransE_{ckpt_suffix}/checkpoint"
            else: # RotatE
                kge_checkpoint = f"RotatE/checkpoints/RotatE_{ckpt_suffix}/checkpoint"
                
            # 3. Logits 캐시 파일명 맵핑 (wn 또는 fb)
            prefix = dataset[:2] 
            logits_dir = f"results/{dataset}/llama3"
            logits_path = f"{logits_dir}/{prefix}_logits_w_gnn.pt"
            
            # 4. 터미널 명령어(Command) 리스트 구성
            cmd = [
                "python", "fast_infer.py",
                "--data_path", data_path,
                "--test_json_path", test_json_path,
                "--kge_checkpoint", kge_checkpoint,
                "--logits_path", logits_path,
                "--kge_model_name", kge_model,
                "--score_strategy", score_strategy,
                "--gamma", str(gamma)
            ]
            
            if args.use_llm:
                cmd.append("--use_llm")
                cmd.extend(["--kge_embedding_path", kge_embedding_path])
                cmd.extend(["--checkpoint_dir", checkpoint_dir])

            if args.divide_length:
                cmd.append("--divide_length")

            # 리스트를 띄어쓰기로 연결하여 하나의 문자열 명령어로 만듦
            cmd_str = " ".join(cmd)
            target_gpu = available_gpus[process_idx % num_gpus]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(target_gpu)
            log_filename = os.path.join(log_dir,f"{dataset}_{kge_model}_{score_strategy}.log")
            f_out = open(log_filename, "w", encoding="utf-8")
            open_files.append(f_out)
            print(f"▶️ [병렬 실행 출발] Dataset = {dataset.upper():<8} | KGE = {kge_model:<6} | 🖥️ GPU = {target_gpu}")                
            p = subprocess.Popen(cmd_str, shell=True, env=env, stdout=f_out, stderr=subprocess.STDOUT)
            processes.append(p)
            process_idx += 1 
        
    for p in processes:
        p.wait()
    for f in open_files:
        f.close()
    print("\n🎉 모든 병렬 자동화 실험이 성공적으로 완료되었습니다!")

def main():
    parser = argparse.ArgumentParser(description="Fast Evaluation Script for KGC Align Model")
    parser.add_argument("--auto_run", action="store_true", help="이 플래그를 넣으면 다중 GPU 병렬 모드가 실행됩니다.")
    parser.add_argument("--data_path", type=str, help="KG_data 경로")
    parser.add_argument("--test_json_path", type=str, help="테스트셋(test.json) 파일 경로")
    parser.add_argument("--kge_checkpoint", type=str, help="사전 학습된 KGE 모델 체크포인트 경로")
    parser.add_argument("--logits_path", type=str, help="미리 추출한 LLM Raw Logits 파일 경로")
    parser.add_argument("--kge_model_name", type=str, choices=["TransE", "RotatE", "transe", "rotate"])
    
    # 🌟 [추가] 튜닝 변수 및 전략 선택 인자
    parser.add_argument("--gamma", type=float, default=9.0)
    parser.add_argument("--score_strategy", type=str, default="weighted_sum", 
                        help="적용할 점수 계산 전략 이름 (예: weighted_sum, modify_query)")
    # 🌟 [수정 4] 누락된 Argument 추가
    parser.add_argument("--use_llm", action="store_true", help="LLM을 로드하여 실제 Greedy Decoding과 비교")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="DrKGC Graph Enhancer 체크포인트")
    parser.add_argument("--kge_embedding_path", type=str, default=None, help="사전학습된 KGE 임베딩 pt 파일")

    parser.add_argument("--divide_length", action="store_true", help="llm_logits를 length로 나눠줄지 여부")

    args = parser.parse_args()
    #breakpoint()
    if args.auto_run:
        run_parallel_experiments(args)
        return
    
    if not args.data_path or not args.test_json_path:
        parser.error("단일 실행 모드에서는 --data_path 등이 필요합니다. 병렬 실행을 원하시면 '--auto_run' 플래그를 추가하세요.")
    evaluator = Evaluator(args)
    #evaluator.compare_greedy_vs_global() ## debugging 
    #return ## debugging
    if args.use_llm:
        dataset_name_prefix = os.path.basename(args.test_json_path)[:2]
        cache_file_name = f"{dataset_name_prefix}_logits_fractal.pt"
        with autocast():
            evaluator.save_llm_confidence(cache_file_name)
        
    alpha_beta_list = [{"alpha": 0.0, "beta": 0.0}]
    beta_list = [0.001, 0.01, 0.05, 0.1]
    for beta in beta_list:
        alpha_beta_list.append({"alpha": -1.0, "beta": beta})
    #alpha_beta_list = [{"alpha": 0.0, "beta": 0.0}] ## for debugging 
    tau_list = [1.0]
    for alpha_beta in alpha_beta_list:
        for tau in tau_list:
            evaluator.ranking_metrics(alpha_beta, tau)

if __name__ == '__main__':
    main()
    