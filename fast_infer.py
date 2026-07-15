import argparse
import json
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

    def ranking_metrics(self, alpha_beta, tau):
        logs = []
        alpha = alpha_beta['alpha']
        beta = alpha_beta['beta']
        for ex_idx, ex in enumerate(self.dataset):
            h, r, t = ex['triplet_id']
            pred_type = ex.get('type').split('_')[-1] # 'head' | 'tail'
            query_key = f"{ex_idx}_{h}_{r}_{t}_{pred_type}"
            llm_logits = self.llm_logits_cache[query_key]
            output = self.scorer.calculate_scores(
                triple_ids=torch.LongTensor([ex['triple_id']]).cuda() , 
                entity_ids=torch.LongTensor([ex['rank_entities_id']]).cuda(), 
                llm_logits=llm_logits,
                is_predicted_tail=pred_type, 
                alpha=alpha, 
                beta=beta, 
                tau=tau
            )
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


def main():
    parser = argparse.ArgumentParser(description="Fast Evaluation Script for KGC Align Model")
    parser.add_argument("--data_path", type=str, required=True, help="KG_data 경로")
    parser.add_argument("--test_json_path", type=str, required=True, help="테스트셋(test.json) 파일 경로")
    parser.add_argument("--kge_checkpoint", type=str, required=True, help="사전 학습된 KGE 모델 체크포인트 경로")
    parser.add_argument("--logits_path", type=str, required=True, help="미리 추출한 LLM Raw Logits 파일 경로")
    parser.add_argument("--kge_model_name", type=str, required=True, choices=["TransE", "RotatE", "transe", "rotate"])
    
    # 🌟 [추가] 튜닝 변수 및 전략 선택 인자
    parser.add_argument("--gamma", type=float, default=9.0)
    parser.add_argument("--score_strategy", type=str, default="weighted_sum", 
                        help="적용할 점수 계산 전략 이름 (예: weighted_sum, modify_query)")

    args = parser.parse_args()
    #breakpoint()
    evaluator = Evaluator(args)

    alpha_beta_list = [{"alpha": 0.0, "beta": 0.0}]
    beta_list = [0.001, 0.01, 0.05, 0.1]
    for beta in beta_list:
        alpha_beta_list.append({"alpha": -1.0, "beta": beta})
    #alpha_beta_list = [{"alpha": -1.0, "beta": 0.01}] ## for debugging 
    tau_list = [1.0]
    for alpha_beta in alpha_beta_list:
        for tau in tau_list:
            evaluator.ranking_metrics(alpha_beta, tau)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main()
    else:
        print("🚀 [Auto Run Mode] 모든 데이터셋과 KGE 모델 조합에 대해 실험을 자동 시작합니다!")
        
        datasets = ["wn18rr", "fb15k237"]
        kge_models = ["TransE", "RotatE"]
        #score_strategy = "modify_query" # 적용하고 싶은 방법론 이름
        score_strategy = "candidate_order_kge" # 적용하고 싶은 방법론 이름
        processes = []
        for dataset in datasets:
            for kge_model in kge_models:
                
                # 1. 데이터셋에 따른 경로 맵핑 (FB15k-237의 폴더명 차이 대응)
                if dataset == "fb15k237":
                    data_path = "KG_data/fb15k-237"
                    test_json_path = "dataset/fb15k237/test.json"
                    ckpt_suffix = "FB15k-237_0"
                    gamma = 9.0
                else: # wn18rr
                    data_path = "KG_data/wn18rr"
                    test_json_path = "dataset/wn18rr/test.json"
                    ckpt_suffix = "wn18rr_0"
                    gamma = 6.0

                # 2. KGE 모델에 따른 체크포인트 경로 맵핑
                if kge_model == "TransE":
                    kge_checkpoint = f"TransE_{ckpt_suffix}/checkpoint"
                else: # RotatE
                    kge_checkpoint = f"RotatE/checkpoints/RotatE_{ckpt_suffix}/checkpoint"
                    
                # 3. Logits 캐시 파일명 맵핑 (wn 또는 fb)
                prefix = dataset[:2] 
                logits_dir = f"results/{dataset}/llama3_seed1213_{kge_model}_kgt5"
                logits_path = f"{logits_dir}/{prefix}_{kge_model}_logits.pt"
                
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
                
                # 리스트를 띄어쓰기로 연결하여 하나의 문자열 명령어로 만듦
                cmd_str = " ".join(cmd)
                print(f"▶️ [병렬 실행 출발] Dataset = {dataset.upper()} | KGE = {kge_model}")
                p = subprocess.Popen(cmd_str, shell=True)
                processes.append(p)
            
        for p in processes:
            p.wait()
            
        print("\n🎉 모든 병렬 자동화 실험이 성공적으로 완료되었습니다!")
                # try:
                #     # subprocess를 통해 독립된 프로세스로 실행 (메모리 누수 원천 차단)
                #     subprocess.run(cmd_str, shell=True, check=True)
                # except subprocess.CalledProcessError as e:
                #     print(f"\n❌ [에러 발생] {dataset} - {kge_model} 실행 중 오류가 발생하여 다음 실험으로 넘어갑니다.")
                #     continue