import os
import json
import numpy as np
from time import time
from tqdm import trange, tqdm
import argparse
from pathlib import Path
import sys
import bitsandbytes as bnb
import torch

import transformers
from transformers import AutoConfig,  GenerationConfig
from transformers import AutoTokenizer, LlamaTokenizer, PreTrainedTokenizer
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, HfArgumentParser
from transformers import set_seed, Seq2SeqTrainer, BitsAndBytesConfig

from peft.tuners.lora import LoraLayer
from peft import LoraConfig, get_peft_model, PeftModelForCausalLM, prepare_model_for_kbit_training, PeftModel

from arguments import Arguments, FinetuningArguments, GenerationArguments
from data import DataModule, QueryCollator
from model import GraphEnhancer, DrKGC, DrKGC_enhanced, KG_enhanced, DrKGC_align, KG_align

from torch.cuda.amp import autocast

import torch
torch.cuda.empty_cache()

import wandb
from dotenv import load_dotenv
import collections
load_dotenv()


DATASET_METADATA = {
    "fb15k237": {"E_dim": 14541, "R_dim": 237, "kgc_loss_weight":0.005},
    "wn18rr": {"E_dim": 40943, "R_dim": 11, "kgc_loss_weight":0.01},
}
KGE_MODEL={"fb15k237":{"R_dim":1000,"gamma":9.0}, "wn18rr":{"R_dim":500,"gamma":6.0}}

class Evaluator:
    def __init__(self, args, tokenizer, model, data_module, generation_config):
        self.args = args
        self.generation_config = generation_config

        self.tokenizer = tokenizer
        self.model = model
        self.data_module = data_module
        if args.llm_confidence:
            self.output_dir = args.exp_output_dir 
        else:
            self.output_dir = os.path.dirname(args.checkpoint_dir)
        if args.use_align:
            alpha_val = getattr(args, 'alpha', -1.0)
            beta_val = getattr(args, 'beta', 0.0)
            if alpha_val in [0.0, 1.0]:
                file_suffix = f"_a{alpha_val}"
            else:
                file_suffix = f"_b{beta_val}"
            self.log_file_path = os.path.join(self.output_dir, f'metrics{file_suffix}.txt')
            file_path = os.path.join(args.dataset_path,'id2entity.json')
            with open(file_path, 'r', encoding='utf-8') as f:
                self.id2entity = {int(k): v for k, v in json.load(f).items()}
            # all_true_triple 만들기... 
            with open(os.path.join(args.data_path, 'entities.dict')) as fin:
                entity2id = dict()
                self.kge_id2entity = dict()
                for line in fin:
                    eid, entity = line.strip().split('\t')
                    entity2id[entity] = int(eid)
                    self.kge_id2entity[int(eid)] = entity
            with open(os.path.join(args.data_path, 'relations.dict')) as fin:
                relation2id = dict()
                for line in fin:
                    rid, relation = line.strip().split('\t')
                    relation2id[relation] = int(rid)
            self.train_triples = self.read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
            self.valid_triples = self.read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
            self.test_triples = self.read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
            self.all_true_triples = self.train_triples + self.valid_triples + self.test_triples
            self.hr2t = collections.defaultdict(set)
            self.tr2h = collections.defaultdict(set)
            for h, r, t in self.all_true_triples:
                self.hr2t[(h, r)].add(t)
                self.tr2h[(t, r)].add(h)
        else:
            self.log_file_path = os.path.join(self.output_dir, 'metrics.txt')

        self.llm_conf_probs = None
        if args.llm_confidence and hasattr(args, 'exp_output_dir'):
            dataset_name_prefix = os.path.basename(args.dataset_path)[:2]
            conf_probs_file_name = f"{dataset_name_prefix}_{args.kge_model_name}_logits_merged.pt"
            conf_probs_path = os.path.join(args.exp_output_dir, conf_probs_file_name)
            if os.path.exists(conf_probs_path):
                print(f"📦 [llm_conf_probs] 기존에 생성된 llm_conf_probs를 로드합니다: {conf_probs_path}")
                self.llm_conf_probs = torch.load(conf_probs_path, map_location='cpu')

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

    @torch.no_grad()
    def save_llm_confidence(self, dataset, file_name):
        #breakpoint()
        if not self.args.llm_confidence:
            raise ValueError("self.args.llm_confidence is False!")
        self.model.eval()
        confidence_cache = {}
        for ex_idx, ex in enumerate(tqdm(dataset)):
            h, r, t = ex['triplet_id']
            pred_type = ex.get('type').split('_')[-1] # 'head' | 'tail'
            query_id = h if pred_type=='tail' else t
            subgraph = [ex['subgraph']] if 'subgraph' in ex else None
            llm_conf_probs = self.model.compute_llm_confidence( 
                prompt=ex['input'], 
                candidates=ex['rank_entities'], 
                query_ids=torch.LongTensor([ex['query_entity_id']]).to(input_ids.device), 
                entity_ids=torch.LongTensor([ex['rank_entities_id']]).to(input_ids.device), 
                subgraph=subgraph, 
                # query_ids=torch.LongTensor([query_id]).cuda(), 
                # entity_ids=torch.LongTensor([ex['topk_id']]).cuda() 
            ) 
            h, r, t = ex['triplet_id']
            pred_type = ex.get('type').split('_')[-1] # 'head' | 'tail'
            query_key = f"{ex_idx}_{h}_{r}_{t}_{pred_type}"
            confidence_cache[query_key] = llm_conf_probs.cpu()
        save_path = os.path.join(self.output_dir, file_name)
        torch.save(confidence_cache, save_path)
        print(f"\n✅ [성공] LLM Confidence 캐시가 다음 경로에 저장되었습니다: {save_path}")

    @torch.no_grad()
    def ranking_metrics(self, dataset):
        self.model.eval()

        preds = []
        logs = []
        ranks = np.array([])
        #breakpoint()
        debug_log_dict={}
        generated = []
        for ex_idx, ex in enumerate(tqdm(dataset)):
            prompt = ex['input']
            llm_conf_probs = None
            h, r, t = ex['triplet_id']
            pred_type = ex.get('type').split('_')[-1] # 'head' | 'tail'
            query_id = h if pred_type=='tail' else t
            query_key = f"{ex_idx}_{h}_{r}_{t}_{pred_type}"
            if self.args.llm_confidence:
                if self.llm_conf_probs is not None and query_key in self.llm_conf_probs:
                    llm_conf_probs = self.llm_conf_probs[query_key].cuda()
                else:
                    llm_conf_probs = self.model.compute_llm_confidence( 
                        prompt=prompt, 
                        candidates=ex['rank_entities'], 
                        query_ids=torch.LongTensor([query_id]).cuda(), #torch.LongTensor([ex['query_entity_id']]).cuda(), 
                        entity_ids=torch.LongTensor([ex['topk_id']]).cuda()#torch.LongTensor([ex['rank_entities_id']]).cuda()
                    ) 
            inputs = self.tokenizer(prompt, return_tensors='pt')
            input_ids = inputs.input_ids.cuda() 
            self.generation_config.eos_token_id = self.tokenizer.eos_token_id 
            subgraph = [ex['subgraph']] if 'subgraph' in ex else None
            if self.args.use_enhanced or self.args.use_align:
                attention_mask = inputs.attention_mask.cuda()
                triple_ids = torch.LongTensor([ex['triple_id']]).cuda() 
                is_predicted_tail = torch.BoolTensor([ex['type']=='predicted_tail']).cuda()
                if 'triplet_id' in ex.keys():
                    triplet_ids = torch.LongTensor([ex['triplet_id']]).cuda() 
                    topk_ids = torch.LongTensor([ex['topk_id']]).cuda() 
                else:
                    triplet_ids = triple_ids
                    topk_ids = torch.LongTensor([ex['rank_entities_id']]).to(input_ids.device)
                output = self.model.generate(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,#
                    query_ids=torch.LongTensor([query_id]).cuda(), #torch.LongTensor([ex['query_entity_id']]).to(input_ids.device), 
                    entity_ids=torch.LongTensor([ex['topk_id']]).cuda(), #torch.LongTensor([ex['rank_entities_id']]).to(input_ids.device), 
                    triple_ids=triple_ids,#
                    is_predicted_tail=is_predicted_tail,#
                    subgraph=subgraph, 
                    generation_config=self.generation_config,
                    triplet_ids = triplet_ids,
                    topk_ids = topk_ids,
                    llm_conf_probs = llm_conf_probs
                ) # outputs.keys() 'sequences' 'past_key_values'
            else:
                output = self.model.generate(
                    input_ids=input_ids, 
                    query_ids=torch.LongTensor([ex['query_entity_id']]).to(input_ids.device), 
                    entity_ids=torch.LongTensor([ex['rank_entities_id']]).to(input_ids.device), 
                    subgraph=subgraph, 
                    generation_config=self.generation_config,
                )
            ## align
            if self.args.use_align: 
                #breakpoint()
                h,r,t = ex['triplet_id']
                pred_type = ex.get('type')
                mode_str = "tail-batch" if pred_type == 'predicted_tail' else "head-batch"
                query_key = f"{h}_{r}_{t}_{mode_str}"
                target_id = t if pred_type == 'predicted_tail' else h
                target_score = output[0, target_id].clone()
                if pred_type == 'predicted_tail':
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
                target_score_val = output[0, target_id].item()
                # # A. 모든 쿼리 성적표를 JSON 딕셔너리에 저장
                # debug_log_dict[query_key] = {
                #     "rank": ranking,
                #     "target_score": target_score_val,
                #     "top10_preds": argsort[0, :10].cpu().tolist()
                # }
                
                # # B. 특정 쿼리의 전체 점수 텐서 추출 (.npy)
                # # 추출하신 RotatE 텐서와 동일한 쿼리를 지정합니다.
                # if h == 1678 and r == 7 and t == 9202:
                #     np.save(f"drkgc_tensor_{query_key}.npy", output[0].cpu().numpy())
                #     print(f"\n🚨 [디버그] {query_key}의 DrKGC 전체 점수 텐서 저장 완료!")
                top10_ids = argsort[0, :10].cpu().tolist()
                top10_preds = [self.kge_id2entity.get(eid, f"[UNKNOWN_ID_{eid}]") for eid in top10_ids]
                top10_scores = [round(output[0, eid].item(), 4) for eid in top10_ids]
                
                ex['target'] = ex.get('output', '')
                ex['pred_rank'] = ranking
                ex['target_score'] = round(target_score.item(), 4) # 정답의 실제 점수
                
                ex['pred'] = top10_preds[0] # 기존 호환성을 위해 Top-1 유지
                ex['top10_preds'] = top10_preds # 1등부터 10등까지의 엔티티 이름 리스트
                ex['top10_scores'] = top10_scores
                if 'input' in ex: ex.pop('input')
                preds.append(ex)
                #generated.append(top1_id)
            else:
                generated.append(output.sequences[0].cpu().numpy().tolist())
                ex.pop('input') # 'input' 키를 삭제하면서 그 value 반환 

        if self.args.use_align:
            metrics = {}
            if len(logs) > 0:
                for metric in logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
            metrics = {k: round(v, 8) for k, v in metrics.items()}
            metrics = {
                'mrr': metrics['MRR'], 'mr': metrics['MR'],
                'hits1': metrics['HITS@1'], 'hits3': metrics['HITS@3'], 'hits10': metrics['HITS@10']
            }
        else:
            batch_preds = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            for ex_idx, ex in enumerate(dataset):
                target = ex.pop('output')
                rank = ex['rank']
                pred = str(batch_preds[ex_idx]).strip()

                topk_names = ex['rank_entities']
                if target == pred:
                    rank = 1
                else:    
                    if pred not in set(topk_names) or topk_names.index(pred) >= rank:
                        rank += 1
                
                ex['target'] = target
                ex['pred_rank'] = rank
                ex['pred'] = pred
                preds.append(ex)
                ranks = np.append(ranks, rank)
            
            metrics = {
            'mrr': np.mean(1. / ranks),
            'hits1': np.mean(ranks <= 1),
            'hits3': np.mean(ranks <= 3),
            'hits10': np.mean(ranks <= 10),
            }
            metrics = {k: round(v, 8) for k, v in metrics.items()}
        
        print("ranking metrics:")
        print(metrics)

        # if self.args.use_align:
        #     import json
        #     with open('drkgc_debug_log.json', 'w', encoding='utf-8') as f:
        #         json.dump(debug_log_dict, f, indent=4)
        #     print("\n✅ 모든 쿼리의 순위/Top10 결과를 'drkgc_debug_log.json'에 저장했습니다.")

        with open(self.log_file_path, 'w', encoding='utf-8') as log_file:
            log_line = f'ranking metrics: {metrics}\n'
            log_file.write(log_line)
        if args.use_wandb:
            wandb.log(metrics)
        return preds


if __name__ == '__main__':
    #set_seed(3407)
    
    hfparser = HfArgumentParser((Arguments, FinetuningArguments, GenerationArguments))
    (data_args, finetune_args, generation_args, _) = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    generation_config = GenerationConfig(**vars(generation_args))
    combined_args = {**vars(data_args), **vars(finetune_args)}
    args = argparse.Namespace(**combined_args)
    set_seed(args.seed_num)
    if args.use_wandb:
        wandb_api_key = os.environ.get("WANDB_API_KEY")
        wandb.login(key=wandb_api_key)
        wandb.init(
            project="DrKGC-Experiments-Align", 
            name=args.run_name, # 예: Eval-checkpoint-final
            config=vars(args) # 하이퍼파라미터도 같이 저장
        )
    print(f"Load LLM: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(['[QUERY]', '[ENTITY]', '[RELATION]'])
    if hasattr(args, 'use_extract') and args.use_extract and args.new_token: 
        tokenizer.add_tokens(['<|extract_kg|>'])

    generation_config.bos_token_id = tokenizer.bos_token_id
    generation_config.pad_token_id = tokenizer.pad_token_id
    model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, low_cpu_mem_usage=True, device_map='auto')
    if hasattr(args, 'use_extract') and args.use_extract and args.new_token:
        model.resize_token_embeddings(len(tokenizer))
    if args.llm_freeze:
        model = PeftModel.from_pretrained(model, args.peft_model_path)
    else:
        model = PeftModel.from_pretrained(model, args.checkpoint_dir)

    model = model.half()
    
    loaded_data = torch.load(args.kge_embedding_path, map_location='cpu')
    if isinstance(loaded_data, dict) and 'model_state_dict' in loaded_data:
        kge_embedding = loaded_data['model_state_dict']['entity_embedding']
    else:
        kge_embedding = loaded_data
    kge_embedding_dim = kge_embedding.shape[1]
    llm_config = model.config
    embed_model = GraphEnhancer(kge_embedding, kge_embedding_dim, 4, 128, 1, 1024, llm_config.hidden_size, llm_config.hidden_act)
    ckpt_dir = Path(args.checkpoint_dir)  
    graph_model_path = ckpt_dir / "graph_model.bin"
    if graph_model_path.exists():
        state = torch.load(ckpt_dir / "graph_model.bin", map_location="cpu")
        embed_model.load_state_dict(state)
        print("✅ loaded graph_model.bin")
    else:
        print(f"\n❌ [CRITICAL ERROR] '{graph_model_path}' 파일을 찾을 수 없습니다!")
        print(f"   학습된 graph_model.bin이 없으면 추론 시 무작위 가중치를 사용하여 성능이 붕괴됩니다.")
        print(f"   checkpoint_dir 경로가 올바른지 확인해주세요: {args.checkpoint_dir}")
        sys.exit(1) # 1은 에러로 종료됨을 의미
    #breakpoint()
    dataset_name = os.path.basename(args.dataset_path)
    if dataset_name not in DATASET_METADATA:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {list(DATASET_METADATA.keys())}")
    E_dim = DATASET_METADATA[dataset_name]["E_dim"]
    R_dim = DATASET_METADATA[dataset_name]["R_dim"]
    kgc_loss_weight = DATASET_METADATA[dataset_name]["kgc_loss_weight"]
    if dataset_name not in KGE_MODEL:
        raise ValueError(f"Unsupported KGE model: {args.kge_model_name}. Supported models: {list(KGE_MODEL.keys())}")
    R_hidden = KGE_MODEL[dataset_name]['R_dim']
    gamma = KGE_MODEL[dataset_name]['gamma']
    if args.use_enhanced:
        enhanced_model = KG_enhanced(E_dim, R_dim, model.config.hidden_size, args.rand_neg, KGE_model_name=args.kge_model_name, R_dim=R_hidden, gamma=gamma)
        enhanced_state = torch.load(ckpt_dir / "enhanced_model.bin", map_location="cpu")
        enhanced_model.load_state_dict(enhanced_state)
        enhanced_model.cuda()
        model = DrKGC_enhanced(tokenizer,model,embed_model,enhanced_model,kgc_loss_weight)
    elif args.use_align:
        print("⚠️ [DEBUG MODE] 순수 KGE 체크포인트를 로드하여 순위 계산을 테스트합니다.")
        
        if hasattr(args, 'checkpoint_path') and args.checkpoint_path is not None:
            kge_state_dict = torch.load(args.checkpoint_path, map_location="cpu")
            pretrained_ent = kge_state_dict['model_state_dict']['entity_embedding']
            pretrained_rel = kge_state_dict['model_state_dict']['relation_embedding']
            
            align_model = KG_align(E_dim, R_dim, model.config.hidden_size, args.rand_neg, args.alpha, args.beta, args.use_d_r, 
                                   pretrained_ent=pretrained_ent, pretrained_rel=pretrained_rel, 
                                   KGE_model_name=args.kge_model_name, R_dim=R_hidden, gamma=gamma)
        # align_model = KG_align(E_dim, R_dim, model.config.hidden_size, args.rand_neg, args.alpha, args.beta, args.use_d_r, KGE_model_name=args.kge_model_name, R_dim=R_hidden, gamma=gamma)
        # align_state = torch.load(ckpt_dir / "align_model.bin", map_location="cpu")
        # align_model.load_state_dict(align_state)
        align_model.cuda()
        model = DrKGC_align(tokenizer, model, embed_model, align_model, args.lm_loss, args.kge_loss, args.struct_loss, args.align_loss, args.tau)
    else:
        model = DrKGC(tokenizer, model, embed_model)
    if hasattr(model, 'llm_model'):
        model.llm_model = model.llm_model.half()
    if hasattr(model, 'graph_model'):
        model.graph_model = model.graph_model.half()
    #model = model.half()
    model.cuda()
    model.eval()

    data_module = DataModule(args, tokenizer)

    evaluator = Evaluator(args, tokenizer, model, data_module, generation_config)
    
    # # 🌟 [신규 추가] LLM Confidence만 계산하여 캐시로 저장하는 로직
    # if args.use_align and args.llm_confidence:
    #     os.makedirs(args.exp_output_dir, exist_ok=True)
    #     dataset_name_prefix = os.path.basename(args.dataset_path)[:2]
    #     cache_file_name = f"{dataset_name_prefix}_{args.kge_model_name}_logits.pt"
    #     with autocast():
    #         evaluator.save_llm_confidence(data_module.test_ds, cache_file_name)

    #=== [기존 평가 및 저장 코드 주석 처리 시작] ===
    with autocast():
        preds = evaluator.ranking_metrics(data_module.test_ds)
    output = {
        'args': vars(args),
        'generation_config': vars(generation_config),
        'prediction': preds,
    }
    if args.use_align:
        alpha_val = getattr(args, 'alpha', -1.0)
        beta_val = getattr(args, 'beta', 0.0)
        if alpha_val in [0.0, 1.0]:
            file_suffix = f"_a{alpha_val}"
        else:
            file_suffix = f"_b{beta_val}"
        if args.llm_confidence:
            output_path = os.path.join(args.exp_output_dir, f'prediction{file_suffix}.json')
        else:
            output_path = os.path.join(os.path.dirname(args.checkpoint_dir), f'prediction{file_suffix}.json')
        #json.dump(output, open(output_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4, default=str)
    else:
        output_path = os.path.join(os.path.dirname(args.checkpoint_dir), f'prediction.json')
        json.dump(output, open(output_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4, default=str)
    
    if args.use_wandb:
        wandb.finish()