from dataclasses import dataclass
from typing import Sequence, Dict, Any

import torch
from torch.nn.utils.rnn import pad_sequence
import transformers
from .dataset import DataModule 
import os

@dataclass
class QueryCollator:
    args: None
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        bos_id, eos_id = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id

        sources = [ex["input"] for ex in instances]
        targets = [ex["output"] for ex in instances]

        src_max = max(1, self.source_max_len - (1 if bos_id is not None else 0))
        tgt_max = max(1, self.target_max_len - (1 if eos_id is not None else 0))

        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=src_max,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=tgt_max,
            truncation=True,
            add_special_tokens=False,
        )

        source_input_ids = tokenized_sources_with_prompt['input_ids']
        target_input_ids = tokenized_targets['input_ids']

        input_ids = []
        labels = []
        for src_ids, tgt_ids in zip(source_input_ids, target_input_ids):
            seq = [bos_id] + src_ids + tgt_ids + [eos_id]
            input_ids.append(torch.tensor(seq, dtype=torch.long))
            lab = torch.full((len(seq),), -100, dtype=torch.long)
            start = len(src_ids) + 1
            lab[start:] = torch.tensor(tgt_ids + [eos_id], dtype=torch.long)
            labels.append(lab)

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        query_ids = torch.tensor([ex['query_entity_id'] for ex in instances], dtype=torch.long)
        entity_ids = torch.tensor([ex['rank_entities_id'] for ex in instances], dtype=torch.long) ##이게 candidate
        subgraph = [ex['subgraph'] for ex in instances]
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': (input_ids != self.tokenizer.pad_token_id).long(),
            'labels': labels,
            "query_ids": query_ids,
            "entity_ids": entity_ids,
            "subgraph": subgraph,
        }

        return data_dict

class QueryCollator_extract(QueryCollator):
    def __init__(self, args, tokenizer, source_max_len, target_max_len):
        super().__init__(args=args, tokenizer=tokenizer, source_max_len=source_max_len, target_max_len=target_max_len)
        self.new_token = args.new_token
        if args.new_token:
            self.extract_id = self.tokenizer.convert_tokens_to_ids('<|extract_kg|>') #int
        ## all_true_triples 세팅
        with open(os.path.join(args.data_path, 'entities.dict')) as fin:
            entity2id = dict()
            for line in fin:
                eid, entity = line.strip().split('\t')
                entity2id[entity] = int(eid)

        with open(os.path.join(args.data_path, 'relations.dict')) as fin:
            relation2id = dict()
            for line in fin:
                rid, relation = line.strip().split('\t')
                relation2id[relation] = int(rid)
        self.train_triples = self.read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
        self.valid_triples = self.read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
        self.test_triples = self.read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
        self.all_true_triples = self.train_triples + self.valid_triples + self.test_triples
        self.count = self.count_frequency(self.all_true_triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.all_true_triples)

    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        bos_id, eos_id = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id

        sources = [ex["input"] for ex in instances]
        targets = [ex["output"] for ex in instances]

        src_max = max(1, self.source_max_len - (1 if bos_id is not None else 0))
        tgt_max = max(1, self.target_max_len - (1 if eos_id is not None else 0))

        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=src_max,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=tgt_max,
            truncation=True,
            add_special_tokens=False,
        )

        source_input_ids = tokenized_sources_with_prompt['input_ids']
        target_input_ids = tokenized_targets['input_ids']

        input_ids = []
        labels = []
        is_predicted_tail = []
        extract_positions = []
        for ex, src_ids, tgt_ids in zip(instances, source_input_ids, target_input_ids): # instances는 batch size 크기의 리스트 각 원소는 딕셔너리 
            # 1. 문장 맨 끝에 extract_id 추가
            if self.new_token:
                seq = [bos_id] + src_ids + [self.extract_id] + tgt_ids + [eos_id] # 
                extract_idx = len(src_ids) + 1 # 아니 이러면...다음토큰정답 정하기 어려워지는데 우선 이건 고려하지 말자
            else:
                seq = [bos_id] + src_ids + tgt_ids + [eos_id]
                extract_idx = len(src_ids)
            input_ids.append(torch.tensor(seq, dtype=torch.long))
            extract_positions.append(extract_idx)
            # 2. 라벨 마스킹: 마지막 extract_id는 -100이 되도록 유지
            lab = torch.full((len(seq),), -100, dtype=torch.long)
            start = len(src_ids) + 1
            if self.new_token:
                lab[start:-1] = torch.tensor(tgt_ids + [eos_id], dtype=torch.long)
            else:
                lab[start:] = torch.tensor(tgt_ids + [eos_id], dtype=torch.long)
            labels.append(lab)
            is_predicted_tail.append(ex['type']=='predicted_tail')
        ## rand_entity_ids_batch, subsamplig_weight_batch 만들기 
        rand_entity_ids_batch = []
        subsampling_weight_batch = []

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        query_ids = torch.tensor([ex['query_entity_id'] for ex in instances], dtype=torch.long)
        entity_ids = torch.tensor([ex['rank_entities_id'] for ex in instances], dtype=torch.long) # candidate(20개로 고정됨 )
        subgraph = [ex['subgraph'] for ex in instances]
        triple_ids = torch.tensor([ex['triple_id'] for ex in instances], dtype=torch.long)
        is_predicted_tail = torch.tensor(is_predicted_tail, dtype=torch.bool)
        extract_positions = torch.tensor(extract_positions, dtype=torch.long)
        if 'triplet_id' in instances[0]:
            triplet_ids = torch.tensor([ex['triplet_id'] for ex in instances], dtype=torch.long)
        else:
            triplet_ids = triple_ids
        if 'topk_id' in instances[0]:
            topk_ids = torch.tensor([ex['topk_id'] for ex in instances], dtype=torch.long)
        else:
            topk_ids = entity_ids
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': (input_ids != self.tokenizer.pad_token_id).long(),
            'labels': labels,
            "query_ids": query_ids,
            "entity_ids": entity_ids,
            "subgraph": subgraph,
            "triple_ids": triple_ids,
            "is_predicted_tail": is_predicted_tail,
            'extract_positions': extract_positions,
            "triplet_ids": triplet_ids, 
            "topk_ids": topk_ids,
            #"rand_entity_ids": torch.stack(rand_entity_ids_batch, dim=0),
            #"subsamplig_weight": torch.FloatTensor(subsampling_weight_batch)
        }

        return data_dict


def make_data_module(args, tokenizer: transformers.PreTrainedTokenizer):
    data_module = DataModule(args, tokenizer)
    data_collator = QueryCollator(
        args=args, tokenizer=tokenizer, 
        source_max_len=args.source_max_len, target_max_len=args.target_max_len
    )

    return {
        'train_dataset': data_module.train_ds,
        'eval_dataset': data_module.eval_ds,
        'data_collator': data_collator,
    }

def make_data_module_extract(args, tokenizer: transformers.PreTrainedTokenizer):
    data_module = DataModule(args, tokenizer)
    data_collator = QueryCollator_extract(
        args=args, tokenizer=tokenizer, 
        source_max_len=args.source_max_len, target_max_len=args.target_max_len
    )

    return {
        'train_dataset': data_module.train_ds,
        'eval_dataset': data_module.eval_ds,
        'data_collator': data_collator,
    }

