import torch
from torch import nn
import json
import argparse
import collections

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)

args = parser.parse_args()

data_configs = {
    "wn18rr": {
        "drkgc_id2entity_path": "dataset/wn18rr/id2entity.json",
        "drkgc_id2relation_path": "dataset/wn18rr/id2relation.json",
        "transe_entity2text_path": "KG_data/wn18rr/entity2text.txt",
        "transe_entities_dict_path": "KG_data/wn18rr/entities.dict",
        "transe_relations_dict_path": "KG_data/wn18rr/relations.dict",
        "original_ckpt": "TransE_wn18rr_0/checkpoint",
        "new_ckpt": "TransE_wn18rr_0/new_checkpoint",
        "gamma": 6.0,
        "epsilon": 2.0
    },
    'fb15k237': {
        "drkgc_id2entity_path": "dataset/fb15k237/id2entity.json",
        "drkgc_id2relation_path": "dataset/fb15k237/id2relation.json",
        "transe_entity2text_path": "KG_data/fb15k-237/entity2text.txt",
        "transe_entities_dict_path": "KG_data/fb15k-237/entities.dict",
        "transe_relations_dict_path": "KG_data/fb15k-237/relations.dict",
        "original_ckpt": "TransE_FB15k-237_0/checkpoint",
        "new_ckpt": "TransE_FB15k-237_0/new_checkpoint",
        "gamma": 9.0,
        "epsilon": 2.0
    }
}

def normalize_entity(text, dataset_name):
    text = str(text)
    if dataset_name == 'wn18rr':
        return text.split(',')[0].strip()
    return text.strip()

def normalize_relation(text, dataset_name):
    text = str(text)
    if dataset_name == 'wn18rr':
        return text.replace('_',' ').strip()
    return text.strip()

def run_for_dataset(dataset):
    config = data_configs[dataset]

    # 1. TransE Entity (string - index) 만들기
    transe_id2string = {}
    with open(config['transe_entity2text_path'], 'r', encoding='utf-8') as f:
        for line in f:
            tid, text = line.strip().split('\t')
            transe_id2string[tid] = normalize_entity(text, dataset) # "/m/05hdf": "Nastassja Kinski"
    transe_string2index = collections.defaultdict(list) # {}
    with open(config['transe_entities_dict_path'], 'r', encoding='utf-8') as f:
        for line in f :
            idx, tid = line.strip().split('\t')
            string_name = transe_id2string[tid]
            transe_string2index[string_name].append(int(idx))
            #transe_string2index[string_name] = int(idx) # "Nastassja Kinski": 8014

    # 2. TransE Relation (string - index) 만들기
    transe_string2index_R = {}
    with open(config['transe_relations_dict_path'], 'r', encoding='utf-8') as f:
        for line in f :
            idx, text = line.strip().split('\t')
            norm_text = normalize_relation(text, dataset)
            transe_string2index_R[norm_text] = int(idx) # "hypernym": 0

    # 3. DrKGC 매핑 로드
    with open(config['drkgc_id2entity_path'], 'r', encoding='utf-8') as f:
        drkgc_id2entity = {int(k): v for k,v in json.load(f).items()}
    with open(config['drkgc_id2relation_path'], 'r', encoding='utf-8') as f:
        drkgc_id2relation = {int(k): v for k,v in json.load(f).items()}

    # 4. 원본 체크포인트 로드 및 새 텐서 준비
    state_dict = torch.load(config['original_ckpt'], map_location='cpu')
    ent_embedding = state_dict['model_state_dict']['entity_embedding']
    new_ent_embedding = torch.zeros_like(ent_embedding)
    rel_embedding = state_dict['model_state_dict']['relation_embedding']
    new_rel_embedding = torch.zeros_like(rel_embedding)
    missing_count = {'ent':0, 'rel':0} 

    # 5. entity 매핑 진행 
    embedding_range = (config['gamma'] + config['epsilon']) / ent_embedding.shape[1]
    for drkgc_id, string_name in drkgc_id2entity.items():
        norm_name = normalize_entity(string_name, dataset)
        if norm_name in transe_string2index:
            transe_idx = transe_string2index[norm_name]
            if len(transe_idx) == 1:
                new_ent_embedding[drkgc_id] = ent_embedding[transe_idx[0]]
            else:
                stacked_embs = torch.stack([ent_embedding[i] for i in transe_idx])
                mean_emb = torch.mean(stacked_embs, dim=0)
                new_ent_embedding[drkgc_id] = mean_emb
        else:
            print(f'entity {norm_name}은 transe에 없음')
            nn.init.uniform_(tensor=new_ent_embedding[drkgc_id].unsqueeze(0), a=-embedding_range, b=embedding_range)
            missing_count['ent'] += 1

    # 6. relation 매핑 진행 
    for drkgc_id, string_name in drkgc_id2relation.items():
        norm_name = normalize_relation(string_name, dataset)
        if norm_name in transe_string2index_R:
            transe_idx = transe_string2index_R[norm_name]
            new_rel_embedding[drkgc_id] = rel_embedding[transe_idx]
        else:
            print(f'entity {string_name}은 transe에 없음')
            nn.init_uniform_(tensor=new_rel_embedding[drkgc_id].unsqueeze(0), a=-embedding_range, b=embedding_range)
            missing_count['rel'] += 1

    # 7. 최종 저장 
    print(f"최종 누락: ent({missing_count['ent']}), rel({missing_count['rel']})")
    state_dict['model_state_dict']['entity_embedding'] = new_ent_embedding
    state_dict['model_state_dict']['relation_embedding'] = new_rel_embedding
    torch.save(state_dict, config['new_ckpt'])
    print(f"새로운 ckpt 저장 완료 {config['new_ckpt']}")


if __name__ == '__main__':
    breakpoint()
    dataset = args.dataset.lower()
    if dataset in data_configs.keys():
        run_for_dataset(dataset)
    else:
        print("No dataset")