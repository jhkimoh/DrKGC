import pickle
import os
import json
from tqdm import tqdm
import argparse


def load_pkl(base_dir, dataset, mode, file_name):
    path = os.path.join(base_dir, dataset, mode, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def load_kg_mappings(kg_data_dir, dataset_name):
    entity2id = {}
    relation2id = {}
    
    # 1. 엔티티 매핑 로드
    ent_path = os.path.join(kg_data_dir, dataset_name, 'entities.dict')
    with open(ent_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)
            
    # 2. 관계 매핑 로드
    rel_path = os.path.join(kg_data_dir, dataset_name, 'relations.dict')
    with open(rel_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
            
    return entity2id, relation2id

def load_lexicon(lexicon_dir, kg_dataset_name):
    # fb15k-237 -> fb15k237 처리
    clean_name = kg_dataset_name.replace("-", "") 
    
    head_path = os.path.join(lexicon_dir, f"{clean_name}_head_prediction.json")
    tail_path = os.path.join(lexicon_dir, f"{clean_name}_tail_prediction.json")
    
    with open(head_path, 'r', encoding='utf-8') as f:
        head_lexicon = json.load(f)
        
    with open(tail_path, 'r', encoding='utf-8') as f:
        tail_lexicon = json.load(f)
        
    return head_lexicon, tail_lexicon

def make_prompt(query_entity, relation, topk_texts, lexicon_dict, dataset_name):
    """DrKGC 프롬프트 양식에 맞춰 input 문자열을 생성합니다."""
    # 1. 튜플 문자열 생성: ('A', 'B', 'C')
    topk_tuple_str = "(" + ", ".join([f"'{text}'" for text in topk_texts]) + ")"
    
    # 2. 임베딩 힌트 문자열 생성: 'A': [ENTITY], 'B': [ENTITY]
    emb_list = [f"'{text}': [ENTITY]" for text in topk_texts]
    entity_embeddings_str = ", ".join(emb_list)
    
    if 'wn18' in dataset_name.lower():
        # WN18RR일 때만 '_hypernym' -> 'hypernym' 형태로 변환
        clean_relation = relation.replace('_', ' ').strip()
    else:
        # FB15k-237 등은 원본 텍스트 그대로 사용
        clean_relation = relation
    
    # 3. 질문(Question) 템플릿 가져오기 (만약 매핑이 안 맞으면 기본 템플릿 사용)
    # lexicon의 관계명은 보통 소문자이거나 특정 형식일 수 있으니 get을 통해 안전하게 가져옵니다.
    question_template = lexicon_dict.get(clean_relation, f"What is the {relation} of {{}}?")
    question_str = question_template.format(query_entity)
    
    # 4. 최종 프롬프트 조립
    prompt = (
        f"You are an excellent linguist. The task is to predict the answer based on the given question, "
        f"and you only need to answer one entity. The answer must be in {topk_tuple_str}.\n"
        f"You can refer to the entity embeddings: '{query_entity}': [QUERY], {entity_embeddings_str}.\n\n"
        f"Question: {question_str}\n"
        f"Answer: "
    )
    return prompt

def process_prediction(triples_ent, triples_text, topk_ents_list, topk_texts_list, is_inverse, entity2id, relation2id, lexicon_dict, dataset_name):
    processed_data = []
    for idx in range(len(triples_ent)):
        h_ent, r_ent, t_ent = triples_ent[idx]
        h_text, r_text, t_text = triples_text[idx]

        topk_ents = topk_ents_list[idx]
        topk_texts = topk_texts_list[idx]

        if is_inverse:
            pred_type = "predicted_head"
            query_entity = t_text
            query_entity_id = t_ent
            target_text = h_text
        else:
            pred_type = "predicted_tail"
            query_entity = h_text
            query_entity_id = h_ent
            target_text = t_text
        
        try:
            rank = topk_texts.index(target_text) + 1
        except ValueError:
            rank = 99999  # Top-K 안에 정답이 없을 경우
        
        input_prompt = make_prompt(query_entity, r_text, topk_texts, lexicon_dict, dataset_name)
        h_idx = entity2id.get(h_ent, -1)
        r_idx = relation2id.get(r_ent, -1)
        t_idx = entity2id.get(t_ent, -1)
        query_idx = entity2id.get(query_entity_id, -1)
        topk_idxs = [entity2id.get(e, -1) for e in topk_ents]

        prediction = {
            "triple": [h_text, r_text, t_text],          
            "triple_id": [h_idx, r_idx, t_idx],          
            "type": pred_type,                           
            "query_entity": query_entity,                
            "query_entity_id": query_idx,        
            "rank_entities": topk_texts,
            "rank_entities_id": topk_idxs,           
            "rank": rank,                                
            "input": input_prompt,                                
            "output": target_text,                        
            "topk_ents": topk_ents,                     
            "topk_id": topk_idxs,
            "triplet": [h_text, r_text, t_text],         
            "triplet_id": [h_idx, r_idx, t_idx]   
        }
        processed_data.append(prediction)
    
    return processed_data

def create_dataset(dataset, mode, base_dir='RotatE'):
    print(f"Processing {dataset} - {mode}...")

    kg_dataset_name = 'fb15k-237' if dataset == 'fb15k' else 'wn18rr'
    lc_dataset_name = 'fb15k237' if dataset == 'fb15k' else 'wn18rr'
    entity2id, relation2id = load_kg_mappings('KG_data', kg_dataset_name)
    head_lexicon, tail_lexicon = load_lexicon('lexicon', lc_dataset_name)

    # 데이터 로드 (Head Prediction)
    h_triples_ent = load_pkl(base_dir, dataset, mode, 'triple_ent_head.pkl')
    h_triples_text = load_pkl(base_dir, dataset, mode, 'triple_text_head.pkl')
    h_topk_ents = load_pkl(base_dir, dataset, mode, 'topk_ent_head.pkl')
    h_topk_texts = load_pkl(base_dir, dataset, mode, 'topk_text_head.pkl')
    
    # 데이터 로드 (Tail Prediction)
    t_triples_ent = load_pkl(base_dir, dataset, mode, 'triple_ent_tail.pkl')
    t_triples_text = load_pkl(base_dir, dataset, mode, 'triple_text_tail.pkl')
    t_topk_ents = load_pkl(base_dir, dataset, mode, 'topk_ent_tail.pkl')
    t_topk_texts = load_pkl(base_dir, dataset, mode, 'topk_text_tail.pkl')

    num_samples = len(h_triples_ent)
    assert (
        num_samples == len(h_triples_text) == len(h_topk_ents) == len(h_topk_texts) ==
        len(t_triples_ent) == len(t_triples_text) == len(t_topk_ents) == len(t_topk_texts)
    ), "❌ Mismatch in lengths of the 8 loaded .pkl files!"

    tail_data = process_prediction(
        t_triples_ent, t_triples_text, t_topk_ents, t_topk_texts, 
        is_inverse=False, entity2id=entity2id, relation2id=relation2id, 
        lexicon_dict=tail_lexicon, dataset_name=kg_dataset_name
    )
    head_data = process_prediction(
        h_triples_ent, h_triples_text, h_topk_ents, h_topk_texts, 
        is_inverse=True, entity2id=entity2id, relation2id=relation2id, 
        lexicon_dict=head_lexicon, dataset_name=kg_dataset_name
    )

    final_output = []
    for idx in tqdm(range(num_samples), desc="Merging Data"):
        final_output.append(tail_data[idx])
        final_output.append(head_data[idx])
        
    output_path = os.path.join('dataset', lc_dataset_name, f"{mode}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
        
    print(f"✅ Saved to {output_path}. Total records: {len(final_output)}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fb15k', help='fb15k | wn18rr')
    parser.add_argument('--mode', type=str, default='train', help='train | test')
    parser.add_argument('--all', action='store_true', help='process all dataset')
    args = parser.parse_args()
    #breakpoint()
    if args.all:
        for dataset in ['fb15k', 'wn18rr']:
            create_dataset(dataset, args.mode)
    else:
        if args.dataset in ['fb15k', 'wn18rr']:
            create_dataset(args.dataset, args.mode)