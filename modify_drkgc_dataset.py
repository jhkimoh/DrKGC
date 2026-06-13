import json
import os

def check_mapping_feasibility(drkgc_dir, dift_dir):
    """
    DrKGC와 DIFT 사이의 1:1 매핑 가능성을 평가합니다.
    오직 'Top-K 후보 이름 리스트의 순서'만을 지문(Key)으로 사용합니다.
    """
    splits = ['train.json', 'valid.json', 'test.json']
    
    for split in splits:
        drkgc_file = os.path.join(drkgc_dir, split)
        dift_file = os.path.join(dift_dir, split)
        
        if not os.path.exists(drkgc_file) or not os.path.exists(dift_file):
            print(f"\n⚠️ [{split}] 파일이 한 쪽 데이터셋에 없어 건너뜁니다.")
            continue
            
        with open(drkgc_file, 'r', encoding='utf-8') as f:
            drkgc_data = json.load(f)
        with open(dift_file, 'r', encoding='utf-8') as f:
            dift_data = json.load(f)
            
        # 1. DrKGC 지문 수집 (Top-K 이름 리스트만!)
        drkgc_keys = set()
        for item in drkgc_data:
            if 'rank_entities' in item:
                # 20개의 이름 순서를 튜플로 만듦 (완벽한 지문)
                drkgc_keys.add(tuple(item['rank_entities']))
                
        # 2. DIFT 지문 수집
        dift_keys = set()
        for item in dift_data:
            if 'topk_names' in item:
                dift_keys.add(tuple(item['topk_names']))
                
        # 3. 매핑 가능성(교집합) 계산
        intersection = drkgc_keys.intersection(dift_keys)
        
        print(f"\n=========================================")
        print(f"📊 [{split}] 매핑 가능성 평가 결과")
        print(f"  - DrKGC 전체 고유 지문 수: {len(drkgc_keys)}")
        print(f"  - DIFT 전체 고유 지문 수 : {len(dift_keys)}")
        print(f"  ✨ 완벽히 일치하는 교집합(매핑 성공) 수: {len(intersection)}")
        
        if len(drkgc_keys) > 0:
            match_rate = (len(intersection) / len(drkgc_keys)) * 100
            print(f"  🚀 DrKGC 기준 매핑 성공률: {match_rate:.2f}%")
            
            if match_rate == 100.0:
                print("  => ✅ 완벽합니다! DrKGC의 모든 문제를 DIFT 데이터로 덮어씌울 수 있습니다.")
            elif match_rate > 80.0:
                print("  => 🟡 준수합니다. 대부분 매핑 가능하나 일부 유실이 발생할 수 있습니다.")
            else:
                print("  => 🚨 매칭률이 낮습니다. DIFT 모델 설정(Top-K 수 등)이 다른지 확인이 필요합니다.")

def explore_candidate(dataset_dir): # drkgc dataset 의 모든 example의 candidate가 고유한지 평가 
    print(dataset_dir)
    splits = ['train.json','valid.json','test.json']
    for split in splits:
        file_path = os.path.join(dataset_dir, split)
        if not os.path.exists(file_path):
            raise FileNotFoundError("파일이 없습니다.")
        with open(file_path, 'r', encoding='utf-8') as f:
            data=json.load(f)
        breakpoint()
        unique_set = set()
        valid_item_count = 0
        for item in data:
            if 'rank_entities' in item:
                valid_item_count += 1
                triple = tuple(item['triple'])
                candidate_tuple = tuple(item['rank_entities'])
                unique_key = (candidate_tuple)
                unique_set.add(unique_key)
        unique_count = len(unique_set)
        print(f"DrKGC [{split}] 평가 결과:")
        print(f"  - 전체 쿼리(데이터) 수: {valid_item_count}")
        print(f"  - 고유한 Top-K 후보 리스트 수: {unique_count}")

def normalize_entity(text, dataset_name):
    text = str(text)
    if dataset_name == 'wn18rr':
        return text.split(',')[0].strip()
    return text.strip()

def normalize_relation(text):
    text = str(text)
    return text.replace('_',' ').strip()

def id2str_map(id2ent_path, dataset):
    transe_id2string = {}
    with open(id2ent_path, 'r', encoding='utf-8') as f:
        for line in f:
            tid, text = line.strip().split('\t')
            transe_id2string[tid] = normalize_entity(text, dataset)
    return transe_id2string

def explore_topk(dataset_dir, id2str):
    print(dataset_dir)
    splits = ['train.json','valid.json','test.json']
    for split in splits:
        file_path = os.path.join(dataset_dir, split)
        if not os.path.exists(file_path):
            raise FileNotFoundError("파일이 없습니다.")
        with open(file_path, 'r', encoding='utf-8') as f:
            data=json.load(f)
        breakpoint()
        unique_set = set()
        valid_item_count = 0
        for item in data:
            if 'topk_names' in item:
                valid_item_count += 1
                triple = item['triplet']
                triple[0] = id2str[triple[0]]
                if 'wn18rr' in dataset_dir.lower():
                    triple[1] = normalize_relation(triple[1])
                triple[2] = id2str[triple[2]]
                triple = tuple(triple)
                candidate_tuple = tuple(item['topk_names'])
                #candidate_tuple = tuple(item['topk_ents'])
                unique_key = (candidate_tuple)
                unique_set.add(unique_key)
        unique_count = len(unique_set)
        print(f"DIFT [{split}] 평가 결과:")
        print(f"  - 전체 쿼리(데이터) 수: {valid_item_count}")
        print(f"  - 고유한 Top-K 후보 리스트 수: {unique_count}")
    

def load_dict_to_idx(dict_path):
    id2idx = {}
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"{dict_path} 파일이 없습니다!")
        
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                idx = int(parts[0])
                string_id = parts[1]
                id2idx[string_id] = idx
    return id2idx

def append_dift_ids_to_drkgc(drkgc_dir, dift_dir, output_dir, dataset_name, ent2idx, rel2idx):
    """
    (Triple + Candidate) 지문으로 매핑하여, 
    DrKGC 데이터에 DIFT의 'topk_ents'를 병합하고 새로운 폴더에 저장합니다.
    """
    splits = ['train.json', 'valid.json', 'test.json']
    
    # 아웃풋 폴더 생성 (기존 데이터 보호)
    os.makedirs(output_dir, exist_ok=True)
    
    for split in splits:
        drkgc_file = os.path.join(drkgc_dir, split)
        dift_file = os.path.join(dift_dir, split)
        output_file = os.path.join(output_dir, split)
        
        if not os.path.exists(drkgc_file) or not os.path.exists(dift_file):
            print(f"\n⚠️ [{split}] 파일이 한 쪽 데이터셋에 없어 건너뜁니다.")
            continue
            
        with open(drkgc_file, 'r', encoding='utf-8') as f:
            drkgc_data = json.load(f)
        with open(dift_file, 'r', encoding='utf-8') as f:
            dift_data = json.load(f)
            
        # 1. DIFT 데이터를 딕셔너리로 구축
        dift_dict = {}
        for item in dift_data:
            if 'topk_names' in item and 'triplet' in item and 'topk_ents' in item:
                candidates = tuple(item['topk_names'])
                dift_dict[candidates] = {'topk_ents':item['topk_ents'], 'triplet':item['triplet']}
                
        # 2. DrKGC 데이터를 순회하며 매핑 및 Append 진행
        matched_count = 0
        total_count = 0
        missing_id_count = 0
        for item in drkgc_data:
            if 'rank_entities' in item and 'triple' in item:
                total_count += 1
                candidates = tuple(item['rank_entities'])
                # 매핑 성공 시, 기존 데이터를 해치지 않고 새 Key-Value만 추가!
                if candidates in dift_dict:
                    dift_info = dift_dict[candidates]
                    topk_ents_list = dift_info['topk_ents']
                    raw_triplet = dift_info['triplet']
                    item['topk_ents'] = topk_ents_list
                    item['triplet'] = raw_triplet
                    topk_id_list = []
                    for ent_str in topk_ents_list:
                        if ent_str in ent2idx:
                            topk_id_list.append(ent2idx[ent_str])
                        else:
                            # 만약 사전에 없는 희귀 케이스 대비 (-1로 처리)
                            topk_id_list.append(-1)
                            missing_id_count += 1
                            
                    item['topk_id'] = topk_id_list
                    h_id, r_id, t_id = raw_triplet[0], raw_triplet[1], raw_triplet[2]
                    h_idx = ent2idx.get(h_id, -1)
                    r_idx = rel2idx.get(r_id, -1)
                    t_idx = ent2idx.get(t_id, -1)
                    if h_idx == -1 or r_idx == -1 or t_idx == -1:
                        missing_triplet_count += 1   
                    item['triplet_id'] = [h_idx, r_idx, t_idx]
                    matched_count += 1
                else:
                    # 매핑 실패한 경우 (확인용)
                    pass 
                    
        # 3. 새로운 JSON 파일로 예쁘게 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(drkgc_data, f, ensure_ascii=False, indent=4)
            
        print(f"\n=========================================")
        print(f"✅ [{split}] 병합 완료 ({dataset_name.upper()})")
        print(f"  - 전체 DrKGC 데이터 수 : {total_count}")
        print(f"  - 매핑 성공 및 추가 완료: {matched_count}")
        print(f"  - 파일 저장 위치: {output_file}")

if __name__ == '__main__':
    # print("\n[ WN18RR 매핑 평가 시작 ]")
    # check_mapping_feasibility(
    #     drkgc_dir='dataset/wn18rr', 
    #     dift_dir='DIFT-dataset/WN18RR/SimKGC/data_KGELlama'
    # )
    # print("\n[ FB15K237 매핑 평가 시작 ]")
    # check_mapping_feasibility(
    #     drkgc_dir='dataset/fb15k237', 
    #     dift_dir='DIFT-dataset/FB15K237/CoLE/data_KGELlama'
    # )
    print("\n[ WN18RR 데이터 병합 시작 ]")
    wn18rr_ent2idx = load_dict_to_idx("KG_data/wn18rr/entities.dict")
    wn18rr_rel2idx = load_dict_to_idx("KG_data/wn18rr/relations.dict")
    breakpoint()
    append_dift_ids_to_drkgc(
        drkgc_dir='dataset/wn18rr', 
        dift_dir='DIFT-dataset/WN18RR/SimKGC/data_KGELlama',
        output_dir='dataset_merged/wn18rr', # 새로운 저장 폴더!
        dataset_name='wn18rr',
        ent2idx=wn18rr_ent2idx,
        rel2idx=wn18rr_rel2idx
    )
    
    print("\n[ FB15K237 데이터 병합 시작 ]")
    fb15k237_ent2idx = load_dict_to_idx("KG_data/fb15k-237/entities.dict")
    fb15k237_rel2idx = load_dict_to_idx("KG_data/fb15k-237/relations.dict")
    append_dift_ids_to_drkgc(
        drkgc_dir='dataset/fb15k237', 
        dift_dir='DIFT-dataset/FB15K237/CoLE/data_KGELlama',
        output_dir='dataset_merged/fb15k237', # 새로운 저장 폴더!
        dataset_name='fb15k237',
        ent2idx=fb15k237_ent2idx,
        rel2idx=fb15k237_rel2idx
    )
    #wn18rr_id2str = id2str_map("KG_data/wn18rr/entity2text.txt", 'wn18rr')
    #fb15k237_id2str = id2str_map("KG_data/fb15k-237/entity2text.txt", 'fb15k237')
    #explore_topk('DIFT-dataset/WN18RR/SimKGC/data_KGELlama', wn18rr_id2str)
    #explore_topk('DIFT-dataset/FB15K237/CoLE/data_KGELlama', fb15k237_id2str)
    #explore_candidate('dataset/wn18rr')
    #explore_candidate('dataset/fb15k237')