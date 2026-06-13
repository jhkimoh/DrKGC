import json
import os
def create_id2entity_mapping(dataset_dir, output_filename):
    id2entity = {}
    splits = ['train.json', 'valid.json', 'test.json']
    for split in splits:
        file_path = os.path.join(dataset_dir, split)
        if not os.path.exists(file_path):
            raise FileNotFoundError("파일이 없습니다.")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            if 'triple' in item and 'triple_id' in item:
                id2entity[item['triple_id'][0]] = item['triple'][0] # Head
                id2entity[item['triple_id'][2]] = item['triple'][2] # Tail
            if 'rank_entities' in item and 'rank_entities_id' in item:
                for ent_str, ent_id in zip(item['rank_entities'], item['rank_entities_id']):
                    id2entity[ent_id] = ent_str
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(id2entity, f, indent=4)
    print(f"총 {len(id2entity)}개의 엔티티 매핑이 {output_filename}에 저장되었습니다.\n")

def create_id2relation_mapping(dataset_dir, output_filename):
    id2entity = {}
    splits = ['train.json', 'valid.json', 'test.json']
    breakpoint()
    for split in splits:
        file_path = os.path.join(dataset_dir, split)
        if not os.path.exists(file_path):
            raise FileNotFoundError("파일이 없습니다.")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            if 'triple' in item and 'triple_id' in item:
                id2entity[item['triple_id'][1]] = item['triple'][1]
            # if 'rank_entities' in item and 'rank_entities_id' in item:
            #     for ent_str, ent_id in zip(item['rank_entities'], item['rank_entities_id']):
            #         id2entity[ent_id] = ent_str
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(id2entity, f, indent=4)
    print(f"총 {len(id2entity)}개의 릴레이션 매핑이 {output_filename}에 저장되었습니다.\n")

def explore_subgraph(dataset_dir):
    splits = ['train.json','valid.json','test.json']
    for split in splits:
        file_path = os.path.join(dataset_dir, split)
        if not os.path.exists(file_path):
            raise FileNotFoundError("파일이 없습니다.")
        with open(file_path, 'r', encoding='utf-8') as f:
            data=json.load(f)
        id2entity = {}
        id2subgraph = {}
        for item in data:
            if 'triple' in item and 'triple_id' in item:
                id2entity[item['triple_id'][0]] = item['triple'][0]
                id2entity[item['triple_id'][2]] = item['triple'][2]
            if 'rank_entities' in item and 'rank_entities_id' in item:
                for ent_str, ent_id in zip(item['rank_entities'], item['rank_entities_id']):
                    id2entity[ent_id] = ent_str
        cnt = set()
        for item in data:
            if 'subgraph' in item:
                for triple in item['subgraph']:
                    if triple[0] in id2entity:
                        continue
                    else:
                        #cnt += 1
                        cnt.add(triple[0])
                        #print(f'{triple[0]} not in id2entity!')
                    if triple[2] in id2entity:
                        continue
                    else:
                        #cnt += 1
                        cnt.add(triple[2])
                        #print(f'{triple[2]} not in id2entity!')
        print(f"total not found ids: {cnt}")

if __name__ == '__main__':
    explore_subgraph('dataset/wn18rr')
    explore_subgraph('dataset/fb15k237')
    #create_id2entity_mapping('dataset/wn18rr', 'dataset/wn18rr/id2entity.json')
    #create_id2entity_mapping('dataset/fb15k237', 'dataset/fb15k237/id2entity.json')
    #create_id2relation_mapping('dataset/wn18rr', 'dataset/wn18rr/id2relation.json')
    #create_id2relation_mapping('dataset/fb15k237', 'dataset/fb15k237/id2relation.json')
