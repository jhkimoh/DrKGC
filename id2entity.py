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

if __name__ == '__main__':
    create_id2entity_mapping('dataset/wn18rr', 'dataset/wn18rr/id2entity.json')
    create_id2entity_mapping('dataset/fb15k237', 'dataset/fb15k237/id2entity.json')