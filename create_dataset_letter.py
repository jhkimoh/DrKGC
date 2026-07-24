import os
import json
import re
import shutil
import random
from tqdm import tqdm
import argparse


def generate_letters(n):
    """후보군 개수에 맞춰 A, B, C ... Z, AA, AB 형태의 기호를 생성합니다."""
    letters = []
    for i in range(n):
        if i < 26:
            letters.append(chr(65 + i))
        else:
            letters.append(chr(65 + i // 26 - 1) + chr(65 + i % 26))
    return letters

def convert_prompt(item):
    """원본 프롬프트의 구조를 최대한 유지하면서 객관식 포맷으로 변경합니다."""
    original_input = item.get("input", "")
    original_output = item.get("output", "")
    
    # 🌟 4가지 핵심 리스트 모두 가져오기
    rank_entities = item.get("rank_entities", [])
    rank_entities_id = item.get("rank_entities_id", [])
    topk_ents = item.get("topk_ents", [])
    topk_id = item.get("topk_id", [])
    pred_type = item.get('type', '')
    triplet_id = item.get('triplet_id',[])
    target_id = triplet_id[2] if 'tail' in pred_type else triplet_id[0]

    if not original_input or not rank_entities:
        return original_input, original_output

    # 1. Prefix 추출 ("The answer must be in" 이전까지)
    prefix_match = re.search(r"(.*?)(The answer must be in)", original_input, flags=re.DOTALL)
    prefix = prefix_match.group(1).strip() if prefix_match else "You are an excellent linguist. The task is to predict the answer based on the given question, and you only need to answer one entity."

    # 2. [QUERY] 토큰 추출
    query_match = re.search(r"('[^']+'\s*:\s*\[QUERY\])", original_input)
    query_str = query_match.group(1) if query_match else ""

    # 3. Question 추출
    question_match = re.search(r"(Question:.*)", original_input, flags=re.DOTALL)
    question_str = question_match.group(1) if question_match else "Question: \nAnswer: "

    # 4. 동적 알파벳 튜플 생성 (업데이트된 current_entities 기준)
    letters = generate_letters(len(rank_entities))
    letters_tuple_str = ", ".join([f"'{l}'" for l in letters])

    # 5. ✨ 원본과 가장 유사한 형태로 조립
    new_input = f"{prefix} The answer must be in ({letters_tuple_str}).\n"
    new_input += "You can refer to the entity embeddings:\n"
    
    if query_str:
        new_input += f"{query_str},\n"

    # 세로로 나열
    for letter, ent in zip(letters, rank_entities):
        new_input += f"{letter}. '{ent}': [ENTITY]\n"

    new_input += f"\n{question_str}"

    # 6. 새로운 Output 매핑
    try:
        #ans_idx = rank_entities.index(original_output)
        ans_idx = topk_id.index(target_id)
        new_output = letters[ans_idx]
    except ValueError:
        new_output = original_output

    return new_input, new_output

def main():
    base_in_dir = "dataset_merged"
    base_out_dir = "dataset_letter"
        
    datasets = ["fb15k237", "wn18rr"]
    splits = ["train.json", "valid.json", "test.json"]
    breakpoint()
    for ds in datasets:
        in_ds_dir = os.path.join(base_in_dir, ds)
        out_ds_dir = os.path.join(base_out_dir, ds)
        os.makedirs(out_ds_dir, exist_ok=True)

        # 환경 호환성을 위해 id2entity.json 등 부가 파일 단순 복사
        for f in os.listdir(in_ds_dir):
            if f.endswith(".json") and f not in splits:
                shutil.copy(os.path.join(in_ds_dir, f), os.path.join(out_ds_dir, f))

        for split in splits:
            in_path = os.path.join(in_ds_dir, split)
            out_path = os.path.join(out_ds_dir, split)

            if not os.path.exists(in_path):
                continue

            print(f"🔄 처리 중: {ds} / {split} ...")
            with open(in_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 프로그레스 바와 함께 변환 진행
            for item in tqdm(data, desc=f"{split} 변환 중"):
                # item 자체를 넘겨서 내부의 4가지 리스트와 input/output을 모두 업데이트
                new_in, new_out = convert_prompt(item)
                item['input'] = new_in
                item['output'] = new_out

            # 새 파일 저장
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"\n✅ 완료되었습니다! '{base_out_dir}' 폴더가 성공적으로 생성되었으며 4가지 후보 리스트가 모두 안전하게 업데이트되었습니다.")

if __name__ == '__main__':
    main()