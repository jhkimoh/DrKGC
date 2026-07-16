import json
import torch
from transformers import AutoTokenizer
from collections import Counter
import numpy as np

def analyze_length_difference_distribution(dataset, tokenizer):
    """
    ① 길이 편향 검증:
    각 example에서 정답 토큰 길이와 나머지 후보들의 최대 토큰 길이 차이 분포를 확인합니다.
    """
    print("\n" + "="*50)
    print("🔍 [1] 길이 편향(Length Bias) 극단성 분포 분석")
    print("="*50)
    
    max_diff_list = []
    target_lengths = []
    
    for ex in dataset:
        h, r, t = ex['triplet_id']
        pred_type = ex.get('type').split('_')[-1]
        
        target_id = t if pred_type == 'tail' else h
        candidate_ids = ex['topk_id']
        
        # 정답이 후보 안에 있는 경우만 분석
        if target_id not in candidate_ids:
            continue
            
        # 정답 및 후보 문자열 가져오기
        target_str = ex['output']
        cand_strs = ex['rank_entities']
        
        # 토큰 길이 계산 (공백 포함)
        target_len = len(tokenizer.encode(" " + target_str, add_special_tokens=False))
        cand_lens = [len(tokenizer.encode(" " + cand, add_special_tokens=False)) for cand in cand_strs]
        
        # (후보군 중 최대 길이) - (정답 길이)
        max_diff = max(cand_lens) - target_len
        max_diff_list.append(max_diff)
        target_lengths.append(target_len)

    # 통계 출력
    print(f"▶ 분석 대상 Example 수: {len(max_diff_list)} 개 (정답이 후보에 포함된 경우)")
    print(f"▶ 정답 엔티티의 평균 토큰 길이: {np.mean(target_lengths):.2f}")
    
    diff_counts = Counter(max_diff_list)
    print("\n📊 [최대 길이 차이 분포 (Max Cand Length - Target Length)]")
    for diff in sorted(diff_counts.keys()):
        count = diff_counts[diff]
        percentage = (count / len(max_diff_list)) * 100
        # 텍스트 히스토그램 시각화
        bar = "█" * int(percentage / 2) 
        print(f"차이 {diff:2d} 토큰 : {count:4d}건 ({percentage:5.2f}%) | {bar}")


def analyze_tokenizer_boundary_shift(dataset, tokenizer):
    """
    ② 토크나이저 경계선 문제 검증:
    strip()을 사용한 기존 슬라이싱 방식과, 이상적인 토큰 매핑(Greedy와 동일)이 얼마나 엇나가는지 확인합니다.
    """
    print("\n" + "="*50)
    print("🔍 [2] 토크나이저 경계선 병합(Boundary Shift) 에러율 분석")
    print("="*50)
    
    mismatch_count = 0
    total_candidates_checked = 0
    
    # 분석을 위한 샘플 프롬프트 (실제 논의했던 끝부분)
    prompt_original = "Question: What is associated with trade name in terms of specific usage or context?\nAnswer: "
    prompt_stripped = prompt_original.strip()
    
    # 기존 코드에서 프롬프트 기준점 잡는 방식
    prompt_len = len(tokenizer.encode(prompt_stripped, add_special_tokens=False))
    
    mismatch_examples = []

    for ex in dataset:
        candidate_ids = ex['rank_entities_id']
        cand_strs = [id2entity.get(cid, "") for cid in candidate_ids]
        
        for cand in cand_strs:
            total_candidates_checked += 1
            
            # [기존 방식] 문자열 통째로 합치고 슬라이싱
            full_text = prompt_stripped + " " + cand
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)
            sliced_cand_ids = full_ids[prompt_len:]
            
            # [이상적 방식] 정답 단어만 따로 토큰화 (" " + cand)
            ideal_cand_ids = tokenizer.encode(" " + cand, add_special_tokens=False)
            
            if sliced_cand_ids != ideal_cand_ids:
                mismatch_count += 1
                if len(mismatch_examples) < 5:  # 디버깅용 샘플 5개 수집
                    mismatch_examples.append({
                        "candidate": cand,
                        "sliced_ids": sliced_cand_ids,
                        "ideal_ids": ideal_cand_ids,
                        "sliced_tokens": tokenizer.convert_ids_to_tokens(sliced_cand_ids),
                        "ideal_tokens": tokenizer.convert_ids_to_tokens(ideal_cand_ids)
                    })

    error_rate = (mismatch_count / total_candidates_checked) * 100
    print(f"▶ 총 검사한 후보 엔티티 수: {total_candidates_checked:,} 개")
    print(f"▶ 어긋남(Mismatch) 발생 횟수: {mismatch_count:,} 번")
    print(f"▶ 🚨 경계선 에러 발생률: {error_rate:.2f} %")
    
    if mismatch_count > 0:
        print("\n[디버깅 샘플 5개 - 실제로 어떻게 쪼개졌는가?]")
        for i, smp in enumerate(mismatch_examples, 1):
            print(f"  {i}. 엔티티: '{smp['candidate']}'")
            print(f"     - 기존 슬라이싱 결과: {smp['sliced_tokens']} (IDs: {smp['sliced_ids']})")
            print(f"     - 이상적(Greedy) 결과: {smp['ideal_tokens']} (IDs: {smp['ideal_ids']})")
            print("-" * 40)


def main():
    # 1. 파일 경로 설정 (작성자님 환경에 맞게 수정 필요)
    test_json_path = "dataset/wn18rr/test.json"
    
    # 2. Llama 3 토크나이저 로드
    # (실제 환경의 Llama-3 경로 입력)
    tokenizer_name = "meta-llama/Meta-Llama-3-8B" 
    print(f"Loading tokenizer: {tokenizer_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # 3. 데이터 로드
    print("Loading data...")
    with open(test_json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # 4. 분석 실행
    analyze_length_difference_distribution(dataset, tokenizer)
    analyze_tokenizer_boundary_shift(dataset, tokenizer)

if __name__ == "__main__":
    main()