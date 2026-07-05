import os
import json
from collections import Counter
import argparse
import numpy as np
import torch 

def evaluate_base_candidates(dataset_path, data_type='test'):
    """
    Base KGE 모델이 생성한 JSON 데이터를 분석하여 
    정답(output)이 주어진 후보(rank_entities) 내에 몇 번째로 등장하는지 카운트합니다.
    """
    file_path = os.path.join(dataset_path, f"{data_type}.json")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    rank_counter = Counter()
    total_samples = len(data)
    
    for item in data:
        target = item.get("output")
        candidates = item.get("rank_entities", [])
        
        # 정답이 후보 리스트 안에 있는지 확인
        if target in candidates:
            # 리스트 인덱스는 0부터 시작하므로 1을 더해 실제 등수(1~20)로 만듦
            rank = candidates.index(target) + 1 
        else:
            # 후보 안에 정답이 없으면 -1
            rank = -1
            
        rank_counter[rank] += 1
        
    # 통계 계산
    hits_20 = sum(count for rank, count in rank_counter.items() if rank != -1)
    hits_1 = rank_counter.get(1, 0)
    not_found = rank_counter.get(-1, 0)
    
    # 예쁘게 터미널에 출력
    print(f"\n📊 [Base KGE Candidate 적중률 평가: {data_type}.json]")
    print("-" * 50)
    print(f"🔹 총 샘플 수: {total_samples:,}개")
    print(f"🔹 후보 내 정답 포함 (Hits@20): {hits_20:,}개 ({hits_20/total_samples*100:.2f}%) -> LLM이 맞출 수 있는 최대치(Upper Bound)")
    print(f"🔹 원래 1등이었던 정답 (Hits@1): {hits_1:,}개 ({hits_1/total_samples*100:.2f}%)")
    print(f"🔹 후보 내 정답 없음 (-1): {not_found:,}개 ({not_found/total_samples*100:.2f}%)\n")
    
    print("📈 [상세 순위 분포 (등수 : 개수)]")
    # 등수별로 정렬하여 출력 (-1, 1, 2, 3 ... 순서)
    for rank in sorted(rank_counter.keys()):
        if rank == -1:
            print(f" ❌ 범위 밖 (-1) : {rank_counter[rank]:,}개")
        else:
            print(f" ✅ {rank:2d}등 : {rank_counter[rank]:,}개")
            
    print("-" * 50)
    
    return dict(rank_counter)

def evaluate_prediction(prediction_path):
    """
    LLM이 예측한 결과(pred)가 Base KGE 모델이 준 후보(rank_entities) 내에 
    몇 번째로 등장하는지 카운트합니다.
    """
    if not os.path.exists(prediction_path):
        raise FileNotFoundError(f"❌ 파일을 찾을 수 없습니다: {prediction_path}")
        
    with open(prediction_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # JSON 파일이 {"prediction": [...]} 형태이므로 리스트를 추출
    predictions = data.get("prediction", [])
    if not predictions:
        print("⚠️ 'prediction' 데이터가 비어있거나 올바른 형식이 아닙니다.")
        return {}

    rank_counter = Counter()
    total_samples = len(predictions)
    
    for item in predictions:
        pred = item.get("pred")
        candidates = item.get("rank_entities", [])
        
        # LLM의 예측값이 후보 리스트 안에 있는지 확인
        if pred in candidates:
            rank = candidates.index(pred) + 1 
        else:
            rank = -1
            
        rank_counter[rank] += 1
        
    # 통계 계산
    hits_in_candidates = sum(count for rank, count in rank_counter.items() if rank != -1)
    not_found = rank_counter.get(-1, 0)
    
    # 터미널 출력
    print(f"\n📊 [LLM 예측값 vs 후보 리스트(Candidate) 포함 여부 평가: {os.path.basename(prediction_path)}]")
    print("-" * 50)
    print(f"🔹 총 샘플 수: {total_samples:,}개")
    print(f"🔹 예측값이 후보 내에 존재함 (포함): {hits_in_candidates:,}개 ({hits_in_candidates/total_samples*100:.2f}%)")
    print(f"🔹 예측값이 후보 내에 없음 (-1): {not_found:,}개 ({not_found/total_samples*100:.2f}%)\n")
    
    print("📈 [예측값의 후보 내 위치 분포 (등수 : 개수)]")
    for rank in sorted(rank_counter.keys()):
        if rank == -1:
            print(f" ❌ 후보 밖 (-1) : {rank_counter[rank]:,}개")
        else:
            print(f" ✅ 후보 내 {rank:2d}번째 : {rank_counter[rank]:,}개")
            
    print("-" * 50)
    
    return dict(rank_counter)

def analyze_sharpness(pt_file_path):
    if not os.path.exists(pt_file_path):
        raise FileNotFoundError(f"❌ 파일을 찾을 수 없습니다: {pt_file_path}")
        
    print(f"📂 파일 로딩 중: {os.path.basename(pt_file_path)}...")
    data = torch.load(pt_file_path, map_location='cpu')
    
    total_samples = len(data)
    if total_samples == 0:
        print("⚠️ 데이터가 비어있습니다.")
        return
        
    max_probs = []
    margins = []
    entropies = []
    
    for key, tensor in data.items():
        # 만약 저장된 값이 softmax를 거치지 않은 raw logit이라면 softmax 적용
        # (합이 1.0 근처인지 확인하여 자동 처리)
        if not torch.isclose(tensor.sum(), torch.tensor(1.0), atol=1e-2):
            tensor = torch.nn.functional.softmax(tensor, dim=0)
            
        # 1. 내림차순 정렬
        sorted_probs, _ = torch.sort(tensor, descending=True)
        
        # 2. 최대 확률 (Top-1)
        top1_prob = sorted_probs[0].item()
        max_probs.append(top1_prob)
        
        # 3. Top-1과 Top-2의 격차 (Margin)
        if len(sorted_probs) > 1:
            margin = (sorted_probs[0] - sorted_probs[1]).item()
            margins.append(margin)
            
        # 4. 엔트로피 (Entropy): -sum(p * log(p))
        # 0이 되는 것을 방지하기 위해 아주 작은 값(1e-9)을 더해줌
        entropy = -torch.sum(tensor * torch.log(tensor + 1e-9)).item()
        entropies.append(entropy)
        
    # === 통계 계산 ===
    max_probs = np.array(max_probs)
    margins = np.array(margins)
    entropies = np.array(entropies)
    
    print(f"\n📊 [LLM Confidence Sharpness 분석 결과: {os.path.basename(pt_file_path)}]")
    print("-" * 60)
    print(f"🔹 총 분석 쿼리 수: {total_samples:,}개")
    
    print("\n📈 1. 1등 후보의 확률 (Max Probability)")
    print(f"   - 평균 (Mean): {max_probs.mean():.4f} (높을수록 Sharp)")
    print(f"   - 중앙값 (Median): {np.median(max_probs):.4f}")
    print(f"   - 90% 이상의 확신을 가진 쿼리 비율: {(max_probs >= 0.90).mean() * 100:.2f}%")
    
    print("\n📈 2. 1등과 2등의 확률 격차 (Margin)")
    print(f"   - 평균 (Mean): {margins.mean():.4f} (클수록 Sharp)")
    print(f"   - 중앙값 (Median): {np.median(margins):.4f}")
    
    print("\n📈 3. 분포 엔트로피 (Entropy) - 20개 후보 기준 최대값은 약 2.99")
    print(f"   - 평균 (Mean): {entropies.mean():.4f} (0에 가까울수록 Sharp)")
    print(f"   - 중앙값 (Median): {np.median(entropies):.4f}")
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate base KGE candidates.")
    parser.add_argument("--dataset_path", type=str, default=None, help="데이터셋 폴더 경로 (예: data/wn18rr)") # dataset/fb15k237
    parser.add_argument("--type", type=str, default="test", help="평가할 데이터 타입 (기본값: test)")
    parser.add_argument("--prediction_path", type=str, default=None, help="prediction 경로")
    parser.add_argument("--pt_path", type=str, default=None, help=".pt 파일 경로")
    
    args = parser.parse_args()
    
    # 함수 실행
    if args.dataset_path is not None:
        result_dict = evaluate_base_candidates(args.dataset_path, args.type)
    if args.prediction_path is not None:
        prediction_dict = evaluate_prediction(args.prediction_path)
    if args.pt_path is not None:
        analyze_sharpness(args.pt_path)
        # 📊 [LLM Confidence Sharpness 분석 결과: wn_RotatE_logits.pt]
        # ------------------------------------------------------------
        # 🔹 총 분석 쿼리 수: 6,268개

        # 📈 1. 1등 후보의 확률 (Max Probability)
        # - 평균 (Mean): 0.7757 (높을수록 Sharp)
        # - 중앙값 (Median): 0.9065
        # - 90% 이상의 확신을 가진 쿼리 비율: 50.64%

        # 📈 2. 1등과 2등의 확률 격차 (Margin)
        # - 평균 (Mean): 0.6236 (클수록 Sharp)
        # - 중앙값 (Median): 0.8426

        # 📈 3. 분포 엔트로피 (Entropy) - 20개 후보 기준 최대값은 약 2.99
        # - 평균 (Mean): 0.4945 (0에 가까울수록 Sharp)
        # - 중앙값 (Median): 0.3716
        # ------------------------------------------------------------