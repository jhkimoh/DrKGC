import json
import os
import argparse

def load_predictions(filepath):
    """prediction.json 파일을 읽어서 반환합니다."""
    if not os.path.exists(filepath):
        print(f"Error: 파일을 찾을 수 없습니다 - {filepath}")
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['prediction'] # 'prediction' 키 안의 리스트 반환

def analyze_errors(baseline_path, extract_path, output_dir="error_analysis"):
    """두 예측 결과를 비교하여 오답 분석을 수행합니다."""
    
    baseline_preds = load_predictions(baseline_path)
    extract_preds = load_predictions(extract_path)
    
    if baseline_preds is None or extract_preds is None:
        return

    # 결과를 저장할 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "1_Common_Correct": [],       # 둘 다 맞춤 (공통 정답)
        "2_Common_Wrong": [],         # 둘 다 틀림 (공통 오답)
        "3_Only_DrKGC_Correct": [],   # 기존 모델만 맞춤 (Extract가 망친 문제)
        "4_Only_Extract_Correct": [], # Extract만 맞춤 (새로 발굴한 정답)
        "Summary": {}
    }
    for i, (base_ex, ext_ex) in enumerate(zip(baseline_preds, extract_preds)):
        # 정답 (Ground Truth)
        target = base_ex['target']
        
        # 모델들의 1순위 예측값 (LLM Output)
        base_pred = base_ex['pred']
        ext_pred = ext_ex['pred']
        
        # 맞췄는지 틀렸는지 판별 (Hits@1 기준: 완벽 일치)
        base_is_correct = (target == base_pred)
        ext_is_correct = (target == ext_pred)
        
        # 보기 좋게 저장할 데이터 레코드 형식
        # json에 'triple' 정보가 있다면 사용하고, 없다면 query_entity를 사용
        head = base_ex.get('triple', [base_ex.get('query_entity', 'N/A')])[0]
        relation = base_ex.get('triple', ['', 'N/A', ''])[1] if 'triple' in base_ex else 'N/A'

        record = {
            "index": i,
            "query_triple": f"({head}, {relation}, ?)",
            "Ground_Truth_Target": target,
            "DrKGC_Predicted": base_pred,
            "Extract_Predicted": ext_pred
        }

        # 4가지 경우의 수에 따라 분류하여 리스트에 추가
        if base_is_correct and ext_is_correct:
            results["1_Common_Correct"].append(record)
        elif not base_is_correct and not ext_is_correct:
            results["2_Common_Wrong"].append(record)
        elif base_is_correct and not ext_is_correct:
            results["3_Only_DrKGC_Correct"].append(record)
        elif not base_is_correct and ext_is_correct:
            results["4_Only_Extract_Correct"].append(record)
    
    total_examples = len(baseline_preds)
    results["Summary"] = {
        "Total_Test_Examples": total_examples,
        "Common_Correct_Count": len(results["1_Common_Correct"]),
        "Common_Wrong_Count": len(results["2_Common_Wrong"]),
        "Only_DrKGC_Correct_Count": len(results["3_Only_DrKGC_Correct"]),
        "Only_Extract_Correct_Count": len(results["4_Only_Extract_Correct"]),
        
        # 두 모델의 최종 Hits@1 (정답 개수)
        "DrKGC_Total_Hits1": len(results["1_Common_Correct"]) + len(results["3_Only_DrKGC_Correct"]),
        "Extract_Total_Hits1": len(results["1_Common_Correct"]) + len(results["4_Only_Extract_Correct"])
    }

    # 콘솔에 깔끔하게 출력
    print("="*60)
    print("🎯 Hits@1 (1순위 예측) 정밀 비교 분석 결과")
    print("="*60)
    for k, v in results["Summary"].items():
        print(f" - {k:<30}: {v} 개")
    print("="*60)
    
    # 📝 핵심 포인트: 추가 실험의 명분 확인
    print("\n💡 [연구 방향성 진단]")
    print(f"Extract 모듈이 새롭게 맞춘 문제 (DrKGC는 틀림): {results['Summary']['Only_Extract_Correct_Count']} 개")
    print(f"Extract 모듈 때문에 틀린 문제 (DrKGC는 맞춤): {results['Summary']['Only_DrKGC_Correct_Count']} 개")
    output_file = os.path.join(output_dir, f'{baseline_path.split(os.sep)[1]}_hits1_comparison_report.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"\n상세 분석 리스트가 저장되었습니다: {output_file}")

if __name__ == "__main__":
    # 파일 경로 설정 (질문자님의 환경에 맞게 수정하세요)
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_path", type=str, default="results/fb15k237/llama3/prediction.json")
    parser.add_argument("--extract_path", type=str, default="results_extract/fb15k237/llama3/prediction.json")
    args = parser.parse_args()
    analyze_errors(args.baseline_path, args.extract_path)