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
    return data['prediction']

def analyze_errors(baseline_path, extract_path, output_dir="error_analysis"):
    original_preds = load_predictions(baseline_path)
    letter_preds = load_predictions(extract_path)
    
    if original_preds is None or letter_preds is None:
        return

    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "1_Common_Correct": [],       
        "2_Common_Wrong": [],         
        "3_Only_Original_Correct": [],
        "4_Only_Letter_Correct": [],  
        "Summary": {}
    }
    
    for i, (orig_ex, lett_ex) in enumerate(zip(original_preds, letter_preds)):
        
        # Original 비교: String Target vs String Pred
        orig_target = str(orig_ex['target']).strip()
        orig_pred = str(orig_ex['pred']).strip()
        orig_is_correct = (orig_target == orig_pred)
        
        # Letter 비교: Letter Target vs Letter Pred
        lett_target = str(lett_ex['target']).strip()
        lett_pred = str(lett_ex['pred']).strip()
        lett_is_correct = (lett_target == lett_pred)
        
        head = orig_ex.get('triple', [orig_ex.get('query_entity', 'N/A')])[0]
        relation = orig_ex.get('triple', ['', 'N/A', ''])[1] if 'triple' in orig_ex else 'N/A'

        record = {
            "index": i,
            "query_triple": f"({head}, {relation}, ?)",
            "Original_Target": orig_target,
            "Original_Predicted": orig_pred,
            "Letter_Target": lett_target,
            "Letter_Predicted": lett_pred
        }

        # 4가지 경우의 수 분류 및 중복 텍스트 카운팅
        if orig_is_correct and lett_is_correct:
            results["1_Common_Correct"].append(record)
        elif not orig_is_correct and not lett_is_correct:
            results["2_Common_Wrong"].append(record)
        elif orig_is_correct and not lett_is_correct:
            # 🌟 Only Original Correct 케이스: rank_entities 내 정답 텍스트 중복 검사
            rank_entities = orig_ex.get('rank_entities', [])
            dup_count = rank_entities.count(orig_target)
            
            record["Candidate_Target_String_Count"] = dup_count
            record["Has_Duplicate_Target_String"] = (dup_count > 1)
            results["3_Only_Original_Correct"].append(record)
            
        elif not orig_is_correct and lett_is_correct:
            results["4_Only_Letter_Correct"].append(record)
    
    total = len(original_preds)
    c_correct = len(results["1_Common_Correct"])
    c_wrong = len(results["2_Common_Wrong"])
    only_orig = len(results["3_Only_Original_Correct"])
    only_lett = len(results["4_Only_Letter_Correct"])
    
    orig_hits = c_correct + only_orig
    lett_hits = c_correct + only_lett
    
    # 🌟 Only Original Correct 내부의 중복 타겟 텍스트 통계 계산
    only_orig_dup_count = sum(1 for rec in results["3_Only_Original_Correct"] if rec.get("Has_Duplicate_Target_String", False))
    only_orig_dup_ratio = (only_orig_dup_count / only_orig * 100) if only_orig > 0 else 0.0

    results["Summary"] = {
        "Total_Test_Examples": total,
        "Common_Correct_Count": f"{c_correct} ({c_correct/total*100:.2f}%)",
        "Common_Wrong_Count": f"{c_wrong} ({c_wrong/total*100:.2f}%)",
        "Only_Original_Correct_Count": f"{only_orig} ({only_orig/total*100:.2f}%)",
        "Only_Letter_Correct_Count": f"{only_lett} ({only_lett/total*100:.2f}%)",
        
        "Original_Total_Hits1": f"{orig_hits} ({orig_hits/total*100:.2f}%)",
        "Letter_Total_Hits1": f"{lett_hits} ({lett_hits/total*100:.2f}%)",
        
        "Analysis_of_Only_Original_Correct": {
            "Total_Cases": only_orig,
            "Target_String_Duplicated_in_Candidates": only_orig_dup_count,
            "Duplication_Ratio": f"{only_orig_dup_ratio:.2f}%"
        }
    }

    # 콘솔 출력 포맷팅
    print("="*70)
    print(f"🎯 Hits@1 비교 분석 결과: {baseline_path.split(os.sep)[1].upper()}")
    print("="*70)
    print(f" 📊 전체 데이터 수 : {total}")
    print(f" ▶️ Original 모델 Hits@1 : {orig_hits} ({orig_hits/total*100:.2f}%)")
    print(f" ▶️ Letter 모델 Hits@1   : {lett_hits} ({lett_hits/total*100:.2f}%)")
    print("-"*70)
    print(f" [상세 분류]")
    print(f"  - 둘 다 맞춤 (Common Correct)       : {c_correct} ({c_correct/total*100:.2f}%)")
    print(f"  - 둘 다 틀림 (Common Wrong)         : {c_wrong} ({c_wrong/total*100:.2f}%)")
    print(f"  - Original만 맞춤 (Only Orig)       : {only_orig} ({only_orig/total*100:.2f}%)")
    print(f"  - Letter만 맞춤 (Only Lett)         : {only_lett} ({only_lett/total*100:.2f}%)")
    print("-"*70)
    print(f" 🔍 [Only Original Correct 심층 분석]")
    print(f"  - 발생 건수: {only_orig}건")
    print(f"  - 이 중 후보군에 정답 텍스트(string)가 2개 이상 중복된 건수: {only_orig_dup_count}건")
    print(f"  - 텍스트 중복(동음이의어) 비율: {only_orig_dup_ratio:.2f}%")
    print("="*70)
    
    output_file = os.path.join(output_dir, f'{baseline_path.split(os.sep)[1]}_hits1_comparison_report.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"\n📁 상세 분석 리스트가 저장되었습니다: {output_file}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_path", type=str, default="results/fb15k237/llama3/prediction.json")
    parser.add_argument("--extract_path", type=str, default="results_letter/fb15k237/llama3/prediction.json")
    args = parser.parse_args()
    
    datasets = ['wn18rr', 'fb15k237']
    for dataset in datasets:
        args.baseline_path = f"results/{dataset}/llama3/prediction.json"
        args.extract_path = f"results_letter/{dataset}/llama3/prediction.json"
        
        print(f"🚀 분석 시작: {dataset}")
        analyze_errors(args.baseline_path, args.extract_path)