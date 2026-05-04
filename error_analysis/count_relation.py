import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

DATASET_METADATA = {
    "fb15k237": {"E_dim": 14541, "R_dim": 237},
    "wn18rr": {"E_dim": 40943, "R_dim": 11},
}

def load_predictions(filepath):
    """prediction.json 파일을 읽어서 반환합니다."""
    if not os.path.exists(filepath):
        print(f"Error: 파일을 찾을 수 없습니다 - {filepath}")
        return None, None
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("3_Only_DrKGC_Correct", []), data.get("4_Only_Extract_Correct", [])

def extract_relation(triple_str):
    """형식: '(head, relation, ?)' 에서 relation을 추출합니다."""
    try:
        # 괄호를 제거하고 쉼표로 분리
        parts = triple_str.strip("()").split(", ")
        
        if len(parts) >= 3:
            # relation 문자열 자체에 쉼표가 있을 수 있으므로 
            # head(첫 요소)와 ?(마지막 요소)를 제외한 중간 부분을 모두 합침
            return ", ".join(parts[1:-1])
        return "unknown"
    except Exception:
        return "unknown"

# ---------------------------------------------------------
# 메인 분석 함수
# ---------------------------------------------------------
def count_relation(filepath):
    """특정 데이터셋의 예측 결과를 분석하여 시각화하고 저장합니다."""
    
    # 파일명으로 데이터셋 이름 유추
    dataset_name = "wn18rr" if "wn18rr" in filepath.lower() else "fb15k237"
    print(f"\n[{dataset_name.upper()}] 분석 시작...")

    drkgc, extract = load_predictions(filepath)
    if drkgc is None or extract is None:
        return

    # 1. Relation 별 카운트 저장용 딕셔너리
    drkgc_counts = defaultdict(int)
    extract_counts = defaultdict(int)
    
    # 2. Relation 별 상세 내역 저장용 딕셔너리
    details = defaultdict(lambda: {"only_drkgc": [], "only_extract": []})

    # ---------------------------------------------------------
    # 데이터 파싱
    # ---------------------------------------------------------
    # DrKGC만 맞춘 데이터
    for item in drkgc:
        rel = extract_relation(item['query_triple'])
        drkgc_counts[rel] += 1
        details[rel]["only_drkgc"].append({
            "triple": item['query_triple'],
            "drkgc_pred": item['DrKGC_Predicted'],
            "extract_pred": item['Extract_Predicted'],
            "target": item['Ground_Truth_Target']
        })

    # Extract만 맞춘 데이터
    for item in extract:
        rel = extract_relation(item['query_triple'])
        extract_counts[rel] += 1
        details[rel]["only_extract"].append({
            "triple": item['query_triple'],
            "drkgc_pred": item['DrKGC_Predicted'],
            "extract_pred": item['Extract_Predicted'],
            "target": item['Ground_Truth_Target']
        })

    # ---------------------------------------------------------
    # 그래프를 위한 데이터프레임 구성
    # ---------------------------------------------------------
    all_relations = set(drkgc_counts.keys()).union(set(extract_counts.keys()))
    
    plot_data = []
    for rel in all_relations:
        plot_data.append({"Relation": rel, "Count": drkgc_counts[rel], "Model": "Only_DrKGC_Correct"})
        plot_data.append({"Relation": rel, "Count": extract_counts[rel], "Model": "Only_Extract_Correct"})
    
    df = pd.DataFrame(plot_data)
    
    # 총 카운트 기준으로 Relation 정렬 (영향력이 큰 것부터 보기 위함)
    df['Total'] = df.groupby('Relation')['Count'].transform('sum')
    df = df.sort_values(by=['Total', 'Relation'], ascending=[False, True]).drop('Total', axis=1)
    
    # FB15k-237처럼 Relation이 너무 많을 경우를 대비해 Top 30개만 추출
    top_relations = df['Relation'].drop_duplicates().head(30).tolist()
    df_top = df[df['Relation'].isin(top_relations)]

    # ---------------------------------------------------------
    # 1. 그래프 PNG 만들기
    # ---------------------------------------------------------
    plt.figure(figsize=(14, max(8, len(top_relations) * 0.4))) # 관계 수에 따라 높이 유동적 조절
    
    # seaborn 바플롯 생성 (가독성을 위해 파란색, 빨간색 지정)
    sns.barplot(
        data=df_top, 
        y="Relation", 
        x="Count", 
        hue="Model", 
        palette={"Only_DrKGC_Correct": "#3498db", "Only_Extract_Correct": "#e74c3c"}
    )
    
    plt.title(f"{dataset_name.upper()} - Correct Predictions by Relation (Top {len(top_relations)})", fontsize=16, fontweight='bold')
    plt.xlabel("Number of Hits@1", fontsize=12)
    plt.ylabel("Relation", fontsize=12)
    plt.legend(title="Model Type")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    img_filename = f"{dataset_name}_relation_comparison.png"
    plt.savefig(img_filename, dpi=300, bbox_inches='tight')
    print(f"✅ 그래프 저장 완료: {img_filename}")
    plt.close()

    # ---------------------------------------------------------
    # 2. 상세 내역 JSON 저장하기
    # ---------------------------------------------------------
    json_filename = f"{dataset_name}_relation_details.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(details, f, ensure_ascii=False, indent=4)
    print(f"✅ 상세 내역 저장 완료: {json_filename}")


if __name__ == "__main__":
    wn18rr = "wn18rr_hits1_comparison_report.json"
    fb15k237 = "fb15k237_hits1_comparison_report.json"
    count_relation(wn18rr)
    count_relation(fb15k237)