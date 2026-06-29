import os
import re
import ast
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description="Plot evaluation results.")
    parser.add_argument("--data_type", type=str, default="wn18rr", help="The type of data to process (e.g., wn18rr, fb15k237)")
    parser.add_argument("--base_dir", type=str, default="results", help="Base directory for results.")
    args = parser.parse_args()

    base_dir = args.base_dir
    ds = args.data_type
    
    # 🌟 타겟 조건 설정
    target_lm = 0.0
    target_st = 0.0
    target_al = 1.0
    target_suffixes = ["a0.0", "b0.05", "b0.01", "a1.0"] 
    metrics = ['mrr', 'hits1', 'hits3', 'hits10']

    data_list = []

    print(f"🔍 1. 결과 파일 파싱을 시작합니다... (대상: {ds}, AL={target_al})\n")
    
    ds_path = os.path.join(base_dir, ds)
    if not os.path.exists(ds_path): 
        print(f"❌ 경로를 찾을 수 없습니다: {ds_path}")
        return
        
    for folder_name in os.listdir(ds_path):
        match = re.match(r"llama3_seed(\d+)_lm([0-9.]+)_st([0-9.]+)_al([0-9.]+)(_dr)?", folder_name)
        if not match: 
            continue
            
        seed = int(match.group(1))
        lm = float(match.group(2))
        st = float(match.group(3))
        al = float(match.group(4))
        use_dr = True if match.group(5) == "_dr" else False
        
        # 타겟 조건 필터링
        if lm != target_lm or st != target_st or al != target_al:
            continue
        
        for suffix in target_suffixes:
            file_path = os.path.join(ds_path, folder_name, f"metrics_{suffix}.txt")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    dict_match = re.search(r"ranking metrics:\s*(\{.*\})", content)
                    if dict_match:
                        try:
                            metrics_dict = ast.literal_eval(dict_match.group(1))
                            metrics_dict.update({
                                'Dataset': ds,
                                'Use_DR': use_dr,
                                'Suffix': suffix
                            })
                            data_list.append(metrics_dict)
                        except Exception as e:
                            print(f"⚠️ 구문 분석 오류 ({file_path}): {e}")

    if not data_list:
        print(f"❌ 파싱할 데이터가 없습니다. ({ds}의 AL=1.0 결과가 있는지 확인하세요.)")
        return

    df = pd.DataFrame(data_list)
    df['Suffix'] = pd.Categorical(df['Suffix'], categories=target_suffixes, ordered=True)

    # Std 제거하고 평균(mean)만 계산
    agg_dict = {m: 'mean' for m in metrics}
    summary_df = df.groupby(['Dataset', 'Use_DR', 'Suffix']).agg(agg_dict).reset_index()

    # 데이터 추출을 위한 딕셔너리화 (결측치 방지)
    vals_dict = {}
    for _, row in summary_df.iterrows():
        vals_dict[(row['Use_DR'], row['Suffix'])] = {m: row[m] for m in metrics}

    os.makedirs("plot_images", exist_ok=True)
    sns.set_theme(style="whitegrid")

    # =========================================================================
    # 🎨 [최종본] 1개 그래프 + 배경색 분할 + 수평(가로) 숫자 표기
    # =========================================================================
    print(f"🎨 2. {ds.upper()} 데이터셋의 단일 통합 그래프(배경 분할)를 생성합니다...")
    
    n_metrics = len(metrics)
    bar_width = 0.18 # 여백을 위해 막대 폭 조정
    palette = sns.color_palette("Set2", n_metrics) 

    # 🌟 X축 위치 설정: 왼쪽 4개(d_r O), 오른쪽 4개(d_r X) 사이에 간격(gap)을 둡니다.
    x_pos = np.array([0, 1, 2, 3, 5, 6, 7, 8]) 
    x_labels = target_suffixes + target_suffixes # 라벨 두 번 반복

    # Y축 최대값 설정 (여유 공간 30% 추가)
    max_overall = summary_df[metrics].max().max()
    ylim_top = max_overall * 1.30 if max_overall > 0 else 0.05

    # 가로로 긴 1개의 도화지 생성
    fig, ax = plt.subplots(figsize=(24, 10))
    fig.suptitle(f"[{ds.upper()}] Performance Comparison (LM=0.0, ST=0.0, AL=1.0)", 
                 fontsize=28, fontweight='bold', y=0.98)

    # 🌟 [배경색 칠하기] axvspan을 사용하여 구역을 나눕니다.
    # 왼쪽 (WITH d_r) - 연한 파란색
    ax.axvspan(-0.6, 3.6, color='#E6F2FF', alpha=0.8, zorder=0)
    # 오른쪽 (WITHOUT d_r) - 연한 주황색/회색
    ax.axvspan(4.4, 8.6, color='#FFF0E6', alpha=0.8, zorder=0)

    # 🌟 구역별 타이틀 크게 적기
    ax.text(1.5, ylim_top * 0.95, "WITH Relation-Specific Vector (d_r)", 
            ha='center', va='top', fontsize=22, fontweight='bold', color='#1A5276', zorder=5)
    ax.text(6.5, ylim_top * 0.95, "WITHOUT Relation-Specific Vector (Base)", 
            ha='center', va='top', fontsize=22, fontweight='bold', color='#935116', zorder=5)

    # 막대 그래프 그리기
    for m_idx, metric in enumerate(metrics):
        vals = []
        for is_dr in [True, False]:
            for suf in target_suffixes:
                vals.append(vals_dict.get((is_dr, suf), {}).get(metric, np.nan))
        
        offset = (m_idx - n_metrics/2 + 0.5) * bar_width
        bars = ax.bar(x_pos + offset, vals, bar_width, 
                      label=metric.upper(), capsize=4, color=palette[m_idx], edgecolor='black', zorder=3)
        
        # 🌟 수치 표시 (가로로 똑바로: rotation=0)
        for i, bar in enumerate(bars):
            val = vals[i]
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, val + (ylim_top * 0.01), 
                        f"{val:.4f}", ha='center', va='bottom', fontsize=12, fontweight='bold', 
                        color='black', rotation=0, zorder=4)
    
    # 축 설정
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=16, fontweight='bold')
    ax.set_ylabel("Score", fontsize=20, fontweight='bold')
    ax.set_ylim(0, ylim_top)
    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=1) # Y축 점선 그리기
    
    # 범례 설정 (가운데 위, 제목 바로 아래에 가로로 넓게 배치)
    leg = ax.legend(fontsize=15, loc='upper center', bbox_to_anchor=(0.5, 0.90), ncol=4, framealpha=1.0)
    leg.set_zorder(5)
    plt.tight_layout(rect=[0, 0, 1, 0.93]) 
    save_path = f"plot_images/{ds}_AL1.0_Target_SingleGraph.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 단일 통합 그래프 저장 완료: {save_path}")

if __name__ == "__main__":
    main()