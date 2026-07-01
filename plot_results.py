import os
import re
import ast
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description="Plot dual evaluation results (TransE & RotatE).")
    parser.add_argument("--data_type", type=str, default="wn18rr", help="The type of data to process (e.g., wn18rr, fb15k237)")
    parser.add_argument("--base_dir", type=str, default="results", help="Base directory for results.")
    args = parser.parse_args()

    base_dir = args.base_dir
    ds = args.data_type
    
    # 🌟 타겟 조건 설정
    target_lm = 0.0
    target_st = 0.0
    target_al = 1.0
    # 🌟 요청하신 X축 정렬 순서 반영
    target_suffixes = ["a0.0", "b0.05", "b0.01", "a1.0"] 
    metrics = ['mrr', 'hits1', 'hits3', 'hits10']

    data_list = []

    print(f"🔍 1. 결과 파일 파싱을 시작합니다... (대상: {ds}, AL={target_al})\n")
    
    ds_path = os.path.join(base_dir, ds)
    if not os.path.exists(ds_path): 
        print(f"❌ 경로를 찾을 수 없습니다: {ds_path}")
        return
        
    for folder_name in os.listdir(ds_path):
        match = re.match(r"llama3_seed(\d+)_lm([0-9.]+)_st([0-9.]+)_al([0-9.]+)(_dr)?_([a-zA-Z]+)", folder_name, re.IGNORECASE)
        if not match: 
            continue
            
        seed = int(match.group(1))
        lm = float(match.group(2))
        st = float(match.group(3))
        al = float(match.group(4))
        use_dr = True if match.group(5) == "_dr" else False
        kge_model = match.group(6)
        
        if kge_model.lower() == 'transe':
            kge_model = 'TransE'
        elif kge_model.lower() == 'rotate':
            kge_model = 'RotatE'
        else:
            continue
            
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
                                'Suffix': suffix,
                                'KGE_Model': kge_model
                            })
                            data_list.append(metrics_dict)
                        except Exception as e:
                            print(f"⚠️ 구문 분석 오류 ({file_path}): {e}")

    if not data_list:
        print(f"❌ 파싱할 데이터가 없습니다.")
        return

    df = pd.DataFrame(data_list)
    df['Suffix'] = pd.Categorical(df['Suffix'], categories=target_suffixes, ordered=True)

    agg_dict = {m: 'mean' for m in metrics}
    summary_df = df.groupby(['Dataset', 'KGE_Model', 'Use_DR', 'Suffix']).agg(agg_dict).reset_index()

    vals_dict = {}
    for _, row in summary_df.iterrows():
        vals_dict[(row['KGE_Model'], row['Use_DR'], row['Suffix'])] = {m: row[m] for m in metrics}

    os.makedirs("plot_images", exist_ok=True)
    sns.set_theme(style="whitegrid")

    # =========================================================================
    # 🎨 가로(Side-by-Side) 듀얼 그래프 생성 
    # =========================================================================
    print(f"🎨 2. {ds.upper()} 데이터셋의 가로 듀얼 그래프(TransE | RotatE)를 생성합니다...")
    
    n_metrics = len(metrics)
    bar_width = 0.20 
    palette = sns.color_palette("Set2", n_metrics) 

    x_pos = np.arange(len(target_suffixes)) 
    x_labels = target_suffixes 

    max_overall = summary_df[metrics].max().max()
    ylim_top = max_overall * 1.30 if max_overall > 0 else 0.05

    # 🌟 1행 2열로 가로로 긴(Side-by-side) 도화지 생성, Y축 스케일 공유(sharey)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 10), sharey=True)
    
    # 전체 제목 위치 상향 조정
    fig.suptitle(f"[{ds.upper()}] Performance Comparison\n(LM=0.0, ST=0.0, AL=1.0, WITH d_r)", 
                 fontsize=28, fontweight='bold', y=1.05)

    models_to_plot = ['TransE', 'RotatE']

    for idx, model_name in enumerate(models_to_plot):
        ax = axes[idx]
        
        ax.set_title(f"Model: {model_name}", fontsize=24, fontweight='bold', pad=15)
        ax.set_facecolor('#F8F9FA') 
        
        bars_list = []
        for m_idx, metric in enumerate(metrics):
            vals = []
            for suf in target_suffixes:
                vals.append(vals_dict.get((model_name, True, suf), {}).get(metric, np.nan))
            
            offset = (m_idx - n_metrics/2 + 0.5) * bar_width
            bars = ax.bar(x_pos + offset, vals, bar_width, 
                          label=metric.upper(), capsize=4, color=palette[m_idx], edgecolor='black', zorder=3)
            
            if idx == 0: 
                bars_list.append(bars)
            
            for i, bar in enumerate(bars):
                val = vals[i]
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width()/2, val + (ylim_top * 0.01), 
                            f"{val:.4f}", ha='center', va='bottom', fontsize=12, fontweight='bold', 
                            color='black', rotation=0, zorder=4)
        
        # 🌟 축 설정 (가로 배치이므로 X축 라벨은 양쪽 다 표시)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=18, fontweight='bold')
        
        # 🌟 Y축 라벨은 왼쪽(첫 번째) 그래프에만 표시
        if idx == 0:
            ax.set_ylabel("Score", fontsize=20, fontweight='bold')
            
        ax.set_ylim(0, ylim_top)
        ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=1)

    # 🌟 공통 범례 설정 (위쪽 가운데 배치)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 0.92), ncol=4, framealpha=1.0)
    
    # 레이아웃 간격 조정 (wspace로 좌우 간격 지정)
    plt.subplots_adjust(wspace=0.1, top=0.83) 
    
    save_path = f"plot_images/{ds}_DualGraph_Horizontal_AL1.0_DR_Only.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 가로 듀얼 그래프 저장 완료: {save_path}")

if __name__ == "__main__":
    main()