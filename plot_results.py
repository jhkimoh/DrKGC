import os
import re
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import collections

def main():
    base_dir = "results"
    
    expected_datasets = ["fb15k237", "wn18rr"]
    expected_seeds = [1213, 622, 626]
    expected_lms = [0.0, 1.0]
    expected_sts = {"fb15k237": [0.0, 0.004], "wn18rr": [0.0, 0.009]}
    
    suffixes = ["a1.0", "b0.0001", "b0.001", "b0.01", "b0.1", "b1.0", "b10.0", "b100.0", "a0.0"]
    metrics = ['mrr', 'hits1', 'hits3', 'hits10'] # 순서 지정

    data_list = []
    found_combinations = set()

    print("🔍 1. 결과 파일 파싱 및 누락 데이터 검사를 시작합니다...\n")
    
    for ds in expected_datasets:
        ds_path = os.path.join(base_dir, ds)
        if not os.path.exists(ds_path): 
            continue
            
        for folder_name in os.listdir(ds_path):
            match = re.match(r"llama3_seed(\d+)_lm([0-9.]+)_st([0-9.]+)", folder_name)
            if not match: 
                continue
                
            seed = int(match.group(1))
            lm = float(match.group(2))
            st = float(match.group(3))
            condition_name = f"LM:{lm} | ST:{st}"
            
            for suffix in suffixes:
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
                                    'Seed': seed,
                                    'Condition': condition_name,
                                    'Suffix': suffix
                                })
                                data_list.append(metrics_dict)
                                found_combinations.add((ds, seed, lm, st, suffix))
                            except Exception as e:
                                print(f"⚠️ 구문 분석 오류 ({file_path}): {e}")

    # 누락 리포트 출력
    missing_report = collections.defaultdict(list)
    for ds in expected_datasets:
        for seed, lm, st, suffix in itertools.product(expected_seeds, expected_lms, expected_sts[ds], suffixes):
            if (ds, seed, lm, st, suffix) not in found_combinations:
                key = f"[{ds.upper()}] LM:{lm} | ST:{st} | Suffix:{suffix}"
                missing_report[key].append(seed)

    if missing_report:
        print("⚠️ [누락 리포트] 다음 실험 결과가 존재하지 않습니다:")
        for key, missing_seeds in missing_report.items():
            print(f"  - {key} -> 누락된 Seed: {missing_seeds}")
        print("-" * 60 + "\n")
    else:
        print("✅ 완벽합니다! 모든 실험 조합의 파일이 존재합니다.\n")

    if not data_list:
        print("❌ 파싱할 데이터가 없습니다.")
        return

    df = pd.DataFrame(data_list)
    df['Suffix'] = pd.Categorical(df['Suffix'], categories=suffixes, ordered=True)

    agg_dict = {}
    for m in metrics:
        agg_dict[f'{m}_mean'] = (m, 'mean')
        agg_dict[f'{m}_std'] = (m, 'std')
        
    summary_df = df.groupby(['Dataset', 'Condition', 'Suffix']).agg(**agg_dict).reset_index()
    summary_df.to_csv("metrics_summary_mean_std.csv", index=False)

    os.makedirs("plot_images", exist_ok=True)
    sns.set_theme(style="whitegrid")

    # =========================================================================
    # 🎨 [기존 기능] 지표별 개별 이미지 생성 (hits1, mrr, hits3, hits10 각각)
    # =========================================================================
    print("🎨 2. 평가지표별 개별 시각화 이미지를 생성합니다...")
    for ds in expected_datasets:
        ds_summary = summary_df[summary_df['Dataset'] == ds].copy()
        if ds_summary.empty: continue
        conditions = ds_summary['Condition'].unique()
        
        for metric in metrics:
            mean_col, std_col = f'{metric}_mean', f'{metric}_std'
            max_val_in_ds = (ds_summary[mean_col].fillna(0) + ds_summary[std_col].fillna(0)).max()
            ylim_top = max_val_in_ds * 1.35 if max_val_in_ds > 0 else 0.05 

            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(22, 14))
            axes = axes.flatten()
            fig.suptitle(f"{metric.upper()} Performance: {ds.upper()}", fontsize=26, fontweight='bold', y=0.98)

            for idx, condition in enumerate(conditions):
                ax = axes[idx]
                cond_data = ds_summary[ds_summary['Condition'] == condition].sort_values('Suffix')
                x_labels = cond_data['Suffix'].astype(str).tolist()
                means, stds = cond_data[mean_col].values, cond_data[std_col].fillna(0).values 
                x_pos = np.arange(len(x_labels))
                
                bars = ax.bar(x_pos, means, yerr=stds, capsize=6, color=sns.color_palette("husl", len(x_labels)), edgecolor='black', alpha=0.8)
                
                ax.set_title(f"[{ds.upper()}] {condition}", fontsize=18, fontweight='bold', pad=15)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(x_labels, rotation=45, fontsize=12)
                ax.set_ylabel(metric.upper(), fontsize=14)
                ax.set_ylim(0, ylim_top)

                for i, bar in enumerate(bars):
                    if not np.isnan(means[i]):
                        ax.text(bar.get_x() + bar.get_width()/2, means[i] + stds[i] + (ylim_top * 0.02), 
                                f"{means[i]:.4f}\n±{stds[i]:.4f}", ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(f"plot_images/{ds}_{metric}_summary.png", dpi=150, bbox_inches='tight')
            plt.close()

    # =========================================================================
    # 🌟 [신규 추가 기능] 4개 지표(MRR, HITS@1, 3, 10) 통합 마스터 요약본 생성
    # =========================================================================
    print("🎨 3. 데이터셋별 4개 지표 통합(All-in-One) 시각화 이미지를 생성합니다...")
    
    n_metrics = len(metrics)
    bar_width = 0.8 / n_metrics # 4개의 막대가 한 구간에 들어가야 하므로 너비 쪼개기
    palette = sns.color_palette("Set2", n_metrics) # 4개 지표를 구분할 4가지 색상

    for ds in expected_datasets:
        ds_summary = summary_df[summary_df['Dataset'] == ds].copy()
        if ds_summary.empty: continue
        conditions = ds_summary['Condition'].unique()
        
        # 4개 지표 전체를 통틀어 가장 높은 값 탐색 (Y축 높이 설정용, 주로 HITS@10이 가장 높음)
        max_overall = 0
        for m in metrics:
            m_max = (ds_summary[f'{m}_mean'].fillna(0) + ds_summary[f'{m}_std'].fillna(0)).max()
            if m_max > max_overall: max_overall = m_max
        
        # 글씨가 세로로 길게 들어가므로 천장 여백을 넉넉하게 50% 부여
        ylim_top = max_overall * 1.5 if max_overall > 0 else 0.05

        # 가로 폭을 28로 넉넉하게 확장
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(28, 16))
        axes = axes.flatten()
        fig.suptitle(f"ALL METRICS OVERVIEW: {ds.upper()}", fontsize=28, fontweight='bold', y=0.98)

        for idx, condition in enumerate(conditions):
            ax = axes[idx]
            cond_data = ds_summary[ds_summary['Condition'] == condition].sort_values('Suffix')
            x_labels = cond_data['Suffix'].astype(str).tolist()
            x_pos = np.arange(len(x_labels))
            
            # 4개의 지표 막대를 약간씩 빗겨가며 그리기 (Grouped Bar Chart)
            for m_idx, metric in enumerate(metrics):
                means = cond_data[f'{metric}_mean'].values
                stds = cond_data[f'{metric}_std'].fillna(0).values
                
                # 막대의 X 좌표 오프셋 계산 (중앙을 기준으로 좌우로 배치)
                offset = (m_idx - n_metrics/2 + 0.5) * bar_width
                
                bars = ax.bar(x_pos + offset, means, bar_width, yerr=stds, 
                              label=metric.upper(), capsize=3, color=palette[m_idx], edgecolor='black', alpha=0.85)
                
                # 막대 위 수치 기록 (세로쓰기)
                for i, bar in enumerate(bars):
                    if not np.isnan(means[i]):
                        label_text = f"{means[i]:.3f}\n±{stds[i]:.3f}" # 좁으므로 소수점 3자리로 단축
                        ax.text(bar.get_x() + bar.get_width()/2, means[i] + stds[i] + (ylim_top * 0.015), 
                                label_text, ha='center', va='bottom', fontsize=8, fontweight='bold', 
                                color='black', rotation=90) # 🌟 90도 회전
            
            ax.set_title(f"[{ds.upper()}] {condition}", fontsize=20, fontweight='bold', pad=15)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, rotation=45, fontsize=14)
            ax.set_ylabel("Score", fontsize=16)
            ax.set_ylim(0, ylim_top)
            ax.legend(fontsize=12, loc='upper left') # 우측 상단 텍스트와 겹치지 않게 좌측 상단 배치

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = f"plot_images/{ds}_all_metrics_summary.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"✅ 통합 그래프 저장 완료: {save_path}")

    print("\n🎉 모든 분석 및 요약본 생성이 완료되었습니다! 'plot_images' 폴더를 확인해 보세요.")

if __name__ == "__main__":
    main()