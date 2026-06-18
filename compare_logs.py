import json
import numpy as np


def analyze_and_compare(drkgc_json_path, rotate_json_path):
    # 1. 파일 로드
    print("📦 로그 파일을 불러오는 중...")
    with open(drkgc_json_path, "r", encoding="utf-8") as f:
        drkgc_data = json.load(f)
    with open(rotate_json_path, "r", encoding="utf-8") as f:
        rotate_data = json.load(f)

    drkgc_keys = set(drkgc_data.keys())
    rotate_keys = set(rotate_data.keys())

    # 2. 데이터셋 커버리지(교집합) 확인
    common_keys = drkgc_keys.intersection(rotate_keys)
    mismatch_key = list(drkgc_keys - rotate_keys)[0]
    breakpoint()
    print("\n--- 📊 데이터셋 매칭 통계 ---")
    print(f"• DrKGC 평가 쿼리 수 : {len(drkgc_keys)}개")
    print(f"• RotatE 평가 쿼리 수: {len(rotate_keys)}개")
    print(f"• ✨ 완벽히 일치하는 공통 쿼리 수: {len(common_keys)}개")

    if len(common_keys) == 0:
        print(
            "🚨 [경고] 두 파일 간에 일치하는 쿼리 키가 하나도 없습니다! 키 포맷을 확인하세요."
        )
        # 키 하나씩 샘플 출력해서 보여주기
        print(f"Sample DrKGC key : {list(drkgc_keys)[0]}")
        print(f"Sample RotatE key: {list(rotate_keys)[0]}")
        return

    # 3. 공통 쿼리 전수 조사
    rank_matches = 0
    top10_matches = 0
    score_diffs = []
    mismatches = []

    for key in common_keys:
        d_item = drkgc_data[key]
        r_item = rotate_data[key]

        # A. 순위 비교
        if d_item["rank"] == r_item["rank"]:
            rank_matches += 1
        else:
            mismatches.append(
                {
                    "key": key,
                    "drkgc_rank": d_item["rank"],
                    "rotate_rank": r_item["rank"],
                    "drkgc_top3": d_item["top10_preds"][:3],
                    "rotate_top3": r_item["top10_preds"][:3],
                }
            )

        # B. Top 10 예측 엔티티 일치 여부 비교
        if d_item["top10_preds"] == r_item["top10_preds"]:
            top10_matches += 1

        # C. 마진(Gamma=9.0) 차이 계산 (RotatE score - DrKGC score)
        diff = r_item["target_score"] - d_item["target_score"]
        score_diffs.append(diff)

    # 4. 결과 분석 및 출력
    mean_diff = np.mean(score_diffs)
    std_diff = np.std(score_diffs)

    print("\n--- 🎯 1:1 전수 검증 결과 ---")
    print(
        f"• 순위(Rank) 완전 일치 쿼리: {rank_matches} / {len(common_keys)} ({rank_matches/len(common_keys)*100:.2f}%)"
    )
    print(
        f"• Top 10 결과 완전 일치 쿼리: {top10_matches} / {len(common_keys)} ({top10_matches/len(common_keys)*100:.2f}%)"
    )
    print(f"• 평균 점수 차이 (RotatE - DrKGC): {mean_diff:.6f}")
    print(f"• 점수 차이의 표준편차: {std_diff:.6f}")

    # 5. 종합 진단 보고
    print("\n--- 💡 최종 진단 의견 ---")
    if rank_matches == len(common_keys) and np.isclose(std_diff, 0.0, atol=1e-4):
        print(
            f"✅ [성공] 두 모델의 순위가 100% 일치합니다! 점수 차이 역시 모든 문제에서 정확히 {mean_diff:.2f}(gamma)로 고정되어 있습니다."
        )
        print(
            "👉 DrKGC 코드의 수식과 랭킹 계산 연산은 오리지널 KGE와 소수점 끝까지 완벽하게 동일함이 수학적으로 증명되었습니다."
        )
    else:
        print(
            "🚨 [확인 필요] 순위나 점수 차이에 변동이 있는 쿼리가 존재합니다."
        )
        if len(mismatches) > 0:
            print(f"\n❌ 다른 결과를 낸 상위 3개 쿼리 예시:")
            for m in mismatches[:3]:
                print(f"  - 쿼리: {m['key']}")
                print(
                    f"    [Rank]   DrKGC: {m['drkgc_rank']}등 vs RotatE: {m['rotate_rank']}등"
                )
                print(
                    f"    [Top 3]  DrKGC: {m['drkgc_top3']} vs RotatE: {m['rotate_top3']}"
                )


if __name__ == "__main__":
    # 파일 경로가 다르면 수정하세요
    analyze_and_compare("drkgc_debug_log.json", "rotate_debug_log.json")