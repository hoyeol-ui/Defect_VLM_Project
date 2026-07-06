import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# =============================================================================
# [1] 기본 설정 및 폰트 세팅 (Mac OS 기준)
# =============================================================================
plt.rcParams['font.family'] = 'AppleGothic'  # Mac 한글 폰트 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path("/Users/hy/PycharmProjects/PythonProject/Defect_VLM_Project")
OUTPUT_BASE_DIR = PROJECT_ROOT / "outputs"


def get_latest_results_file():
    """outputs 폴더 내에서 가장 최근에 생성된 JSON 파일을 동적으로 찾습니다."""
    json_files = list(OUTPUT_BASE_DIR.glob("**/pilot_grounded_consistency_results.json"))
    if not json_files:
        raise FileNotFoundError("결과 JSON 파일을 찾을 수 없습니다. VLM 추론을 먼저 실행해주세요.")
    # 파일 수정 시간 기준으로 가장 최신 파일 선택
    latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
    return latest_file


# =============================================================================
# [2] 능동 학습 획득 함수 (Acquisition Function) 정의
# =============================================================================
def calculate_priority_score(row):
    """
    VLM 점수를 기반으로 라벨링 우선순위(Priority Score)를 계산합니다.
    - 일관성이 낮을수록 (1 - consistency) 점수 증가
    - 정합성이 낮을수록 (1 - groundedness/2.0) 점수 증가
    - 두 지표를 결합하여 최종 불확실성(Uncertainty) 수치화 (Max: 2.0)
    """
    unc_consistency = 1.0 - row['consistency_score']
    unc_groundedness = 1.0 - (row['groundedness_score'] / 2.0)

    # 💡 추후 실험에 따라 가중치(alpha, beta)를 조절할 수 있습니다.
    alpha, beta = 1.0, 1.0
    return (alpha * unc_consistency) + (beta * unc_groundedness)


# =============================================================================
# [3] 데이터 로드 및 분석 메인 루틴
# =============================================================================
def main():
    # 1. 최신 결과 파일 로드
    latest_json = get_latest_results_file()
    print(f"[*] 데이터 로드 완료: {latest_json.parent.name}")

    with open(latest_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. 분석을 위해 Pandas DataFrame으로 변환
    records = []
    for item in data:
        records.append({
            "image_name": item["image_name"],
            "dataset_type": item["dataset_type"],
            "consistency_score": item["consistency_score"],
            "groundedness_score": item["groundedness"]["total_score"]
        })

    df = pd.DataFrame(records)

    # 3. 우선순위 점수 계산 및 랭킹 정렬
    df["priority_score"] = df.apply(calculate_priority_score, axis=1)
    df = df.sort_values(by="priority_score", ascending=False).reset_index(drop=True)

    # =========================================================================
    # [4] 시각화 (논문 방어용 Scatter Plot)
    # =========================================================================
    plt.figure(figsize=(10, 7))

    # 산점도 그리기 (데이터셋별 색상 구분)
    colors = {'NEU-DET': 'blue', 'GC10-DET': 'red'}
    for dataset in df['dataset_type'].unique():
        subset = df[df['dataset_type'] == dataset]
        plt.scatter(
            subset['consistency_score'],
            subset['groundedness_score'],
            label=dataset,
            color=colors.get(dataset, 'gray'),
            alpha=0.7,
            s=100,  # 마커 크기
            edgecolors='k'
        )

    # 각 점에 이미지 이름 텍스트 달기
    for i, row in df.iterrows():
        plt.text(row['consistency_score'] + 0.005, row['groundedness_score'] + 0.02,
                 row['image_name'].split('.')[0], fontsize=8, alpha=0.8)

    # 4사분면 의미 구역 강조 (Hard Sample 구역)
    # 일관성도 낮고 정합성도 낮은 좌측 하단 구역에 박스 하이라이트
    plt.axvline(x=0.65, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    rect = patches.Rectangle((0.5, -0.2), 0.15, 1.2, linewidth=2, edgecolor='red', facecolor='red', alpha=0.1)
    plt.gca().add_patch(rect)
    plt.text(0.51, 0.8, "최우선 라벨링 타겟\n(High Priority)", color='red', weight='bold')

    plt.title("VLM 복합 지표 분포 (Consistency vs Groundedness)", fontsize=16)
    plt.xlabel("설명 일관성 (Consistency Score) ➔ 낮을수록 불확실", fontsize=12)
    plt.ylabel("시각적 정합성 (Groundedness Score) ➔ 낮을수록 환각", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    # 그래프 저장
    plot_path = latest_json.parent / "active_learning_strategy_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n[!] 분석 그래프 저장 완료: {plot_path.name}")

    # =========================================================================
    # [5] 터미널 출력: Active Learning 라벨링 추천 리스트
    # =========================================================================
    print("\n" + "=" * 60)
    print(" 🔥 VLM 기반 Active Learning 추천 큐 (우선순위 Top 5) 🔥")
    print("=" * 60)
    for i, row in df.head(5).iterrows():
        print(f"{i + 1}위. {row['image_name']} ({row['dataset_type']})")
        print(f"    - 통합 Priority 점수: {row['priority_score']:.4f}")
        print(f"    - 세부 지표: 일관성 {row['consistency_score']:.4f} | 정합성 {row['groundedness_score']}/2.0")
    print("=" * 60)


if __name__ == "__main__":
    main()