import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# (기존에 작성하신 데이터 로드 코드 가정)
dataset_path = "wn18rr"
base = Path("dataset/"+dataset_path)
with (base / "train.json").open("r", encoding="utf-8") as f:
    train_json = json.load(f)

if dataset_path =="wn18rr":
    relation_len = 11
    entity_len = 40943
else:
    relation_len = 237
    entity_len = 14541

rel_counts = np.zeros(relation_len, dtype=int)
ent_counts = np.zeros(entity_len, dtype=int)

print("Counting IDs...")
for item in train_json:
    h, r, t = item['triple_id']
    rel_counts[r] += 1
    ent_counts[h] += 1
    ent_counts[t] += 1

# ==========================================
# 1. 최솟값 / 최댓값 계산 및 출력 파트
# ==========================================
# 등장 횟수가 0보다 큰(실제로 등장한) 인덱스들만 추출
active_relations = np.where(rel_counts > 0)[0]
active_entities = np.where(ent_counts > 0)[0]

active_rel_counts = rel_counts[rel_counts > 0]
active_ent_counts = ent_counts[ent_counts > 0]

print("-" * 40)
if len(active_relations) > 0:
    print(f"[Relation ID] 최소 ID: {active_relations.min():>5} / 최대 ID: {active_relations.max():>5}")
    print(f"[Relation 빈도] 최소 빈도: {active_rel_counts.min():>5} / 최대 빈도: {active_rel_counts.max():>5}")
else:
    print("[Relation] 등장한 데이터가 없습니다.")

print("-" * 40)

if len(active_entities) > 0:
    print(f"[Entity ID]   최소 ID: {active_entities.min():>5} / 최대 ID: {active_entities.max():>5}")
    print(f"[Entity 빈도]   최소 빈도: {active_ent_counts.min():>5} / 최대 빈도: {active_ent_counts.max():>5}")
else:
    print("[Entity] 등장한 데이터가 없습니다.")
print("-" * 40)

# ==========================================
# 2. Relation 분포 시각화 (X축: ID)
# ==========================================
print("Saving Relation Bar Chart...")
plt.figure(figsize=(10, 6))
plt.bar(range(relation_len), rel_counts, color='skyblue', edgecolor='black')
plt.title(f'{dataset_path} Distribution of Relation IDs', fontsize=15)
plt.xlabel('Relation ID', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(range(relation_len)) 
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('images/'+dataset_path+'_relation_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# 3. Entity 분포 시각화 (X축: ID)
# ==========================================
print("Saving Entity Plot (ID vs Frequency)...")
plt.figure(figsize=(15, 6)) # 4만개의 X축을 표현하기 위해 가로로 길게 설정

# 4만개를 bar로 그리면 너무 무거워지고 깨질 수 있으므로, 
# 얇은 선(line)이나 width=1 인 bar를 사용합니다. 여기서는 빽빽한 bar 형태를 흉내냅니다.
plt.bar(range(entity_len), ent_counts, color='lightcoral', width=1.0)

plt.title(f'{dataset_path} Distribution of Entity IDs', fontsize=15)
plt.xlabel('Entity ID (0 to 40942)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# 앞선 질문에서 다룬 롱테일 법칙 때문에 특정 노드만 빈도가 수만 번일 수 있습니다.
# Y축의 극단적인 격차를 줄여서 잘잘한 노드들도 보이게 하려면 아래 로그 스케일 주석을 해제하세요.
plt.yscale('log')

plt.xlim(0, entity_len) # X축 범위 고정
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('images/'+dataset_path+'_entity_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("Done! Check the saved images.")