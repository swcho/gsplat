# `Parser(normalize=True)`는 무엇을 하는가?

카메라 위치 기준 similarity 변환으로 월드 좌표를 정규화한다. 내부적으로 `similarity_from_cameras`와 `align_principal_axes`를 사용한다.

- [hi.md](hi.md) — 고교 교과 과정에서 출발한 단계별 설명

---

## 0. 한 줄 요약

COLMAP이 내놓은 월드 좌표계는 **회전·위치·스케일이 제멋대로**다. `normalize=True`는 그 좌표계를
"카메라들이 원점 주변에 반경 약 1로 모여 있고, 씬의 얇은 방향이 z축인" **표준 자세로 강제 이동**시킨다.
학습률·밀도화 임계값 같은 하이퍼파라미터를 데이터셋마다 다시 튜닝하지 않기 위한 전처리다.

## 1. 왜 필요한가 — SfM 좌표계의 자유도

COLMAP(Structure-from-Motion)은 이미지들만 보고 카메라 포즈와 sparse 포인트를 복원한다.
그런데 사진만으로는 **절대적인 위치·방향·크기를 알 수 없다.**

- 씬 전체를 옆으로 100 m 옮겨도 → 모든 사진이 똑같이 찍힌다.
- 씬 전체를 뒤집어 회전시켜도 → 똑같다.
- 씬 전체를 2배로 키우고 카메라 간격도 2배로 늘려도 → 똑같다 (원근 투영은 크기와 거리의 비율만 본다).

즉 SfM 결과는 **similarity 변환(회전 3 + 평행이동 3 + 균등 스케일 1 = 7 자유도)** 만큼 임의성이 있다.
같은 미술관을 두 번 찍어 COLMAP을 돌리면 좌표 단위가 한 번은 0.3, 한 번은 47이 될 수 있다.

이 임의성을 그대로 두면 곤란한 이유:

| 학습 요소 | 씬 크기에 의존하는 이유 |
|---|---|
| `means`의 학습률 `1.6e-4 * scene_scale` | 위치 갱신량은 "씬 크기의 몇 %"여야 의미가 있다 |
| `grow_scale3d = 0.01`, `prune_scale3d = 0.1` (`gsplat/strategy/default.py:101,103`) | Gaussian이 "크다/작다"는 판단은 씬 대비 상대 크기 |
| MCMC noise, depth loss, 뷰어의 초기 카메라·바닥면 | z축이 위쪽이라는 가정을 쓴다 |

그래서 gsplat은 좌표계를 먼저 표준화하고(`normalize=True`), 남은 스케일 정보를 `parser.scene_scale`
하나로 뽑아 학습률·임계값에 곱해 쓴다. `simple_trainer.py:104`에서 `normalize_world_space: bool = True`가
기본값이고, `simple_trainer.py:458`에서 `scene_scale = parser.scene_scale * 1.1 * global_scale`로 쓰인다.

## 2. 코드 흐름

`examples/datasets/colmap.py:275-303`, 실제 변환 함수는 `examples/datasets/normalize.py`에 있다.

```python
# Normalize the world space.
if normalize:
    T1 = similarity_from_cameras(camtoworlds)     # (a) 자세 + 중심 + 스케일
    camtoworlds = transform_cameras(T1, camtoworlds)
    points = transform_points(T1, points)

    T2 = align_principal_axes(points)             # (b) 포인트클라우드 PCA 정렬
    camtoworlds = transform_cameras(T2, camtoworlds)
    points = transform_points(T2, points)

    transform = T2 @ T1

    # 위아래 뒤집힘 보정: 바닥이 찍혔다면 아래쪽에 점이 더 많다는 가정
    if np.median(points[:, 2]) > np.mean(points[:, 2]):
        T3 = np.diag([1.0, -1.0, -1.0, 1.0])      # x축 180도 회전
        camtoworlds = transform_cameras(T3, camtoworlds)
        points = transform_points(T3, points)
        transform = T3 @ transform
else:
    transform = np.eye(4)
```

핵심은 **카메라 포즈와 SfM 포인트에 똑같은 4×4 행렬을 적용**한다는 것이다.
둘을 같은 변환으로 옮기므로 "어느 카메라에서 어느 점이 어떻게 보이는가"는 전혀 변하지 않는다.
바뀌는 것은 숫자를 적는 좌표계뿐이다. 최종 누적 변환은 `self.transform`(4×4)에 보존되어,
학습 결과를 원래 COLMAP 좌표로 되돌릴 수 있다.

## 3. `similarity_from_cameras` — 카메라만 보고 좌표계 정하기

`normalize.py:19-78`. 이름 그대로 **카메라 포즈 배열 `c2w` [N,4,4]만 입력**으로 받아 4×4 similarity 행렬을 만든다.
포인트는 쓰지 않는다(포인트가 지저분해도 카메라 궤적은 비교적 신뢰할 수 있으므로).

### (1) 위쪽(up) 축 정렬

```python
ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)   # 각 카메라의 up 방향(월드 좌표)
world_up = np.mean(ups, axis=0); world_up /= np.linalg.norm(world_up)
```

OpenCV 카메라 규약에서 카메라 좌표계의 **+y가 아래쪽**이므로, `-y` 방향이 그 카메라의 "위"다.
`R @ [0,-1,0]`은 그 위 방향을 월드 좌표로 옮긴 벡터다. 사람이 카메라를 들고 찍으면 대부분
비슷하게 세워서 찍으므로, 이 벡터들을 **평균 내면 씬의 실제 위쪽**이 추정된다.

그 다음 Rodrigues 공식으로 `world_up`을 고정 기준 벡터 `up_camspace = (0,-1,0)`에 정확히 겹치는 회전을 만든다.

```python
c = (up_camspace * world_up).sum()          # 내적 = cos(각도)
cross = np.cross(world_up, up_camspace)     # 외적 = 회전축 * sin(각도)
R_align = np.eye(3) + skew + (skew @ skew) / (1 + c)
```

`c == -1`(정확히 반대 방향)이면 위 공식이 0으로 나누므로, x축 180도 회전으로 예외 처리한다.

### (2) 중심 이동 — `center_method="focus"`

카메라 위치의 평균을 원점으로 삼는 순진한 방법 대신, 기본값은 **"카메라들이 바라보는 곳"** 을 원점으로 잡는다.

```python
nearest = t + (fwds * -t).sum(-1)[:, None] * fwds   # 각 카메라 중심 광선에서 원점에 가장 가까운 점
translate = -np.median(nearest, axis=0)
```

`fwds`는 각 카메라의 시선 방향(OpenCV에서 +z가 앞). 카메라 $i$의 중심 광선 위 점 중
현재 원점에 가장 가까운 점을 구한 뒤, 그 점들의 **중앙값(median)** 을 새 원점으로 삼는다.
평균이 아니라 중앙값을 쓰는 이유는 포즈 하나가 크게 틀려도 결과가 흔들리지 않게 하기 위함이다(robust).
`center_method="poses"`를 주면 카메라 위치 자체의 중앙값을 쓴다.

### (3) 스케일 재조정

```python
scale_fn = np.max if strict_scaling else np.median
scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))
transform[:3, :] *= scale
```

새 원점에서 각 카메라까지의 거리를 재고, 그 **중앙값이 1이 되도록** 전체를 나눈다.
결과적으로 `x' = s (R_{align} x + \text{translate})` 형태의 similarity 변환이 완성된다.
회전·평행이동·균등 스케일만 쓰므로 각도와 형상 비율은 그대로 보존된다 — 씬이 찌그러지지 않는다.

## 4. `align_principal_axes` — 포인트클라우드 PCA로 축 고정

`normalize.py:81-112`. 여기서는 카메라가 아니라 **SfM 포인트**를 쓴다. 사실상 PCA다.

1. `centroid = np.median(point_cloud, axis=0)` — 평균이 아닌 중앙값(이상치 방어)
2. 중심 이동 후 `np.cov`로 3×3 공분산 행렬 계산
3. `np.linalg.eigh`로 고유값/고유벡터 → 고유값 **내림차순** 정렬
4. `np.linalg.det(eigenvectors) < 0`이면 첫 열의 부호를 뒤집어 **오른손 좌표계(회전, 반사 아님)** 유지
5. `transform = [R | -R·centroid]`, `R = eigenvectors.T`

정렬 순서가 내림차순이므로 새 좌표계에서 **x = 분산이 가장 큰 방향, z = 분산이 가장 작은 방향**이다.
실내·실외 장면은 보통 수평으로 넓게 퍼지고 수직으로는 얇으므로, 분산이 가장 작은 축이
대체로 **중력 방향**이 된다. 즉 z축이 "위/아래"가 되고 바닥면이 xy 평면 근처에 놓인다.

## 5. 뒤집힘 보정 (T3)

PCA는 축의 **방향**만 정하고 **부호**는 정하지 못한다(고유벡터에 −1을 곱해도 여전히 고유벡터).
그래서 z축이 위를 향하는지 아래를 향하는지 알 수 없다. 코드는 통계적 힌트를 쓴다.

```python
if np.median(points[:, 2]) > np.mean(points[:, 2]):   # 분포가 위로 치우침 → 뒤집힘
    T3 = 180도 x축 회전
```

바닥이 찍힌 데이터셋은 아래쪽(바닥·지면)에 점이 촘촘하고 위쪽(하늘·천장·먼 벽)에 점이 드물게 길게
뻗는다. 이런 **왼쪽으로 긴 꼬리** 분포에서는 `median > mean`이면 방향이 반대라고 판단해 x축 180도
회전(`y, z` 부호 반전)으로 뒤집는다. 어디까지나 휴리스틱이라 하늘만 찍은 데이터에서는 틀릴 수 있다.

## 6. 적용 함수의 디테일

```python
def transform_points(matrix, points):
    return points @ matrix[:3, :3].T + matrix[:3, 3]

def transform_cameras(matrix, camtoworlds):
    camtoworlds = np.einsum("nij, ki -> nkj", camtoworlds, matrix)   # matrix @ c2w
    scaling = np.linalg.norm(camtoworlds[:, 0, :3], axis=1)
    camtoworlds[:, :3, :3] = camtoworlds[:, :3, :3] / scaling[:, None, None]
    return camtoworlds
```

- 포인트는 그냥 아핀 적용.
- 카메라는 $M C$ 로 좌합성한다(월드 쪽이 바뀌었으므로 camera-to-world 행렬의 왼쪽에 곱한다).
- **주의**: similarity 변환에는 스케일 $s$가 들어 있어 곱하면 회전 블록이 $sR$이 된다.
  회전 블록은 정규직교여야 하므로 마지막 두 줄에서 첫 행의 노름으로 나눠 $s$를 제거한다.
  카메라 **위치**(4번째 열)에는 스케일이 남고, **자세**만 정규직교로 복원되는 것이 핵심이다.

## 7. 결과로 얻는 `scene_scale`

`colmap.py:436-440`, 정규화가 끝난 카메라 위치로 계산된다.

```python
camera_locations = camtoworlds[:, :3, 3]
scene_center = np.mean(camera_locations, axis=0)
dists = np.linalg.norm(camera_locations - scene_center, axis=1)
self.scene_scale = np.max(dists)          # 가장 멀리 떨어진 카메라까지의 거리
```

`similarity_from_cameras`에서 카메라 거리의 **중앙값**을 1로 맞췄으므로, **최댓값**인 `scene_scale`은
보통 1 근처(대략 1~3)의 값이 된다. 워크스루의 `scene_scale = parser.scene_scale * 1.1`은
`simple_trainer.py:458`과 같은 식이며, 여기서 1.1은 카메라 궤적 바깥까지 씬이 조금 더 있다고 보는 여유분이다.

이 값이 곱해지는 곳:

- `means` 학습률: `1.6e-4 * scene_scale` (`simple_trainer.py:336`)
- 랜덤 초기화 범위: `init_extent * scene_scale` (`simple_trainer.py:315`)
- 밀도화 판단: `strategy.initialize_state(scene_scale=...)` → `grow_scale3d * scene_scale`,
  `prune_scale3d * scene_scale` (`gsplat/strategy/default.py:301,355`)
- 뷰어/평가용 `bounds * scene_scale` (`simple_trainer.py:1327`)

즉 **정규화 → `scene_scale` 하나 → 모든 길이 단위 하이퍼파라미터**로 전파되는 구조다.
`normalize=False`로 두면 `transform = I`가 되고, `scene_scale`이 COLMAP의 임의 단위로 나오므로
기본 하이퍼파라미터가 전혀 맞지 않게 된다(학습률이 100배 작거나 커진다).

## 8. 정리

| 단계 | 함수 | 입력 | 하는 일 |
|---|---|---|---|
| T1 | `similarity_from_cameras` | 카메라 포즈만 | 위 방향 정렬 + 시선 교점 중앙값을 원점으로 + 카메라 거리 중앙값 = 1 |
| T2 | `align_principal_axes` | SfM 포인트 | PCA로 축 정렬(z = 분산 최소 방향), 오른손계 보정 |
| T3 | 인라인 휴리스틱 | SfM 포인트의 z 분포 | `median > mean`이면 x축 180도 회전으로 위아래 보정 |
| 저장 | — | — | `self.transform = T3 @ T2 @ T1`, `self.scene_scale` |

자주 헷갈리는 점:

- **정규화는 이미지 픽셀값이 아니라 월드 좌표를 건드린다.** 이미지 정규화(0~1 스케일링)와 무관하다.
- **similarity 변환이므로 씬의 모양은 보존된다.** affine/비등방 스케일이 아니다.
- **T1은 카메라만, T2는 포인트만 본다.** 두 정보원을 순서대로 쓰는 2단 구조다.
- **`normalize`의 기본값은 `Parser`에서 `False`**지만, `simple_trainer`의
  `normalize_world_space` 기본값이 `True`라 실제 학습에서는 켜져 있다.
