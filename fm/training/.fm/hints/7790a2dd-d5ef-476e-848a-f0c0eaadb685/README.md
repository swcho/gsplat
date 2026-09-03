# `parser.scene_scale`은 왜 중요한 값인가?

> **Q.** `parser.scene_scale`은 왜 중요한 값인가?
>
> **A.** 씬의 대략적 크기를 나타내며, 이후 learning rate와 밀도화 임계값의 기준 단위가 된다. 씬 크기에 무관하게 동작하도록 만들어 주는 스케일 인자다.

관련 소스: `examples/datasets/colmap.py:437~440` (계산), `examples/datasets/normalize.py:19~78` (`similarity_from_cameras`), `examples/simple_trainer.py:458` (`* 1.1 * global_scale`), `:315`·`:336` (초기화·lr), `:512` (전략 상태), `gsplat/strategy/default.py:301`·`:355` (밀도화 임계값), `gsplat/losses.py:224` (depth loss), 워크스루 1단계(`training_walkthrough.py` L89~108).

---

## 0. 한 문장 요약

COLMAP이 복원한 좌표계는 **단위가 없다**. `scene_scale`은 "이 씬에서 1이란 얼마인가"를 카메라 배치로부터 추정한 **단 하나의 스칼라**이고, 3DGS의 하이퍼파라미터 중 *길이 차원을 가진 것 전부*를 이 스칼라로 나눠 무차원화한 덕분에 `1.6e-4`, `0.01`, `0.1` 같은 기본값이 방 한 칸짜리 씬과 야외 정원 씬에서 똑같이 통한다.

---

## 1. 어떻게 계산되나 — "카메라들이 퍼져 있는 반경"

`Parser.__init__` 맨 끝, undistort 준비까지 끝난 뒤에 딱 3줄이다 (`colmap.py:437~440`):

```python
# size of the scene measured by cameras
camera_locations = camtoworlds[:, :3, 3]                          # [N,3] 카메라 위치
scene_center = np.mean(camera_locations, axis=0)                  # 카메라 중심(평균)
dists = np.linalg.norm(camera_locations - scene_center, axis=1)   # 중심으로부터의 거리
self.scene_scale = np.max(dists)                                  # 그 중 최댓값
```

읽는 법: **카메라 위치들의 무게중심에서 가장 멀리 떨어진 카메라까지의 거리** = 촬영 궤적을 감싸는 구의 반지름.

여기서 중요한 세 가지 뉘앙스:

| 관찰 | 의미 |
|---|---|
| **포인트클라우드가 아니라 카메라**로 잰다 | SfM 포인트는 하늘·먼 배경 등에 튀는 outlier가 많아 크기 추정이 불안정하다. 카메라 위치는 실제로 사람이 걸어다닌 범위라 훨씬 견고하다. |
| `mean`(중심)인데 `max`(반경) | 중심은 평균이라 부드럽게, 반경은 최댓값이라 "가장 바깥까지 커버"하도록 넉넉하게. 그래서 값이 다소 보수적(큰 쪽)으로 나온다. |
| **"씬의 크기"가 아니라 "관측 궤적의 크기"** | 카드의 표현대로 어디까지나 *대략적* 크기다. 카메라가 한 점에서 회전만 했다면(파노라마) `scene_scale ≈ 0`이 되어 lr이 0으로 죽는다 — 이런 데이터셋에서 `--global_scale`을 손으로 만져야 하는 이유. |

### `normalize=True`와의 관계 — 값의 범위가 정해지는 이유

`normalize=True`면 `scene_scale`을 재기 **전에** 월드가 이미 정규화되어 있다 (`colmap.py:276~289`):

```python
T1 = similarity_from_cameras(camtoworlds)   # 카메라 분포 기준 similarity
camtoworlds = transform_cameras(T1, camtoworlds)
points      = transform_points(T1, points)
T2 = align_principal_axes(points)           # 포인트 주축 정렬 (회전만)
...
```

그리고 `similarity_from_cameras`의 스케일 항이 (`normalize.py:73~76`):

```python
scale_fn = np.max if strict_scaling else np.median
scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))   # 기본: median
transform[:3, :] *= scale
```

즉 **카메라 거리의 중앙값이 1이 되도록** 월드를 리스케일한다. 그 위에서 `scene_scale`은 *최댓값*을 취하므로

> `scene_scale` ≈ (최대 카메라 거리) / (중앙값 카메라 거리) ≳ 1

대체로 **1.0 ~ 2.0 사이**의 값이 나온다. 그래서 mipnerf360 계열에서 "Scene scale: 1.x"가 찍히는 게 정상이다.

`normalize=False`면 정규화가 없으므로 COLMAP이 준 임의 단위(어떤 씬은 0.03, 어떤 씬은 400)가 그대로 튀어나온다. **이 경우에도 학습이 되는 이유가 바로 `scene_scale` 무차원화**다 — 절대 단위가 바뀌어도 lr과 임계값이 같은 비율로 따라 움직이니까.

> 주의: 정규화 변환은 포즈와 포인트에 **똑같이** 적용된다. 한쪽만 바꾸면 씬이 깨진다. `self.transform = T3 @ T2 @ T1`로 보관해 두었다가 나중에 원 COLMAP 좌표계로 되돌릴 수 있다.

---

## 2. 트레이너가 한 번 더 손보는 지점 — `* 1.1 * global_scale`

`Runner.__init__` (`simple_trainer.py:458~459`):

```python
self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
print("Scene scale:", self.scene_scale)
```

워크스루도 이걸 그대로 재현한다 (`training_walkthrough.py` L106):

```python
scene_scale = parser.scene_scale * 1.1   # simple_trainer.py:458과 동일
```

- **`1.1`** — 원 3DGS 구현에서 온 10% 여유. 학습 도중 Gaussian이 카메라 궤적 바깥으로 조금 나가는 것을 정상으로 봐 주려는 마진이다(prune 임계값이 조금 관대해진다).
- **`global_scale`** (`simple_trainer.py:101~102`, 기본 `1.0`) — 주석 그대로 *"A global scaler that applies to the scene size related parameters"*. 자동 추정이 실패한 씬(파노라마, 드론 항공, 카메라가 한 방향으로만 직진하는 forward-facing 등)에서 **씬 스케일 의존 하이퍼파라미터 전체를 한 손잡이로** 조절하는 탈출구다.

이후 코드에서는 `parser.scene_scale`이 아니라 이 `self.scene_scale`이 쓰인다. 카드에서 말하는 "기준 단위"의 실체가 이 값이다.

---

## 3. 무엇에 쓰이나 — 소비처 전수

### (a) `means`의 learning rate

```python
("means", torch.nn.Parameter(points), means_lr * scene_scale),   # simple_trainer.py:336
```
워크스루 (L185):
```python
"means": 1.6e-4 * scene_scale,   # 위치는 씬 크기에 비례
```

**왜 위치만?** Adam의 step 크기는 `lr` 단위로 결정되고, `means`는 **길이 차원(m)** 을 갖는 유일한 파라미터다. 씬을 2배로 확대하면 "적절한 한 스텝의 이동량"도 2배가 되어야 한다. 반면

| 파라미터 | 차원 | lr에 `scene_scale`? |
|---|---|---|
| `means` | 길이 | ✅ `1.6e-4 * scene_scale` |
| `scales` | log(길이) | ❌ `5e-3` — 로그공간이므로 스케일 변환이 **상수 오프셋**이 되어 gradient가 불변 |
| `quats` | 무차원(회전) | ❌ `1e-3` |
| `opacities` | 무차원(logit) | ❌ `5e-2` |
| `sh0` / `shN` | 무차원(색) | ❌ `2.5e-3` / `2.5e-3/20` |

`scales`가 로그공간에 저장되는 설계가 여기서 배당을 준다 — $s \to \alpha s$는 $\log s \to \log s + \log\alpha$이므로 gradient가 스케일에 무관해지고, lr을 건드릴 필요가 없다.

### (b) 밀도화(densification) 임계값 — `DefaultStrategy`

전략은 상태 안에 이 값을 넣어 들고 다닌다 (`default.py:116~129`, 트레이너 `simple_trainer.py:510~513`):

```python
state = {"grad2d": None, "count": None, "scene_scale": scene_scale}
```

그리고 3D 크기 판정에 곱해진다:

```python
# default.py:299~302  — duplicate vs split 갈림길
is_small = (
    torch.exp(params["scales"]).max(dim=-1).values
    <= self.grow_scale3d * state["scene_scale"]        # grow_scale3d = 0.01
)

# default.py:353~356  — 비대해진 Gaussian prune
is_too_big = (
    torch.exp(params["scales"]).max(dim=-1).values
    > self.prune_scale3d * state["scene_scale"]        # prune_scale3d = 0.1
)
```

docstring이 표현을 못 박아 둔다 (`default.py:57~62`): *"GSs with 3d scale **(normalized by scene_scale)** below this value…"*. 즉 `grow_scale3d`/`prune_scale3d`는 **길이가 아니라 비율**이다.

워크스루 5단계 표(L300~302)를 이 눈으로 다시 읽으면 의미가 딱 맞는다:

| 동작 | 조건 | `scene_scale`이 하는 일 |
|---|---|---|
| **duplicate** | grad > `2e-4` **and** 크기 ≤ **1%·scene_scale** | "씬 반경의 1%보다 작다" = 작은 Gaussian → 복제 |
| **split** | grad > `2e-4` **and** 크기 > **1%·scene_scale** | "씬 반경의 1%보다 크다" = 큰 Gaussian → 2개로 쪼개고 크기 /1.6 |
| **prune** | opacity < `0.005` **or** 크기 > **10%·scene_scale** | 씬 반경의 10%를 넘는 비대한 것 제거 |

`grow_grad2d = 2e-4`나 `prune_opa = 0.005`에는 `scene_scale`이 안 붙는다 — 화면공간 gradient는 NDC `[-1,1]`에서 화면 해상도로 환산될 뿐 월드 길이와 무관하고(`default.py:248~249`: `grads[...,0] *= width/2 * n_cameras`), opacity는 애초에 무차원이기 때문이다. **길이 차원을 가진 임계값만 `scene_scale`이 붙는다**는 규칙이 여기서도 일관된다.

> 대조: `MCMCStrategy`는 `initialize_state()`를 **인자 없이** 부른다 (`simple_trainer.py:515`). MCMC는 3D 크기 임계값 대신 `cap_max`(총 개수 상한)와 `noise_lr`로 밀도를 제어하므로 `scene_scale`이 필요 없다. "밀도화 임계값의 기준 단위"라는 서술은 `DefaultStrategy`에 한정된 이야기다.

### (c) 랜덤 초기화 범위

```python
elif init_type == "random":
    points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)  # :315
```

SfM 포인트를 안 쓸 때도 초기 점을 **씬 크기에 맞는 정육면체** 안에 뿌려야 한다. 그래서 `init_type="random"`이어도 `Parser`(=`scene_scale`)는 여전히 필요하다.

### (d) depth loss 가중

```python
depthloss = depth_l1_loss(depths, depths_gt, scene_scale=self.scene_scale)  # :977~979
```
```python
# gsplat/losses.py:222~224
disp    = torch.where(pred_depth > 0.0, 1.0 / pred_depth, 0)
disp_gt = torch.where(gt_depth   > 0.0, 1.0 / gt_depth,   0)
return F.l1_loss(disp, disp_gt) * scene_scale
```

disparity(역깊이)는 **길이의 역수** 차원이다. 씬을 2배 키우면 disparity와 그 L1이 절반이 되므로, `scene_scale`을 다시 곱해 원상복구한다 → `depth_lambda`가 씬 크기에 무관해진다.

### (e) 렌더 궤적 반경

```python
bounds=self.parser.bounds * self.scene_scale,   # :1327 (spiral 궤적)
```
평가용 spiral 카메라 경로의 크기도 씬 단위에 맞춘다.

---

## 4. 차원 분석으로 보는 "왜 이게 작동하는가"

씬 전체를 $\alpha$배 확대하는 변환($\mathbf{x} \to \alpha\mathbf{x}$)을 생각하자. 그러면 `scene_scale`도 정확히 $\alpha$배가 된다 (정의가 카메라 위치의 거리이므로 1차 동차). 각 항이 어떻게 변하는지:

| 양 | 스케일 변화 | `scene_scale` 보정 후 |
|---|---|---|
| `means` gradient | $\propto \alpha^{-1}$ | lr $\propto \alpha$ → step $\propto \alpha$ ✅ 상대 이동량 불변 |
| `exp(scales)` (3D 크기) | $\propto \alpha$ | 임계값 $\propto \alpha$ ✅ 판정 결과 불변 |
| disparity L1 | $\propto \alpha^{-1}$ | $\times\alpha$ ✅ 손실 값 불변 |
| 화면공간 grad, opacity, SH | 불변 | 보정 없음 ✅ |

**모든 열이 "불변"으로 끝난다** = 하이퍼파라미터를 하나도 안 바꾸고 씬 크기만 달라져도 학습 궤적이 (수치오차 범위에서) 동일하다. 이게 카드가 말하는 *"씬 크기에 무관하게 동작하도록 만들어 주는 스케일 인자"*의 정확한 뜻이다.

반대로 말하면 — `scene_scale`이 **틀리면** 이 불변성이 깨지고 다음이 동시에 어긋난다:

- 너무 **작게** 추정 → `means` lr 과소 → 포인트가 제자리에서 안 움직여 흐릿하게 수렴 + prune 임계값이 과하게 엄격해 Gaussian이 계속 지워짐.
- 너무 **크게** 추정 → `means` lr 과대 → 발산·floater 폭증 + `grow_scale3d` 판정이 전부 "small"로 몰려 split이 안 일어나고 큰 Gaussian이 남아 뭉개짐.

증상이 "lr 문제"처럼 보이는데 원인은 데이터 스케일이라는 점이 디버깅을 어렵게 만든다. **학습 시작 시 찍히는 `Scene scale: ...` 로그를 먼저 확인하는 습관**이 그래서 유용하다 (`simple_trainer.py:459`).

---

## 5. 흔한 함정

- **`parser.scene_scale` ≠ 학습에 쓰이는 `scene_scale`.** 후자는 `* 1.1 * global_scale`이 붙은 `Runner.scene_scale`이다. 워크스루도 `parser.scene_scale * 1.1`로 맞춘다.
- **포인트클라우드 bounding box가 아니다.** 카메라 위치 분포의 반경이다. 하늘·먼 배경 포인트가 있어도 값이 안 튄다.
- **카메라가 거의 한 곳에 모인 데이터(파노라마·순수 회전)에서는 `scene_scale ≈ 0`.** lr이 0에 가까워져 학습이 멈춘다 → `--global_scale`로 수동 보정.
- **`normalize=False`로 바꾸면 `scene_scale`의 절대값이 완전히 달라진다.** 그래도 무차원화 덕에 학습은 되지만, 로그에 찍힌 숫자를 다른 실행과 직접 비교하면 안 된다.
- **`scales`의 lr에는 `scene_scale`을 곱하지 않는다.** 로그공간 저장이기 때문. 여기 곱하면 오히려 씬 크기 의존성이 생긴다.
- **`MCMCStrategy`에는 `scene_scale`을 넘기지 않는다.** 넘길 자리도 없다(`initialize_state()` 무인자). 전략을 바꿀 때 "왜 scene_scale이 안 쓰이지?"로 헤매지 말 것.
- **`grow_scale3d=0.01` / `prune_scale3d=0.1`은 길이가 아니라 비율(1%, 10%)이다.** 절대 길이로 오해하면 값 조정 방향을 정반대로 잡는다.

---

## 6. 기억용 한 줄 정리

> **`scene_scale` = "카메라들이 퍼져 있는 반경"** — SfM 좌표계에 단위를 부여하는 자(ruler).
> 길이 차원을 가진 하이퍼파라미터(`means` lr, 3D 크기 임계값, disparity 손실)를 이 자로 재서 무차원화하기 때문에, `1.6e-4`·`1%`·`10%` 같은 기본값이 어떤 씬에서도 그대로 통한다.
