# `scales`는 log 공간에 살고, 렌더 직전에만 `exp`로 깨어난다

**질문** `scales` 파라미터는 어떤 공간에 저장되고 어떤 활성화를 거치는가?

**답** log 공간에 저장되고 렌더 시 `exp`를 통과한다. 초기값은 3-최근접 이웃 평균거리의 log이며, 이웃이 멀면 큰 Gaussian이 된다.

---

## 1. 큰 그림: "제약 있는 값은 비제약 공간에 저장한다"

3D Gaussian 하나는 5종류의 파라미터로 표현되는데, 이 중 **값의 범위에 제약이 있는 것들은
제약 없는 공간(unconstrained space)에 저장하고 렌더 직전에 활성화 함수를 통과시킨다.**
워크스루 2단계의 표가 그 규칙을 한눈에 정리한다
([`training_walkthrough.py` 2단계](../assets/training_walkthrough.py)):

| 파라미터 | shape | 저장 공간 → 활성화 | 초기값 |
|---|---|---|---|
| `means` | [N,3] | 그대로 | SfM 포인트 위치 |
| **`scales`** | **[N,3]** | **log → `exp`** | **log(3-최근접 이웃 평균거리)** |
| `quats` | [N,4] | 미정규화 → 내부 normalize | 랜덤 |
| `opacities` | [N] | logit → `sigmoid` | logit(0.1) |
| `sh0`/`shN` | [N,1,3]/[N,15,3] | SH 계수 (제약 없음) | DC=(rgb−0.5)/0.2821, 고차항=0 |

`scales`의 제약은 **양수(> 0)** 다. 크기가 0이나 음수가 되면 공분산
$\Sigma = R\,S\,S^\top R^\top$ 가 특이(singular)해지거나 의미를 잃는다.
`exp`를 쓰면 Adam이 log 값을 어디로 밀어붙이든 **실제 크기는 항상 (0, ∞)** 이라
projection·clamping 같은 후처리가 전혀 필요 없다. 그래서 `gsplat` 코드베이스 어디에도
`scales.clamp_min(0)` 류의 방어 코드가 없다 — 파라미터화 자체가 제약을 흡수한다.

## 2. 왜 굳이 log인가 — 세 가지 이유

### (a) 양의 정부호(positive-definite) 공분산 보장

$S = \mathrm{diag}(e^{s_x}, e^{s_y}, e^{s_z})$ 이므로 $S$ 의 대각원소는 항상 양수고,
$\Sigma = R\,S\,S^\top R^\top = (RS)(RS)^\top$ 는 정의상 양의 준정부호다.
`quats`가 회전을, `scales`가 축 길이를 담당하는 이 분해 덕분에
"공분산 행렬 6개 성분을 직접 최적화하다가 PD 조건이 깨지는" 문제를 원천 차단한다.

### (b) 곱셈적(상대적) 업데이트 = 스케일 불변성

`exp`의 미분은 자기 자신이므로 chain rule에서

$$\frac{\partial L}{\partial s} = \frac{\partial L}{\partial \sigma}\cdot \sigma,\qquad \sigma = e^{s}$$

즉 **log 공간의 덧셈 = 실제 공간의 곱셈**이다. Adam은 업데이트 크기를
대략 `lr` 규모로 정규화하므로, log 공간에서 한 스텝은
$\sigma \leftarrow \sigma \cdot e^{\pm \text{lr}}$ — `scales_lr = 5e-3`이면
**한 스텝마다 크기의 약 ±0.5% 상대 변화**다.
이 상대 변화율은 1 m짜리 거대한 Gaussian이든 1 mm짜리 미세한 Gaussian이든 동일하다.
선형 공간에서 최적화했다면 작은 Gaussian은 한 스텝에 음수로 튀고 큰 Gaussian은
꿈쩍도 하지 않는 — lr 하나로 수 자릿수 범위를 커버할 수 없는 — 상황이 된다.

이것이 `means`와의 결정적인 차이다. `means`는 선형 공간이라 lr에
`scene_scale`을 곱해줘야 씬 크기에 무관해지고
(`means_lr = 1.6e-4 * scene_scale`, `simple_trainer.py:335`),
게다가 `ExponentialLR(gamma = 0.01**(1/max_steps))` 스케줄러까지 붙는다
(`simple_trainer.py:808-812`). 반면 **`scales`는 `scene_scale` 곱셈도, 스케줄러도 없는
상수 `5e-3`** — log 파라미터화가 그 두 보정을 이미 공짜로 해주기 때문이다.

### (c) 자릿수(orders of magnitude) 표현

한 씬 안에서 Gaussian 크기는 벽면을 덮는 큰 것부터 나뭇잎 디테일까지 수백~수천 배 차이가
난다. log 공간에서는 이 범위가 좁은 구간(예: −6 ~ 0)에 균등하게 펴지므로
gradient의 동적 범위도 함께 압축된다.

## 3. 초기값: 3-최근접 이웃 평균거리의 log

### 실제 코드

`simple_trainer.py:321-323` (`create_splats_with_optimizers`):

```python
# Initialize the GS size to be the average dist of the 3 nearest neighbors
dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)   # [N,]  자기 자신 제외한 3개
dist_avg  = torch.sqrt(dist2_avg)
scales    = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
```

워크스루의 최소 재구현도 같은 논리다 (`knn_mean_dist`, `init_splats_with_optimizers`):

```python
dist_avg = knn_mean_dist(points, k=3)                    # cdist → topk(k+1) → 자기 제외 → mean
scales   = torch.log(dist_avg)[:, None].repeat(1, 3)     # [N,3] log-space
```

읽는 순서를 정확히 짚어보면:

1. `knn(points, 4)` — **4-NN**을 뽑는다. 첫 번째 이웃은 자기 자신(거리 0)이므로
   `[:, 1:]`로 잘라내면 **진짜 3-최근접 이웃**이 남는다. "3-최근접"인데 4를 넘기는 이유가 이것.
2. 그 3개 거리를 평균해 `dist_avg`를 얻는다.
   (미묘한 차이: `simple_trainer`는 *제곱거리의 평균 후 sqrt* = RMS 거리,
   워크스루는 *거리의 산술평균*. 값이 거의 같아 실전에서는 무의미한 차이지만
   코드를 대조해 읽을 때 눈에 걸릴 수 있다.)
3. `* init_scale` — CLI 플래그 `--init-scale` (기본 `1.0`)로 초기 크기를 일괄 배율한다.
   log 안에서 곱해지므로 log 공간에서는 `+ log(init_scale)` 상수 이동이다.
4. `log(...)` 후 `.repeat(1, 3)` — 세 축에 **같은 값**을 넣으므로 모든 Gaussian은
   **등방(isotropic) 구**로 출발한다. 학습 중 축별 gradient가 갈리면서 비등방
   타원체(표면을 얇게 덮는 pancake 등)로 발달한다.

### "이웃이 멀면 큰 Gaussian"의 의미

`dist_avg`는 **국소 점 밀도의 역수**에 해당하는 양이다.

- SfM 포인트가 촘촘한 영역(텍스처 풍부, 여러 뷰에서 잘 매칭된 곳) → 이웃이 가깝다 →
  `dist_avg` 작다 → 작은 Gaussian → 세밀한 디테일 담당.
- 포인트가 희박한 영역(하늘, 무텍스처 벽, 배경) → 이웃이 멀다 →
  `dist_avg` 크다 → 큰 Gaussian → 넓은 면적을 한 개로 커버.

목표는 워크스루 주석대로 **"빈틈없이 덮되 과하게 겹치지 않도록"**이다.
너무 작게 시작하면 초기 렌더가 점묘화처럼 구멍이 뚫려 gradient가 흐르지 않고,
너무 크게 시작하면 모든 픽셀이 뿌옇게 뭉개져 densification 신호가 뭉뚱그려진다.
이웃거리 기반 초기화는 어떤 씬에서도 자동으로 그 중간을 잡아준다 — 하이퍼파라미터가
아니라 **데이터에서 읽어낸 값**이라는 점이 핵심이다.

### 수치 감각

| `dist_avg` (m) | 저장되는 log 값 | `exp` 후 실제 크기 | 성격 |
|---|---|---|---|
| 0.002 | −6.21 | 0.002 | 초고밀도 디테일 |
| 0.01 | −4.61 | 0.01 | 촘촘한 표면 |
| 0.05 | −3.00 | 0.05 | 보통 |
| 0.5 | −0.69 | 0.5 | 희박한 배경 |
| 2.0 | +0.69 | 2.0 | 하늘/원경 |

log 값의 실제 범위가 대략 −6 ~ +1 정도의 좁은 구간에 들어오는 것을 볼 수 있다.
이게 (c)에서 말한 자릿수 압축이다.

## 4. `exp`가 실제로 적용되는 지점

렌더 함수 진입 직전, 단 한 곳이다.
`simple_trainer.py:669` / 워크스루 `rasterize_splats()`:

```python
means     = splats["means"]                       # [N,3]  변환 없음
quats     = splats["quats"]                       # [N,4]  rasterization 내부에서 normalize
scales    = torch.exp(splats["scales"])           # [N,3]  log → 실제 크기  ★
opacities = torch.sigmoid(splats["opacities"])    # [N]    logit → (0,1)
colors    = torch.cat([splats["sh0"], splats["shN"]], 1)   # [N,16,3]

return rasterization(means=means, quats=quats, scales=scales,
                     opacities=opacities, colors=colors, ...)
```

즉 **`rasterization()`은 "실제 크기"를 받는 API**다. log를 아는 것은 학습 루프 쪽이고,
CUDA 커널은 선형 크기만 본다. 커널 안에서는 이 `scales`가
`quat_scale_to_covar_preci`를 거쳐 $\Sigma = RSS^\top R^\top$ 로 조립되고,
그 다음 `fully_fused_projection`이 EWA splatting으로 2D conic과 `radii`를 만든다
(워크스루 3단계의 4단계 커널 파이프라인 중 2번).

`exp`는 미분 가능하므로 backward에서 gradient가 그대로 log 공간으로 되돌아온다
(§2(b)의 $\partial L/\partial s = \sigma \cdot \partial L/\partial \sigma$).
`torch.autograd`가 이 곱셈을 알아서 처리하니 별도 코드는 없다.

## 5. log 공간 규약이 파이프라인 전체에서 지켜지는 방식

`splats["scales"]`가 log라는 사실은 학습 루프 밖의 모든 소비자가 알아야 하는 계약이다.
그래서 코드에는 **exp/log 왕복이 정확히 짝을 이루는 지점들**이 흩어져 있다.

| 위치 | 코드 | 하는 일 |
|---|---|---|
| 렌더 | `simple_trainer.py:669` | `torch.exp(splats["scales"])` |
| **duplicate** 판정 | `strategy/default.py:298-301` | `torch.exp(params["scales"]).max(-1).values <= grow_scale3d * scene_scale` → 크기 ≤ 1%·scene_scale이고 grad 크면 복제 |
| **prune** 판정 | `strategy/default.py:352-356` | `torch.exp(params["scales"]).max(-1).values > prune_scale3d * scene_scale` → 10%·scene_scale보다 크면 제거 |
| **split** 실행 | `strategy/ops.py:196, 211` | `scales = exp(p[sel])`로 샘플링용 실제 크기를 얻고, 새 값은 `torch.log(scales / 1.6)` — 선형 공간에서 1/1.6배 = **log 공간에서 −log(1.6) ≈ −0.47 이동** |
| duplicate 실행 | `strategy/ops.py` (`else` 분기) | 복제는 크기를 그대로 복사 (log 값 그대로 repeat) |
| **MCMC relocate** | `strategy/ops.py:331, 341` | `compute_relocation(scales=torch.exp(...))`로 실제 크기를 넘기고, 반환값은 `torch.log(new_scales)`로 되돌려 저장 |
| **MCMC noise** | `strategy/ops.py:496` | `scales = torch.exp(params["scales"])` → 공분산을 만들어 noise 방향을 정한다 (CUDA fused 경로는 log 텐서를 직접 받아 내부에서 exp) |
| **`scale_reg_loss`** | `gsplat/losses.py:689-700` | `return torch.exp(log_scales).mean()` — **인자 이름 자체가 `log_scales`**. 실제 크기의 평균을 벌점화해 비대한 Gaussian을 억제 (`simple_trainer.py:996`, MCMC 전략용) |
| **PLY export/import** | `simple_trainer.py:1093`, `gsplat/exporter.py` | export 시 `self.splats["scales"]`를 **exp 없이 그대로** `export_splats()`에 넘긴다. 짝이 되는 `load_ply()`의 반환 규격도 `scales: (N, 3) log-scales`(`exporter.py:454`) — INRIA 원본 3DGS의 `.ply` 규약이 log-scale 저장이기 때문 |

읽는 요령: **`exp`가 보이면 "지금부터 물리적 크기 이야기", `log`가 보이면
"파라미터로 되돌려 저장하는 중"** 이다. `default.py`의 densification 판정이
`scene_scale`과 크기를 비교하는 대목은 반드시 `exp` 뒤에서 일어나야 의미가 있다 —
log 값을 `scene_scale`과 비교하는 것은 단위가 다른 수의 비교라 무의미하다.

## 6. 자주 걸려 넘어지는 지점

- **`splats["scales"]`를 그대로 크기로 읽는 실수.** 값이 음수라 "크기가 −4.6"처럼
  보인다면 log를 안 벗긴 것이다. 반대로 `exp`를 두 번 적용하면 (예: 이미 exp된 텐서를
  다시 `rasterize_splats`에 통과) 크기가 폭발해 화면이 전부 뿌옇게 된다.
- **split의 `/1.6`은 선형 공간 연산.** log 공간에서 `p/1.6`을 하는 것은 완전히 다른
  연산이다 (`ops.py:211`이 굳이 `exp` → 나눗셈 → `log`로 왕복하는 이유).
- **PLY를 읽어와 다시 학습에 넣을 때.** export가 log를 그대로 썼으므로 import도
  log로 받아야 한다. exp를 한 번 끼우면 크기가 e배 이상 어긋난다.
- **`scales_lr`을 크게 올리는 유혹.** log 공간에서 lr은 상대 변화율이므로
  `5e-3` → `5e-2`는 "한 스텝에 5% 크기 변화"가 되어 densification 판정
  (`grow_scale3d`, `prune_scale3d` 임계값)이 스텝마다 요동친다.
- **등방 초기화는 초기값일 뿐.** `[N,3]`으로 세 축을 따로 두는 이유는 학습 중
  비등방으로 갈라지게 하기 위함이다. 세 축을 묶어 `[N,1]`로 두면 표면을 얇게 덮는
  납작한 Gaussian을 만들 수 없다.

## 7. 한 문장 정리

> `scales`는 **양수 제약을 파라미터화로 흡수하기 위해 log 공간에 `[N,3]`으로 저장**되고,
> **렌더 직전 `torch.exp`** 로 실제 크기가 되어 $\Sigma = RSS^\top R^\top$ 조립에 들어간다.
> 초기값은 **3-최근접 이웃 평균거리의 log**(`log(dist_avg * init_scale)`, 세 축 동일)이라
> 점이 희박한 곳은 큰 Gaussian, 촘촘한 곳은 작은 Gaussian으로 자동 출발하고,
> log 파라미터화 덕분에 lr `5e-3` 하나가 씬 크기와 무관하게 **상대적 크기 변화**로 작동한다.
