# 3DGS의 기본 손실 함수는?

> $\mathcal{L} = (1-\lambda)\,\mathcal{L}_{L1} + \lambda\,(1-\mathrm{SSIM})$이고 $\lambda = 0.2$다.
> 즉 **0.8·L1 + 0.2·SSIM loss**다.

## 한 줄 요약

3DGS의 손실은 **렌더 이미지와 GT 이미지를 비교하는 두 항의 가중합뿐**이다.
기하(Gaussian 위치·모양·개수)에 대한 정규화 항이 기본으로 하나도 없다 —
"어디에 Gaussian을 둘지"는 손실이 아니라 **밀도화 전략(densification)** 이 담당한다.
이 분업이 3DGS를 이해하는 핵심이다.

원논문: Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*
(SIGGRAPH 2023) 식 (7). 논문 표기는 $\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}}$, $\lambda = 0.2$.

## 코드에서 이 한 줄

`examples/simple_trainer.py:955~961` — 학습 루프에서 손실을 만드는 전부다.

```python
l1loss   = l1_loss(colors, pixels).mean()
ssimloss = ssim_loss(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2))
loss     = torch.lerp(l1loss, ssimloss, cfg.ssim_lambda)   # cfg.ssim_lambda = 0.2
```

`ssim_lambda`의 기본값 선언은 `examples/simple_trainer.py:171` (`ssim_lambda: float = 0.2`).

### `torch.lerp`이 왜 그 식인가

`lerp`(선형 보간)의 정의는 $\mathrm{lerp}(a, b, w) = a + w\,(b-a)$이므로

$$\mathrm{lerp}(\mathcal{L}_{L1},\, \mathcal{L}_{SSIM},\, 0.2)
= \mathcal{L}_{L1} + 0.2\,(\mathcal{L}_{SSIM} - \mathcal{L}_{L1})
= 0.8\,\mathcal{L}_{L1} + 0.2\,\mathcal{L}_{SSIM}$$

가중치 합이 항상 1이 되도록 **λ 하나만 노출**한 것이다.
`0.8*l1 + 0.2*ssim`을 직접 쓰는 것과 수학적으로 동일하고, 커널 하나로 끝나 조금 빠르다.
워크스루도 같은 주석을 달아 둔다(`training_walkthrough.py:284`, `:377`).

## 두 항은 각각 무엇을 담당하는가

| 항 | 보는 대상 | 없으면 생기는 문제 |
|---|---|---|
| **L1** (0.8) | 픽셀 하나하나의 **절대 색 오차** | 색이 전역적으로 안 맞고 톤이 어긋난다. 국소 구조만 맞추다 색이 표류 |
| **1−SSIM** (0.2) | 11×11 이웃의 **평균·분산·공분산 = 국소 구조/대비** | L1만 쓰면 평균만 맞추는 방향으로 수렴해 **뭉개짐(blur)** — 고주파 텍스처가 사라진다 |

L1을 주력(0.8)으로 두는 이유는 픽셀 단위 재구성이 학습 신호로 안정적이고
아웃라이어(노출 차이, 지나가는 사람 등)에 L2보다 강건하기 때문이다.
그 위에 SSIM을 0.2만 얹어 **"L1이 만들려는 평균-이미지"를 구조 쪽으로 밀어준다.**

### 왜 하필 0.2인가

- λ가 더 크면 SSIM이 지배해 국소 대비는 살지만 색/밝기가 어긋나기 쉽다.
- λ가 더 작으면(0에 가까우면) 전형적인 L1 블러가 남는다.
- 0.2는 원논문이 정한 값이고 gsplat도 그대로 따른다. 사실상 **모든 3DGS 후속 연구의 관행값**이라
  논문 비교 실험에서 건드리지 않는 상수로 취급된다.
- 스케일 감각: 수렴한 씬에서 L1은 대략 0.02~0.05, `1−SSIM`은 0.03~0.10 수준이라
  두 항의 크기가 같은 자릿수다. λ=0.2는 "SSIM을 보조 항으로 둔다"는 의도대로 동작한다.

## SSIM 항의 내부 — `1 - SSIM`이라는 부호 방향

`gsplat/losses.py:154` `ssim_loss()`는 이름 그대로 **손실**이라 이미 $1-\mathrm{SSIM}$을 돌려준다.
카드 답의 "$1-\mathrm{SSIM}$"이 함수 안에 들어 있다는 뜻 — 학습 코드에서 다시 `1 -`를 붙이지 않는다.

SSIM 자체(`torch_ssim_loss`, `gsplat/losses.py:110`)는 Wang et al. 2004의 정의다.

$$\mathrm{SSIM}(x,y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}
{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

- $\mu, \sigma$는 **11×11 가우시안 윈도우**(σ=1.5)로 가중 평균한 국소 통계.
  `create_ssim_window()`가 1D 커널의 외적으로 2D 윈도우를 만들고
  `F.conv2d(..., groups=channel)`로 채널별(depthwise) 컨볼루션한다.
- $C_1 = 0.01^2,\; C_2 = 0.03^2$ — 분모가 0에 가까울 때(평탄한 영역) 폭주하지 않게 하는 안정화 상수.
  입력이 `[0, 1]` 범위라는 가정이 여기 들어 있다.
- SSIM ∈ [−1, 1], 완전 일치면 1 → 손실은 0. 그래서 `1 - ssim_map.mean()`.

### 구현 디테일 4가지

1. **`permute(0, 3, 1, 2)`** — 렌더러 출력은 `[B, H, W, 3]`(NHWC)인데 SSIM은 conv2d를 쓰므로
   `[B, 3, H, W]`(NCHW)가 필요하다. L1은 shape에 무관해서 permute가 없다.
2. **`fused_ssim` 빠른 경로** (`gsplat/losses.py:182`) — 설치되어 있고
   `window_size == 11`, GT가 `requires_grad=False`, CUDA 텐서일 때만 CUDA 커널로 계산한다.
   조건이 안 맞으면 순수 PyTorch 참조 구현으로 폴백한다.
   두 경로는 패딩이 달라(`padding="valid"` vs 참조 구현의 `padding=5`) **값이 미세하게 다르다** —
   fused는 테두리를 버리고, 참조 구현은 제로패딩된 테두리까지 평균에 넣는다.
   같은 씬의 loss 곡선을 환경 간에 비교할 때 이 차이가 보인다.
3. **윈도우 캐시** — `_ssim_window_cache[(window_size, channel, device, dtype)]`.
   매 스텝 윈도우를 다시 만들지 않는다.
4. **`[0, 1]` 범위 계약** — `GSPLAT_ENFORCE_CONTRACTS=1`일 때만 assert로 검사한다
   (`gsplat/losses.py:42`). 평소엔 GPU 동기화 비용 때문에 꺼져 있으므로,
   렌더가 1을 넘는 상태에서 SSIM 값이 이상해도 조용히 넘어간다.

## L1 항의 내부

`l1_loss()`(`gsplat/losses.py:53`)는 `F.l1_loss(..., reduction="none")` — **요소별 텐서**를 돌려준다.
그래서 학습 코드가 `.mean()`을 붙여 스칼라로 만든다. 이 설계 덕에 마스킹이 자연스럽다.

```python
if masks is not None:
    l1loss = l1_loss(colors[masks], pixels[masks]).mean()   # 픽셀을 아예 제외
    colors_ssim = colors * masks[..., None]                 # SSIM은 양쪽을 0으로
    pixels_ssim = pixels * masks[..., None]
```

두 항의 마스킹 방식이 다른 이유(`simple_trainer.py:946~948` 주석):
L1은 픽셀 독립이라 골라내면 끝이지만, **SSIM은 패치 기반**이라 픽셀을 빼면 윈도우가 깨진다.
그래서 양쪽 이미지를 똑같이 0으로 만들어 마스크된 패치가 임의의 색으로 끌려가지 않게 한다.

## 기본 손실 "이외"의 항 — 전부 기본 off

카드가 "**기본** 손실 함수"라고 묻는 이유가 여기 있다. `simple_trainer.py`에는 옵션 항이 더 있지만
기본 설정에서는 하나도 켜지지 않는다.

| 항 | 플래그 (기본값) | 하는 일 |
|---|---|---|
| depth loss | `depth_loss=False`, `depth_lambda=1e-2` | SfM 포인트 위치에서 렌더 깊이를 `grid_sample`로 뽑아 **disparity 공간 L1** (`depth_l1_loss`) |
| opacity 정규화 | `opacity_reg=0.0` | `sigmoid(opacities).mean()` — 불투명도를 낮추도록 압박 |
| scale 정규화 | `scale_reg=0.0` | `exp(log_scales).mean()` — 거대 Gaussian 억제 |
| 후처리 정규화 | `post_processing=None` | `bilateral_grid`면 TV loss ×10, `ppisp`면 모듈 자체 정규화 |
| random background | `random_bkgd=False` | 손실 항이 아니라 **입력 조작**: 알파에 랜덤 배경을 합성해 투명 영역으로 도망가는 것 방지 |

예외는 **MCMC 프리셋**이다(`simple_trainer.py:1584~1590`): `python simple_trainer.py mcmc`는
`opacity_reg=0.01`, `scale_reg=0.01`을 켠다. MCMC 전략에서는 이 두 정규화가 샘플링 관점의
필수 구성요소라서, "3DGS 기본 손실 = 광도 항 2개"라는 말은 `default` 전략 기준임을 기억하자.

## 이 손실이 학습 루프에서 놓이는 자리

워크스루가 정리한 한 스텝(`training_walkthrough.py:320~330`, `:370~380`):

```
rasterization()  →  renders, alphas, info
strategy.step_pre_backward(info)      # means2d.retain_grad() — 화면공간 grad 확보
l1  = l1_loss(renders, pixels).mean()
ssim = ssim_loss(renders.permute(0,3,1,2), pixels.permute(0,3,1,2))
loss = torch.lerp(l1, ssim, 0.2)
loss.backward()                       # 픽셀 오차 → 알파블렌딩 → 투영 → Gaussian 파라미터
optimizer.step(); strategy.step_post_backward()   # 여기서 densify/prune
```

즉 **손실은 오직 "이 픽셀 색이 틀렸다"만 말한다.** 그 gradient가 래스터라이저를 거꾸로 타고
`means`/`scales`/`quats`/`opacities`/`sh`로 흘러간다.
"Gaussian이 부족하다/과하다"는 판단은 손실이 아니라 `step_post_backward()`가
화면공간 gradient 통계를 보고 내린다 — 손실 함수가 이렇게 단순할 수 있는 이유다.

## 평가 지표와 혼동하지 말 것

학습 손실의 SSIM은 `gsplat.losses.ssim_loss`지만,
평가(`Runner.eval`, `simple_trainer.py:1258`)의 SSIM은 **torchmetrics의
`StructuralSimilarityIndexMeasure(data_range=1.0)`** 이다.

| | 학습 | 평가 |
|---|---|---|
| 목적 | 미분 가능한 최적화 신호 | 논문 표에 실을 지표 |
| SSIM 구현 | `fused_ssim` 또는 참조 conv2d | torchmetrics |
| 방향 | 손실이므로 `1 - SSIM`, **낮을수록 좋음** | SSIM 그대로, **높을수록 좋음** |
| 함께 보는 값 | loss, L1, SSIM loss | PSNR / SSIM / LPIPS (+ 색보정판 `cc_*`) |

## 흔한 오해

| 오해 | 사실 |
|---|---|
| L2(MSE)를 쓴다 | **L1**이다. NeRF 계열의 MSE와 다른 선택 |
| SSIM 가중치가 0.8 | 거꾸로다. λ=0.2가 **SSIM** 쪽 가중치, L1이 0.8 |
| 코드에서 `1 - ssim`을 직접 계산한다 | `ssim_loss()`가 이미 `1 - SSIM`을 반환한다 |
| 3D 위치/모양에 대한 정규화가 들어 있다 | 기본 손실에는 없다. `opacity_reg`/`scale_reg`는 기본 0(MCMC 프리셋만 0.01) |
| Gaussian 개수를 손실이 조절한다 | 개수는 densification 전략(`DefaultStrategy`/`MCMCStrategy`)의 몫 |
| SSIM은 전체 이미지 하나의 값 | 11×11 윈도우로 계산한 **국소 SSIM 맵의 평균**이다 |
| 마스크가 있으면 두 항 모두 픽셀을 제외한다 | L1만 제외. SSIM은 패치 기반이라 양쪽을 0으로 곱한다 |

## 암기 포인트

- 식: $\mathcal{L} = 0.8\,\mathcal{L}_{L1} + 0.2\,(1-\mathrm{SSIM})$
- λ=0.2가 붙는 쪽은 **SSIM**
- 구현: `torch.lerp(l1loss, ssimloss, 0.2)` — `simple_trainer.py:961`
- SSIM 윈도우: **11×11 가우시안(σ=1.5)**, $C_1=0.01^2$, $C_2=0.03^2$
- 기하 정규화 없음 → 그 역할은 densification
