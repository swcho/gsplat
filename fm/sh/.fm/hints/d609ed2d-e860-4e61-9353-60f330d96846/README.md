# 노트북의 Adam 학습 셀은 3DGS 학습의 어떤 요소들을 흉내내는가?

**답**: `sh0`와 `shN`에 다른 학습률(2.5e-2 와 그 1/20), `min(step // 500, 3)`로 차수 점진 활성화, 예측에 `+0.5`와 `clamp_min(0)` 적용, L1 loss 사용이다.

---

## 1. 문제의 셀 (sh_walkthrough.py §4.2 "방법 B: Adam으로 계수 학습")

```python
obs_d = F.normalize(torch.randn(60, 3, device=DEVICE), dim=-1)     # 카메라 60대의 시점 방향
obs_c = f_star(obs_d)                                              # 그 방향에서 "관측한" 색
sh0 = torch.zeros(1, 3, device=DEVICE, requires_grad=True)         # DC 계수     [1,3]
shN = torch.zeros(15, 3, device=DEVICE, requires_grad=True)        # 고차 계수   [15,3]
opt = torch.optim.Adam([{"params": [sh0], "lr": 2.5e-2},
                        {"params": [shN], "lr": 2.5e-2 / 20}])
A_obs = sh_bases(obs_d, 3)                                         # 관측 방향의 기저값 [60,16]
for step in range(4000):
    degree_to_use = min(step // 500, 3)                            # sh_degree_interval 흉내
    Kd = (degree_to_use + 1) ** 2
    coeffs = torch.cat([sh0, shN], dim=0)                          # [16,3]
    pred = torch.clamp_min(A_obs[:, :Kd] @ coeffs[:Kd] + 0.5, 0.0)
    loss = F.l1_loss(pred, obs_c)
    opt.zero_grad(); loss.backward(); opt.step()
```

이 셀의 목적은 "3DGS는 SH 계수를 적분(사영)으로 구하지 않고, **여러 카메라에서 본 색과 렌더 결과의 차이를 역전파**해서 맞춘다"는 점을 Gaussian **한 개**에 대해 축소 재현하는 것이다. 그래서 실제 학습 루프(`examples/simple_trainer.py`)의 "색 학습에 직접 관여하는" 장치들만 골라 옮겨 놓았다.

## 2. 셀의 각 줄 ↔ `simple_trainer.py` 1:1 대응표

| 흉내낸 요소 | 노트북 셀 | gsplat `examples/simple_trainer.py` (및 관련 파일) | 비고 |
|---|---|---|---|
| **파라미터 분리 sh0 / shN** | `sh0 = zeros(1,3)`, `shN = zeros(15,3)` 두 텐서를 따로 만들고 `torch.cat([sh0, shN])`으로 합쳐 사용 | `create_splats_with_optimizers`: `params.append(("sh0", Parameter(colors[:, :1, :]), sh0_lr))`, `params.append(("shN", Parameter(colors[:, 1:, :]), shN_lr))` → 렌더 시 `torch.cat([splats["sh0"], splats["shN"]], 1)` `[N,K,3]` | 실제는 Gaussian N개 × 16 × 3; 노트북은 N=1 |
| **학습률 비율 1/20** | `{"lr": 2.5e-2}` vs `{"lr": 2.5e-2 / 20}` | `sh0_lr: float = 2.5e-3`, `shN_lr: float = 2.5e-3 / 20` (Config 기본값). 원본 Inria 3DGS의 `feature_lr=0.0025`, `feature_rest` = `/20`과 동일 | 비율 1/20은 같고, **절대값은 노트북이 10배 크다** (아래 §3 참고) |
| **차수 램프업** | `degree_to_use = min(step // 500, 3)`; 총 4000스텝 → 0/500/1000/1500 스텝에서 차수 0→1→2→3 | `sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)`; `sh_degree_interval: int = 1000`, `sh_degree: int = 3`, `max_steps = 30_000` → 0/1000/2000/3000 스텝에서 차수 +1 | 형태는 동일, 간격만 500 vs 1000. 노트북은 총 스텝이 4000이라 절반으로 줄였다 |
| **+0.5 오프셋** | `A_obs[:, :Kd] @ coeffs[:Kd] + 0.5` | `gsplat/rendering.py` `_maybe_evaluate_sh`: `features = spherical_harmonics(sh_degree, means, viewmats, features, masks=masks)` 뒤에 `features + 0.5` | 계수가 전부 0일 때 색이 검정이 아닌 중간 회색(0.5)이 되도록 하는 장치. 초기화식 `(rgb−0.5)/C0`의 근거 |
| **clamp_min(0)** | `torch.clamp_min(... + 0.5, 0.0)` | 같은 함수: `if clamp: features = torch.clamp_min(features + 0.5, 0.0)` (주석: "make it apple-to-apple with Inria's CUDA Backend"). CUDA 경로에서는 `SHPostOp` 코드 `shift_relu -> max(x + 0.5, 0)`으로 융합 | 음수 색을 잘라내는 ReLU. Inria 원본 커널의 `max(result + 0.5, 0)`과 동일 |
| **L1 손실** | `loss = F.l1_loss(pred, obs_c)` | `l1loss = l1_loss(colors, pixels).mean()` (`gsplat/losses.py`) — 하지만 실제 최종 손실은 `loss = torch.lerp(l1loss, ssimloss, cfg.ssim_lambda)` = `0.8·L1 + 0.2·(1 − SSIM)`, `ssim_lambda = 0.2`, `ssim_loss = 1.0 - fused_ssim(...)` | 노트북은 L1만 사용. D-SSIM은 이미지 패치 구조 손실이므로 "점 하나의 색 60개"에는 정의되지 않는다 |
| **Adam 옵티마이저** | `torch.optim.Adam([...])` 기본 하이퍼파라미터 (`eps=1e-8`, `betas=(0.9, 0.999)`) | 파라미터 이름별로 옵티마이저 하나씩: `torch.optim.Adam([{"params": splats[name], "lr": lr*sqrt(BS), "name": name}], eps=1e-15/sqrt(BS), betas=(1-BS*(1-0.9), 1-BS*(1-0.999)), fused=True)`. 옵션에 따라 `SparseAdam` / `SelectiveAdam`(visible_adam) | 실제는 `eps=1e-15`(Inria와 동일)와 배치 크기 스케일링이 붙는다 |
| **관측 = 카메라 방향 샘플** | `obs_d = normalize(randn(60,3))` 무작위 방향 60개, `obs_c = f_star(obs_d)` | 카메라마다 `viewmats`에서 카메라 위치 `−Rᵀt`를 구해 `d = μ − o_cam` (`spherical_harmonics(sh_degree, means, viewmats, coeffs)`); 정답은 학습 이미지 픽셀 `pixels` | 실제로는 한 스텝에 카메라 1대(`batch_size=1`)씩 순회하지만, 노트북은 60개 방향을 매 스텝 한 번에 배치로 본다 |

정리하면 셀은 **"색 파라미터(sh0/shN) → 시점 방향에서 SH 평가 → +0.5, clamp → 관측과 비교 → Adam"** 이라는 3DGS 색 학습의 뼈대를, 파라미터 분리·학습률 비율·차수 램프업까지 포함해 그대로 옮긴 것이다.

## 3. 노트북이 **생략한** 3DGS 요소

축소 재현이므로 아래는 의도적으로 빠져 있다. 실제 `simple_trainer.py`와 비교할 때 혼동하지 말 것.

| 생략된 요소 | 실제 gsplat에서는 | 노트북에서는 |
|---|---|---|
| **알파 블렌딩(여러 Gaussian 혼합)** | `rasterization()`이 픽셀마다 깊이순으로 수백 개 Gaussian의 색을 `α`-합성(`Σ c_i α_i Π(1−α_j)`)해 최종 픽셀 색을 만들고, 손실은 그 픽셀에 대해 계산 | Gaussian 1개, 픽셀 = SH 평가값 자체. 겹침·가림이 없다 |
| **위치·스케일·회전·불투명도 동시 최적화** | `means`(1.6e-4 × scene_scale), `scales`(5e-3), `quats`(1e-3), `opacities`(5e-2)가 sh0/shN과 **같은 스텝에서** 함께 갱신되며, 색 그라디언트가 기하 파라미터에도 흘러간다 | SH 계수 16×3개만 학습. 기하는 존재하지 않는다 |
| **Densification / pruning** | `DefaultStrategy`: `refine_start_iter=500`~`refine_stop_iter=15_000` 동안 `refine_every=100` 스텝마다 화면 그라디언트(`grow_grad2d=2e-4`) 기준으로 split/duplicate, `prune_opa=0.005` 미만 제거, `reset_every=3000`마다 불투명도 리셋. 대안으로 `MCMCStrategy` | 없음. Gaussian 수가 1로 고정 |
| **학습률 스케줄** | `means` 옵티마이저에 `ExponentialLR(gamma=0.01 ** (1/max_steps))` → 30k 스텝에 걸쳐 1/100로 감쇠. (pose/appearance 옵티마이저도 동일 스케줄) SH 학습률은 실제로도 고정 | 스케줄 없음. 단, SH 학습률은 실제로도 고정이므로 이 부분은 차이가 아니다 |
| **절대 학습률 값 차이** | `sh0_lr = 2.5e-3`, `shN_lr = 2.5e-3/20 = 1.25e-4` (× `sqrt(batch_size·world_size)`) | `2.5e-2`, `1.25e-3` — **10배 크다**. 실제는 30k 스텝에 걸쳐 수렴하지만 노트북은 4000스텝 안에 수렴시켜 그림을 보여 줘야 하므로 크게 잡았다. 비율(1/20)만 보존 |
| **차수 램프업 간격** | `sh_degree_interval = 1000` (전체 30k의 1/30 × 3) | `500` (전체 4000의 1/8 × 3). 총 스텝에 맞춰 축소 |
| **손실 구성** | `0.8·L1 + 0.2·(1−SSIM)` (`ssim_lambda=0.2`), 선택적으로 `depth_loss`, 마스크 처리, `random_bkgd` | L1만. SSIM은 11×11 패치 기반 이미지 지표라 점 하나에는 적용 불가 |
| **배치·이미지 단위 손실** | 스텝마다 학습 이미지 1장(`batch_size=1`)을 렌더해 **모든 픽셀**에 대해 손실 → 한 Gaussian은 한 스텝에 한 카메라 방향만 관측 | 방향 60개를 한 텐서로 만들어 매 스텝 **전부** 관측(풀배치). 확률적이지 않은 결정적 경사하강 |
| **가시성 마스킹** | `_maybe_evaluate_sh`에서 `masks = (radii > 0)`로 컬링된 Gaussian은 SH 평가/그라디언트에서 제외. `SelectiveAdam`은 보이지 않은 Gaussian의 모멘트 갱신을 건너뜀 | 항상 모든 관측이 유효 |
| **Adam 세부 설정** | `eps=1e-15/sqrt(BS)`, `betas` 배치 보정, `fused=True`, 파라미터별 옵티마이저 분리(densification 시 상태 재배열을 위해) | PyTorch 기본 `eps=1e-8`, 옵티마이저 1개에 param group 2개 |
| **초기값** | `sh0 = rgb_to_sh(rgb) = (rgb−0.5)/C0` (SfM 포인트 색), `shN = 0` | `sh0 = 0`, `shN = 0` — 즉 시작색이 회색(0.5). 그래서 셀 끝에서 `coeffs_gd[0] += 0.5 / C0`로 오프셋을 DC에 흡수해 `f*`와 비교한다 |

## 4. 왜 이렇게 흉내냈는가 — 각 장치의 의도

- **sh0/shN 분리 + 1/20 학습률**: 기본색(DC)이 먼저 자리를 잡고, 시점 의존성(고차)은 천천히 배우게 한다. 고차가 빨리 움직이면 소수의 카메라에 과적합하여 관측 사이 방향에서 색이 요동한다(§4.2 최소제곱 표의 `n=8, L=3` 케이스).
- **차수 램프업**: 같은 맥락의 커리큘럼. 처음에는 `Kd=1`이므로 `shN`에는 그라디언트가 흐르지 않고(`coeffs[:1]`만 사용), 500스텝마다 기저를 열어 준다. 손실 곡선 그림 제목이 "500스텝마다 차수 +1"인 이유.
- **+0.5, clamp_min(0)**: 렌더러가 실제로 하는 후처리를 넣지 않으면 학습된 계수가 gsplat에서 기대하는 값과 다른 의미를 가진다. 특히 `clamp_min`은 그라디언트를 0으로 막는 비선형성이므로 최소제곱(선형)과 결과가 달라질 수 있다.
- **L1**: 3DGS 논문(Kerbl et al. 2023)의 손실 `(1−λ)·L1 + λ·D-SSIM`, λ=0.2 중 점 하나에 적용 가능한 절반만 가져온 것.

## 5. 한 줄 정리

노트북 셀은 3DGS 학습 중 **"SH 색 파라미터를 Adam으로 맞추는 부분"** 만 잘라 낸 것이다. 보존한 것은 sh0/shN 분리, 학습률 비율 1/20, `min(step // interval, 3)` 차수 램프업, `+0.5`·`clamp_min(0)`, L1, Adam, 카메라 방향 샘플 관측이고, 버린 것은 알파 블렌딩, 기하 파라미터 동시 최적화, densification/pruning, means 학습률 감쇠, D-SSIM, 이미지·배치 단위 손실, 그리고 절대 학습률(10배)과 램프업 간격(500 vs 1000)의 스케일이다.
