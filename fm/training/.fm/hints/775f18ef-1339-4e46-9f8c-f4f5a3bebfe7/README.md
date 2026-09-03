# simple_trainer의 선택적 손실 항들

**Q.** `simple_trainer`에 있는 선택적 손실 항들은 무엇인가? (기본은 모두 꺼짐)

**A.** `depth_loss`(SfM 포인트 disparity L1), `opacity_reg`/`scale_reg`(MCMC 전략용 정규화), `random_bkgd`(투명 영역이 배경색으로 도망가는 것 방지)가 있다.

---

## 1. 출발점: 기본 손실은 딱 두 항

워크스루 4단계(`training_walkthrough.py`)가 강조하는 것처럼, 3DGS의 기본 손실은 놀랄 만큼 단순하다.

$$\mathcal{L} = (1-\lambda)\,\mathcal{L}_{L1} + \lambda\,(1-\mathrm{SSIM}), \qquad \lambda = \texttt{ssim\_lambda} = 0.2$$

`examples/simple_trainer.py`에서는 이걸 `torch.lerp`로 한 줄에 쓴다.

```python
l1loss   = l1_loss(colors, pixels).mean()
ssimloss = ssim_loss(colors_ssim.permute(0, 3, 1, 2), pixels_ssim.permute(0, 3, 1, 2))
loss     = torch.lerp(l1loss, ssimloss, cfg.ssim_lambda)   # = 0.8*L1 + 0.2*SSIM
```

- **L1**: 픽셀 단위 색 재구성. 노이즈에 강건.
- **SSIM**: 11×11 가우시안 윈도우 기반 구조 유사도. 국소 대비/구조를 맞춰 L1만 쓸 때 생기는 뭉개짐을 억제.

**이 두 항만이 항상 켜져 있다.** 그 뒤에 붙는 것들이 이 카드가 묻는 "선택적 항"이고, `Config`의 기본값은 전부 off다.

## 2. 선택적 항 4개 — 기본값 한눈에

| 플래그 | 기본값 | 정체 | 켜지는 대표 상황 |
|---|---|---|---|
| `depth_loss` | `False` | SfM 포인트 depth 감독 (disparity L1) | 기하가 흐물거릴 때 / 뷰가 적을 때 (experimental) |
| `depth_lambda` | `1e-2` | 위 항의 가중치 | `depth_loss=True`일 때만 의미 |
| `opacity_reg` | `0.0` | `sigmoid(opacity)` 평균 페널티 | `mcmc` 프리셋에서 `0.01` |
| `scale_reg` | `0.0` | `exp(log_scale)` 평균 페널티 | `mcmc` 프리셋에서 `0.01` |
| `random_bkgd` | `False` | 매 스텝 랜덤 배경색 합성 | 투명/floater가 생기는 씬 |

`Config` 주석 자체가 이 성격을 그대로 말해준다.

```python
# Use random background for training to discourage transparency
random_bkgd: bool = False
# Opacity regularization
opacity_reg: float = 0.0
# Scale regularization
scale_reg: float = 0.0
# Enable depth loss. (experimental)
depth_loss: bool = False
# Weight for depth loss
depth_lambda: float = 1e-2
```

> 엄밀히 말하면 `random_bkgd`는 "손실 항"이 아니라 **렌더 출력에 가하는 증강**이다. 손실 식에 새 항을 더하지 않고, L1/SSIM에 들어가는 `colors`를 바꿔서 결과적으로 손실의 의미를 바꾼다. 하지만 워크스루는 "선택적 항" 목록에 같이 묶어 소개하므로 카드도 같이 외운다.

---

## 3. `depth_loss` — SfM 포인트로 기하를 붙잡기

포토메트릭 손실만으로는 "색만 맞고 깊이는 엉뚱한" 해가 얼마든지 존재한다(shape-radiance ambiguity). SfM이 이미 삼각측량해 둔 3D 포인트를 약한 깊이 감독으로 재활용하는 게 이 항이다.

**(a) 데이터 쪽 준비.** `depth_loss`를 켜면 데이터셋이 `load_depths=True`로 만들어진다.

```python
Dataset(..., load_depths=cfg.depth_loss, ...)
```

`examples/datasets/colmap.py`가 하는 일은 depth map 파일을 읽는 게 아니라, **씬의 SfM 3D 포인트를 그 뷰의 카메라로 직접 투영**하는 것이다.

```python
points_world = self.parser.points[point_indices]          # 이 이미지가 관측한 3D 포인트만
points_cam   = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
points_proj  = (K @ points_cam.T).T
points  = points_proj[:, :2] / points_proj[:, 2:3]        # (M, 2) 픽셀 좌표
depths  = points_cam[:, 2]                                # (M,)  카메라 z = GT depth
# 화면 밖 / z<=0 은 필터링
```

즉 GT는 **희소한 M개 점의 (픽셀좌표, depth) 쌍**이다. 조밀한 depth map이 아니다.

**(b) 렌더 쪽.** depth를 감독하려면 depth를 렌더해야 하므로 render_mode가 바뀐다.

```python
render_mode="RGB+ED" if cfg.depth_loss else "RGB",
...
if renders.shape[-1] == 4:
    colors, depths = renders[..., 0:3], renders[..., 3:4]
```

`ED` = **expected depth**(알파 가중 평균 깊이). 4번째 채널로 함께 나온다.

**(c) 손실 계산.** 렌더된 depth 맵에서 SfM 포인트 위치의 값을 `grid_sample`로 뽑고, **disparity(역depth) 공간**에서 L1을 잰다.

```python
points = torch.stack([points[:, :, 0] / (width - 1) * 2 - 1,
                      points[:, :, 1] / (height - 1) * 2 - 1], dim=-1)  # [-1,1] 정규화
grid   = points.unsqueeze(2)                                  # [1, M, 1, 2]
depths = F.grid_sample(depths.permute(0, 3, 1, 2), grid, align_corners=True)
depths = depths.squeeze(3).squeeze(1)                         # [1, M]
depthloss = depth_l1_loss(depths, depths_gt, scene_scale=self.scene_scale)
loss += depthloss * cfg.depth_lambda
```

`gsplat/losses.py`의 구현:

```python
def depth_l1_loss(pred_depth, gt_depth, scene_scale=1.0):
    disp    = torch.where(pred_depth > 0.0, 1.0 / pred_depth, torch.zeros_like(pred_depth))
    disp_gt = torch.where(gt_depth   > 0.0, 1.0 / gt_depth,   torch.zeros_like(gt_depth))
    return F.l1_loss(disp, disp_gt) * scene_scale
```

**왜 disparity인가** — 카드가 "disparity L1"이라고 못 박은 이유:
- depth 공간 L1은 먼 점의 절대오차가 압도적으로 커서, 손실이 "하늘/원거리 배경"에 지배된다. 1/d로 바꾸면 **가까운 기하에 가중치가 실리고 원거리 오차는 자동으로 눌린다.**
- `d <= 0`(무효)은 양쪽 다 0으로 처리해 그 점의 기여를 0으로 만드는 sentinel 규약.
- `scene_scale`을 곱해 씬 크기에 따라 항의 크기가 들쭉날쭉해지지 않게 정규화한다.
- 최종 가중치는 `depth_lambda=1e-2`. **매우 약한 힌트**로만 쓰겠다는 뜻이다 (SfM 포인트 자체에 노이즈가 있으므로).

## 4. `opacity_reg` / `scale_reg` — MCMC를 위한 두 줄짜리 정규화

구현은 허무할 정도로 짧다 (`gsplat/losses.py`).

```python
def opacity_reg_loss(opacities: Tensor) -> Tensor:
    """Opacity regularization: mean of sigmoid activations."""
    return torch.sigmoid(opacities).mean()      # 파라미터는 pre-sigmoid logit

def scale_reg_loss(log_scales: Tensor) -> Tensor:
    """Scale regularization: mean of exponentiated log-scales."""
    return torch.exp(log_scales).mean()         # 파라미터는 log-scale
```

호출부는 손실 마지막, `loss.backward()` 바로 앞이다.

```python
# regularizations
if cfg.opacity_reg > 0.0:
    loss += cfg.opacity_reg * opacity_reg_loss(self.splats["opacities"])
if cfg.scale_reg > 0.0:
    loss += cfg.scale_reg * scale_reg_loss(self.splats["scales"])
```

포인트: 두 함수 모두 **활성화 함수를 통과시킨 뒤(실제 물리량으로) 평균**을 낸다. 로짓/로그값에 직접 L1을 걸면 의미가 달라진다.

- `opacity_reg`: 불투명도 **총량**에 세금을 매긴다 → 재구성에 기여하지 않는 Gaussian은 opacity가 0 쪽으로 밀려 죽고, MCMC의 `min_opacity=0.005` prune에 걸려 회수된다. 즉 **"쓸모없는 Gaussian을 스스로 신고하게 만드는" 압력**.
- `scale_reg`: 크기에 세금을 매긴다 → 화면을 넓게 덮어 손실을 싸게 깎는 거대 splat(floater, 바늘 모양 아티팩트)을 억제.

**왜 "MCMC 전략용"인가.** `DefaultStrategy`는 gradient 휴리스틱으로 분할/복제/prune을 하지만, `MCMCStrategy`는 성격이 다르다 (`gsplat/strategy/mcmc.py` docstring):

> - Periodically **teleport** GSs with low opacity to a place that has high opacity.
> - Periodically **introduce new GSs** sampled based on the opacity distribution.
> - Periodically **perturb** the GSs locations.

즉 MCMC는 개수 상한 `cap_max` 아래에서 **opacity 분포를 샘플링 확률로 삼아 Gaussian을 재배치**한다. 그래서 opacity가 "이 Gaussian이 정말 필요한가"를 정직하게 반영해야 relocate가 제대로 동작하고, 정규화가 그 신호를 만들어준다. 워크스루 마무리 절이 정확히 이렇게 적어둔 이유다.

> `MCMCStrategy`: 휴리스틱 분할 대신 SGLD 방식 — Gaussian 수 상한을 두고 opacity 기반 확률적 재배치 + 노이즈 주입. `opacity_reg`/`scale_reg`와 함께 사용

CLI 프리셋에도 이 짝이 하드코딩돼 있다.

```python
"mcmc": (
    "Gaussian splatting training using densification from the paper
     '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
    Config(
        init_opa=0.5,      # default 프리셋은 0.1
        init_scale=0.1,    # default 프리셋은 1.0
        opacity_reg=0.01,
        scale_reg=0.01,
        strategy=MCMCStrategy(verbose=True),
    ),
),
```

→ `python simple_trainer.py mcmc ...` 로 실행하면 이 두 항이 자동으로 `0.01`로 켜진다. `default` 프리셋에서는 `0.0`, 즉 꺼짐.

## 5. `random_bkgd` — "투명하게 남는 게 이득"인 지름길을 막기

```python
if cfg.random_bkgd:
    bkgd = torch.rand(1, 3, device=device)
    colors = colors + bkgd * (1.0 - alphas)
```

래스터라이저는 색 `colors`와 누적 알파 `alphas`를 따로 준다. `alpha < 1`인 픽셀은 "덜 채워진" 픽셀이고, 위 식은 그 빈 공간에 배경색 `bkgd`를 합성(over-composite)한다.

**고정 배경(예: 검정)이면 왜 문제인가.** GT 픽셀이 어두우면, Gaussian을 제대로 배치하지 않고 그냥 `alpha`를 낮춰 검정 배경이 비치게 두는 것만으로 손실이 낮아진다. 최적화가 기하를 만드는 대신 **투명도로 도망친다**(카드 문구의 "배경색으로 도망가는 것"). 그 결과가 반투명 안개/floater다.

**랜덤 배경이 막는 방식.** `bkgd`가 매 스텝 새로 뽑히는 uniform 랜덤 RGB이므로, 특정 배경색을 노린 해가 존재할 수 없다. `alpha < 1`인 픽셀은 스텝마다 다른 색이 섞여 들어와 손실이 평균적으로 커진다. 유일하게 안전한 해는 **`alpha → 1`로 만들어 배경 항 `(1-alpha)`를 없애는 것** — 즉 불투명한 실제 기하를 만드는 쪽으로 gradient가 밀린다.

주의: 진짜로 비어 있어야 하는 영역(하늘, 마스크된 바깥)까지 불투명하게 채우라는 압력이 되므로 만능은 아니다. 기본이 off인 이유.

---

## 6. 손실 조립 전체 순서 (읽는 순서 = 코드 순서)

```
1) L1 + SSIM                      ← 항상
   (mask가 있으면 L1은 masked 픽셀 제외,
    SSIM은 patch 기반이라 양쪽을 0으로 눌러 계산)
2) + depth_lambda * depth_l1(disparity)     ← if depth_loss
3) + post_processing 정규화                  ← if bilateral_grid(TV) / ppisp
4) + opacity_reg * mean(sigmoid(opacity))    ← if opacity_reg > 0
   + scale_reg   * mean(exp(log_scale))      ← if scale_reg > 0
5) loss.backward()
```

3번(`bilateral_grid`의 total-variation, `ppisp` 정규화)은 이 카드가 묻는 목록에는 없지만 실제로 같은 자리에 더해진다. 카드가 꼽은 4개는 **"밀도화/기하 품질에 직접 관계된" 선택 항**이라는 점에서 한 묶음이다.

## 7. 암기 포인트

- 항상 켜진 것: **L1 + 0.2·SSIM**. 나머지는 전부 opt-in.
- `depth_loss`: GT는 **조밀 depth map이 아니라 SfM 포인트 투영**, 손실은 **disparity(1/d) 공간 L1**, 가중치 `1e-2`, render_mode가 `RGB+ED`로 바뀜.
- `opacity_reg` = `sigmoid(o).mean()`, `scale_reg` = `exp(s).mean()` — **활성화 후 평균**. `mcmc` 프리셋에서 둘 다 `0.01`.
- `random_bkgd`는 손실 항이 아니라 **렌더 합성 트릭**: `colors + rand_rgb * (1 - alpha)` → 투명으로 도망가는 해를 매 스텝 다른 색으로 처벌.

## 참고 위치

- `examples/simple_trainer.py` — `Config` 플래그(≈L171~250), 손실 조립(≈L930~997), CLI 프리셋(≈L1583~1592)
- `gsplat/losses.py` — `l1_loss`, `ssim_loss`, `depth_l1_loss`(L209), `opacity_reg_loss`(L675), `scale_reg_loss`(L689)
- `examples/datasets/colmap.py` — `load_depths` 분기(≈L511~532)
- `gsplat/strategy/mcmc.py` — `MCMCStrategy` docstring/기본값(L40~92)
- `fm/training/.fm/assets/training_walkthrough.py` — 4단계 "손실 함수", 마무리 "여기서 더 볼 것들"
