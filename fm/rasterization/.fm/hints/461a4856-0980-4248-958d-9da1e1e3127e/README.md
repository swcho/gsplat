# `render_mode="RGB+ED"` — RGB 3채널 + 기대 깊이 1채널

## 한 줄 요약

`rasterization(..., render_mode="RGB+ED")`는 출력 텐서를 `[C, H, W, 4]`로 만든다.
앞 3채널은 평소의 RGB, **마지막 1채널이 기대 깊이(Expected Depth)** 다.
깊이는 별도의 깊이 커널로 계산되는 게 아니라, **색과 똑같이 "채널 하나"로 취급되어 같은 알파 블렌딩 루프에서 함께 누적**되고,
커널이 끝난 뒤 파이썬 쪽에서 **`render_alphas`로 한 번 나눠서** 기대값으로 바꾼다.

워크스루의 해당 셀(`.fm/assets/rasterization_walkthrough.py`, 0절):

```python
render, alpha, meta = rasterization(
    means, quats, scales, opacities, sh_coeffs, viewmats, Ks, W, H,
    sh_degree=SH_DEGREE, packed=False, render_mode="RGB+ED",
)
# render_colors (3, H, W, 4)   render_alphas (3, H, W, 1)

axes[c].imshow(render[c, ..., :3].clamp(0, 1).cpu())   # RGB
d = render[0, ..., 3].cpu()                            # ← 마지막 채널 = expected depth
d[alpha[0, ..., 0].cpu() < 0.5] = float("nan")         # 빈 영역은 마스킹해서 본다
```

---

## 1. 어떻게 "채널 하나"가 되는가

`gsplat/rendering.py`는 래스터화 직전에 색 텐서 뒤에 깊이를 그냥 이어 붙인다.

```python
# gsplat/rendering.py  (rasterize to pixels 직전)
if render_mode_has_depth_channel(render_mode) and render_mode_has_color(render_mode):
    colors = torch.cat((colors, depths[..., None]), dim=-1)   # [..., N, 3] → [..., N, 4]
```

- `depths`는 ③ 투영 단계(`fully_fused_projection`)가 뱉은 `meta["depths"] [C,N]`, 즉 **카메라 좌표계의 z**다
  (Gaussian **중심**의 z 하나. Gaussian 내부에서 픽셀마다 달라지지 않는 3DGS의 근사).
- 그래서 CUDA 커널 `rasterize_to_pixels_3dgs_fwd_kernel<CDIM,...>` 입장에서는 채널 수가 3이 아니라 4일 뿐,
  코드 경로는 완전히 동일하다. 워크스루 6절의 `rasterize_naive`도 `colors[N,D]`의 D가 4가 되는 것뿐이다.
- 채널 수는 아무 값이나 되지 않는다. 워크스루 마지막 절이 짚듯 `Config.h`의 `GSPLAT_NUM_CHANNELS`에
  **컴파일 시점에 박아 둔 값**들만 인스턴스화되어 있다.
- `depth-only` 모드(`"D"`, `"ED"`)일 때는 `colors = depths[..., None]`로 **1채널짜리 "색"** 을 넘긴다.

---

## 2. 블렌딩 수식 — D와 ED의 차이

워크스루 6절의 알파 블렌딩을 가중치 $w_i$로 다시 쓰면:

$$\alpha_i=\min(0.99,\;o_i e^{-\sigma_i}),\qquad
T_i=\prod_{j<i}(1-\alpha_j),\qquad
w_i=\alpha_i T_i$$

$$C_p=\sum_i w_i\,c_i,\qquad
\boxed{\;\text{render\_alpha}=1-T=\sum_i w_i\;}$$

여기서 **알파 자체가 가중치의 합**이라는 게 핵심이다. 깊이 채널도 똑같이 누적되므로 커널이 내놓는 값은

$$\text{depth\_raw}=\sum_i w_i\,z_i$$

이고, 이게 그대로 **`"D"` (Accumulated Depth, 누적 깊이)** 다. 여기에 정규화를 한 번 더 하면 **`"ED"` (Expected Depth, 기대 깊이)** 다.

| 모드 | 수식 | 정규화 | 성질 |
|---|---|---|---|
| `"D"` / `"RGB+D"` | $\sum_i w_i z_i$ | 없음 | **alpha가 곱해진 값**. 반투명·빈 영역에서 0쪽으로 끌려간다. 물리적 거리가 아님 |
| `"ED"` / `"RGB+ED"` | $\dfrac{\sum_i w_i z_i}{\sum_i w_i}$ | `/ alpha` | **z의 가중 평균**. 단위가 그대로 카메라 z(미터). 사람이 읽을 수 있는 깊이 |

관계는 단순하다: $\text{ED} = \text{D} / \text{alpha}$.
즉 `"RGB+D"`로 받아 놓고 직접 `render[..., 3:4] / alpha`를 해도 `"RGB+ED"`와 같은 값이 나온다.

> 참고: 최신 gsplat에는 소문자 계열 `"d"`, `"Ed"`, `"RGB-d"`, `"RGB-Ed"`가 추가됐다.
> 대문자 D/ED는 **투영 깊이(카메라 z)**, 소문자 d/Ed는 **광선 방향 실제 거리(hit distance)** 다.
> 정규화 규칙(대문자 E가 붙으면 alpha로 나눔)은 동일하다. `rendering.py`의
> `render_mode_has_expected_depth()`가 `{"Ed", "ED", "RGB-Ed", "RGB+ED"}`를 판정한다.

---

## 3. 왜 alpha로 나누는가 — 가중 평균이기 때문

$\sum_i w_i = 1-T = \text{alpha}$ 는 **"이 픽셀이 얼마나 채워졌는가"** 다.
가중 평균의 정의상 분모는 가중치의 합이어야 하고, 여기서 그 합이 정확히 alpha다.

- **alpha ≈ 1 (불투명한 표면)**: $\text{D}\approx\text{ED}$. 나누나 마나 큰 차이가 없다.
- **alpha = 0.3 (성긴/반투명 영역)**: 표면이 진짜로 3 m에 있어도 `"D"`는 $0.3\times 3 = 0.9$를 내놓는다.
  "0.9 m에 뭔가 있다"가 아니라 **"3 m짜리 표면이 30 %만 있다"** 는 뜻인데, 깊이맵으로 보면 무조건 앞으로 튀어나온 것처럼 보인다.
  alpha로 나눠 3.0을 복구해야 비로소 거리다.
- **정규화 없이는 loss도 못 건다.** GT 깊이는 alpha와 무관한 물리량이므로 $\sum w_i z_i$와 직접 비교하면
  "깊이를 맞춰라"가 아니라 "불투명도를 1로 밀어라"라는 엉뚱한 그래디언트가 섞인다.

구현은 커널 밖(파이썬)에서 한 줄이다.

```python
# gsplat/rendering.py
if render_mode_has_expected_depth(render_mode):
    render_depth = render_colors[..., -1:] / render_alphas.clamp(min=1e-10)
    render_colors = torch.cat([render_colors[..., :D], render_depth], dim=-1)
```

`clamp(min=1e-10)`이 **0으로 나누기 방어**다. Gaussian이 하나도 안 닿은 픽셀은 alpha=0이 되어
$0/10^{-10}=0$, 즉 **깊이 0**이 나온다. 이건 "카메라에 붙어 있다"가 아니라 **"정보 없음"** 이라는 뜻이므로
그대로 시각화하거나 loss에 넣으면 안 된다. 워크스루가 `alpha < 0.5`인 픽셀을 `NaN`으로 칠하고 그리는 이유가 이것이다.

---

## 4. 배경(background) 처리

RGB 쪽 배경 합성은 커널 말미에 `pix_out += background * T` 로 들어간다(워크스루 6절 의사코드).
그런데 **깊이 채널에는 배경 색을 넣으면 안 된다** — "배경의 깊이"라는 게 정의되지 않기 때문이다.
그래서 `rendering.py`가 배경 텐서에도 깊이 자리만큼 **0을 이어 붙인다**.

```python
if backgrounds is not None:
    backgrounds = torch.cat(
        [backgrounds, torch.zeros(batch_dims + (C, 1), device=backgrounds.device)], dim=-1
    )
```

`"D"`/`"ED"` 단독 모드에서는 배경 전체를 `torch.zeros(...)`로 갈아치운다. 결과적으로:

- 깊이 채널에 대한 배경 기여는 항상 0 → `"D"`의 깊이는 순수하게 $\sum w_i z_i$로 남는다.
- 빈 픽셀의 ED는 위에서 본 대로 0.
- **따라서 하류에서 쓸 때는 항상 `render_alphas`를 같이 들고 다니며 마스크로 써야 한다.**
  (시각화 시 `alpha < 0.5` 마스킹, loss 시 유효 픽셀만 샘플링.)

또 하나: 워크스루가 짚듯 투과율이 `TRANSMITTANCE_THRESHOLD = 1e-4` 이하가 되면 그 픽셀은 조기 종료하고,
`α < 1/255`인 Gaussian은 건너뛴다. 즉 뒤쪽 아주 먼 Gaussian의 z는 애초에 평균에 들어오지 않는다 —
이게 ED가 "앞쪽 표면 근처"로 잘 수렴하는 실용적 이유이기도 하다.

---

## 5. 학습에서의 깊이 감독 (depth supervision)

`examples/simple_trainer.py`가 표준 사용례다. `--depth_loss`를 켜면 렌더 모드가 통째로 바뀐다.

```python
render_mode="RGB+ED" if cfg.depth_loss else "RGB",
...
if renders.shape[-1] == 4:
    colors, depths = renders[..., 0:3], renders[..., 3:4]   # 4채널을 색/깊이로 쪼갠다
else:
    colors, depths = renders, None
```

그 다음 loss는:

```python
if cfg.depth_loss:
    # COLMAP sparse 3D point를 이 뷰에 투영한 픽셀 좌표 points를 [-1,1]로 정규화
    grid   = points.unsqueeze(2)                                    # [1, M, 1, 2]
    depths = F.grid_sample(depths.permute(0, 3, 1, 2), grid, align_corners=True)
    depths = depths.squeeze(3).squeeze(1)                           # [1, M]
    depthloss = depth_l1_loss(depths, depths_gt, scene_scale=self.scene_scale)
    loss += depthloss * cfg.depth_lambda
```

포인트별로 정리하면:

1. **GT는 조밀한 깊이맵이 아니라 sparse SfM 포인트**다. COLMAP이 각 이미지에서 본 3D 점의 z가 `depths_gt`,
   그 투영 픽셀 위치가 `points`. 그래서 렌더된 깊이맵 전체가 아니라 `grid_sample`로 **그 위치들만 뽑아** 비교한다.
   (`load_depths=cfg.depth_loss`로 데이터로더에서 미리 준비한다.)
2. **비교는 disparity(역깊이) 공간에서** 한다 (`gsplat/losses.py`):
   ```python
   disp    = torch.where(pred_depth > 0.0, 1.0 / pred_depth, torch.zeros_like(pred_depth))
   disp_gt = torch.where(gt_depth   > 0.0, 1.0 / gt_depth,   torch.zeros_like(gt_depth))
   return F.l1_loss(disp, disp_gt) * scene_scale
   ```
   깊이 공간에서 L1을 걸면 먼 점의 절대 오차가 압도적으로 커져 학습이 배경에 끌려간다.
   1/z로 바꾸면 가까운 구조에 가중이 실리고, 하늘·먼 배경(z가 큰 곳)의 영향이 자연스럽게 줄어든다.
   `pred_depth > 0.0` 가드가 앞서 말한 **alpha=0 → depth=0 픽셀을 걸러 내는 역할**도 겸한다.
3. **그래디언트 경로**: 깊이 채널은 색과 같은 커널·같은 backward를 타므로,
   $\partial L/\partial(\text{ED})$가 $w_i$와 $z_i$ 양쪽으로 흐른다. 즉 깊이 loss는
   **Gaussian의 위치(특히 z), 불투명도, 스케일**을 동시에 움직인다.
   덕분에 "색은 맞는데 기하가 틀린" 떠다니는 Gaussian(floater)이 억제되고, 텍스처 없는 면에서 깊이가 안정된다.
   `depth_lambda`(기본 매우 작음)로 세기를 조절하고, `scene_scale`로 씬 크기 차이를 흡수한다.

### ED vs. median depth

ED는 **평균**이라 깊이 불연속(물체 경계)에서 앞뒤 표면이 섞여 중간값이 나오는 *edge fattening*이 생긴다.
2DGS(`rasterization_2dgs`)에는 `depth_mode="expected" | "median"` 선택지가 있고,
median은 누적 가중치가 0.5를 넘는 지점의 z를 취해 경계를 더 날카롭게 만든다.
평면 추출·메시화가 목적이면 median, 미분 가능한 부드러운 감독이 목적이면 ED가 낫다.

### 뷰어에서

`simple_trainer.py`의 뷰어는 두 모드를 따로 노출한다:

```python
RENDER_MODE_MAP = {"rgb": "RGB", "depth(accumulated)": "D", "depth(expected)": "ED", "alpha": "RGB"}
```

`"D"`를 고르면 alpha가 곱해진 그림이라 성긴 영역이 어둡게(가깝게) 보이고, `"ED"`를 고르면 거리 그대로 보인다.
둘 다 `near/far`로 [0,1] 정규화한 뒤 컬러맵을 씌워 표시한다.

---

## 6. 외우기용 정리

- `"RGB+ED"` → 출력 `[C,H,W,4]`, **마지막 채널이 깊이**. alpha는 언제나 별도 `[C,H,W,1]`.
- 깊이는 특별 대접 없이 **색 채널 하나로 concat되어 같이 블렌딩**된다.
- `D = Σ wᵢzᵢ` (누적) / `ED = Σ wᵢzᵢ / Σ wᵢ` (기대) / `Σ wᵢ = alpha` / **`ED = D / alpha`**.
- **나누는 이유**: 가중 평균의 분모가 곧 alpha라서. 안 나누면 "얇은 곳일수록 가까워 보이는" 값이 된다.
- **배경 깊이 기여는 0**, 빈 픽셀 ED는 `0/1e-10 = 0` → 반드시 alpha로 마스킹.
- **학습**: `--depth_loss` → `RGB+ED` → 4번째 채널을 sparse SfM 포인트 위치에서 `grid_sample` → **disparity L1**.

## 관련 파일

- 카드 원본: `.fm/assets/rasterization_walkthrough.py` (0절 호출/시각화, 6절 알파 블렌딩, 마지막 절 확장 포인트)
- `gsplat/rendering.py`: `RenderMode`, `render_mode_has_expected_depth()`, 깊이 concat, `/ render_alphas.clamp(min=1e-10)`
- `examples/simple_trainer.py`: `render_mode="RGB+ED" if cfg.depth_loss else "RGB"`, `grid_sample` 감독, `RENDER_MODE_MAP`
- `gsplat/losses.py`: `depth_l1_loss` (disparity 공간 L1)
