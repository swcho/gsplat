# `fully_fused_projection()` — 2단계 투영이 산출하는 값들

## 카드 요약

`rasterization()`은 4개의 CUDA 커널 단계로 이루어진다 (asset `training_walkthrough.py` 3단계 절).

| # | 단계 | 함수 |
|---|---|---|
| 1 | SH 평가 | `spherical_harmonics` |
| **2** | **투영 (EWA splatting)** | **`fully_fused_projection`** |
| 3 | 타일 교차 | `isect_tiles` |
| 4 | 픽셀 래스터화 | `rasterize_to_pixels` |

2단계의 출력이 카드가 묻는 것이다. `gsplat/cuda/_wrapper.py:819`의 반환값 4+1개:

| 반환값 | shape (packed=False) | 의미 |
|---|---|---|
| `radii` | `[..., C, N, 2]` **int32** | 화면상 픽셀 단위 x/y 반경. **0이면 컬링됨(무효)** |
| `means2d` | `[..., C, N, 2]` | 투영된 중심의 화면 좌표(픽셀) |
| `depths` | `[..., C, N]` | 카메라 좌표계 z 깊이 (정렬 키) |
| `conics` | `[..., C, N, 3]` | **2D 공분산의 역행렬** 상삼각 3원소 |
| `compensations` | `[..., C, N]` | `rasterize_mode="antialiased"`일 때만. 뷰 의존 불투명도 보정 |

카드 answer의 "2D 공분산(conic)"이라는 표현에는 중요한 함정이 하나 있다. **`conics`는 2D 공분산 그 자체가 아니라 그 역행렬**이다. 아래 6절에서 이유를 설명한다.

---

## 1. "fully fused"가 뜻하는 것

docstring이 직접 밝힌다.

> This function fuse the process of computing covariances (`quat_scale_to_covar_preci()`), transforming to camera space (`world_to_cam()`), and projection (`proj()`).

즉 원래 3개의 연산이던 것을 커널 하나로 합쳤다. Gaussian이 수백만 개인데 각 단계마다 `[C, N, 3, 3]` 크기의 중간 텐서를 HBM에 쓰고 다시 읽으면 대역폭에서 죽는다. fused 커널은 Gaussian 하나를 레지스터에 올린 뒤 3단계를 끝내고 최종 출력만 쓴다. 그래서 파라미터로 `covars`를 직접 주거나 `{quats, scales}`를 주면 커널 내부에서 공분산으로 변환한다(둘 중 하나만).

`gsplat/rendering.py:850`의 non-fused 경로와 `gsplat/cuda/_torch_impl.py:262` `_fully_fused_projection()`은 같은 수학의 PyTorch 참조 구현이므로, 아래 설명은 그 코드를 따라간다.

---

## 2. 입력: 3D 공분산 $\Sigma = R S S^\top R^\top$

Gaussian의 모양은 3×3 공분산 행렬 하나로 표현된다. 이것을 직접 학습하면 학습 도중 양의 정부호(positive-definite)가 깨져 "뒤집힌" 공분산이 나올 수 있으므로, gsplat은 회전과 스케일로 **분해된 형태**를 파라미터로 둔다.

$$\Sigma = R\,S\,S^\top R^\top,\qquad S=\mathrm{diag}(s_x,s_y,s_z),\ R=R(q)$$

- `quats` $q$ → 회전행렬 $R$ (내부에서 normalize됨)
- `scales`는 log 공간에 저장되고 `exp`로 활성화되므로 $s_i>0$ 보장

이 형태는 임의의 $s_i>0$에 대해 항상 양의 정부호다. $\Sigma = (RS)(RS)^\top$ 꼴이므로 어떤 벡터 $v\ne 0$에 대해 $v^\top\Sigma v = \lVert S^\top R^\top v\rVert^2 > 0$. asset의 2단계 절이 "항상 양의 정부호가 보장된다"고 적은 게 이것이다.

---

## 3. 월드 → 카메라 (`_world_to_cam`)

`viewmats`의 회전 $R_{cw}$, 평행이동 $t$로

$$\mu_c = R_{cw}\mu + t,\qquad \Sigma_c = R_{cw}\,\Sigma\,R_{cw}^\top$$

공분산의 변환에는 $t$가 등장하지 않는다. 공분산은 "중심으로부터의 퍼짐"이므로 평행이동에 불변하다. 그리고 $R_{cw}$는 직교행렬이라 이 변환은 모양·부피를 보존한다 — 여기까지는 근사가 하나도 없다.

---

## 4. 원근 투영과 EWA splatting (`_persp_proj`)

여기가 유일하게 근사가 들어가는 곳이다. 핀홀 카메라의 투영

$$\pi(x,y,z) = \left(f_x\frac{x}{z}+c_x,\ f_y\frac{y}{z}+c_y\right)$$

는 **비선형**이다. Gaussian을 비선형 사상으로 밀면 결과는 더 이상 Gaussian이 아니다. 그러므로 $\mu_c$ 주변에서 **1차 테일러 전개(=야코비안)** 로 국소 선형화한다.

$$J = \frac{\partial\pi}{\partial\mu_c}\bigg|_{\mu_c} = \begin{pmatrix} \dfrac{f_x}{z} & 0 & -\dfrac{f_x x}{z^2}\\[6pt] 0 & \dfrac{f_y}{z} & -\dfrac{f_y y}{z^2}\end{pmatrix}\in\mathbb{R}^{2\times 3}$$

$$\boxed{\ \Sigma' = J\,\Sigma_c\,J^\top\ }\in\mathbb{R}^{2\times 2}$$

이것이 **EWA(Elliptical Weighted Average) splatting** (Zwicker et al., *EWA Volume Splatting*, 2001)이며, 3DGS가 이 논문에서 그대로 가져온 부분이다. 3×3 공분산이 2×3 야코비안에 의해 2×2로 사영되면서 3D 타원체가 화면상의 **타원**(footprint)이 된다.

코드에서 두 가지 실무적 디테일:

```python
lim_x_pos = (width - cx) / fx + 0.3 * tan_fovx
tx = tz * torch.clamp(tx / tz, min=-lim_x_neg, max=lim_x_pos)
```

화면 바깥 멀리 있는 Gaussian은 $x/z$가 커서 $-f_x x/z^2$ 항이 폭주한다. 그래서 **야코비안을 평가하는 위치를 시야각 근처로 clamp**해 수치적으로 미친 $\Sigma'$가 나오지 않게 한다. `fisheye`/`ortho` 카메라는 각각 다른 $J$를 쓰고(`_fisheye_proj`, `_ortho_proj`), `ftheta` 같은 왜곡 카메라는 야코비안 대신 unscented transform 경로(`fully_fused_projection_with_ut`)로 간다.

---

## 5. `means2d`와 `depths`

$$\texttt{means2d} = \left(f_x\frac{x_c}{z_c}+c_x,\; f_y\frac{y_c}{z_c}+c_y\right),\qquad \texttt{depths}=z_c$$

- `means2d`는 **픽셀 단위** 좌표다(NDC 아님). 4단계에서 픽셀 중심과의 차 $\Delta = p - \texttt{means2d}$를 만드는 데 쓴다.
- `depths`는 3단계 `isect_tiles`에서 **정렬 키의 하위 비트**로 들어간다. 타일 안에서 앞→뒤 알파 블렌딩 순서를 정하는 값이 바로 이것이다.
- `means2d`는 동시에 **밀도화 전략의 입력**이다. asset 5단계가 설명하듯 `DefaultStrategy.step_pre_backward()`가 `info["means2d"].retain_grad()`를 걸고, backward 후 그 gradient 크기로 duplicate/split을 판정한다. 2단계 출력 중 학습 제어에 직접 쓰이는 유일한 값이다.

---

## 6. 왜 `conics`(= 역행렬)인가

4단계 알파 계산은 픽셀마다 다음을 평가한다.

$$\alpha_i = o_i\exp\!\left(-\tfrac12\,\Delta^\top\Sigma'^{-1}\Delta\right)$$

즉 필요한 것은 $\Sigma'$가 아니라 **$\Sigma'^{-1}$** 다. 픽셀은 Gaussian 하나당 수십~수백 개가 걸리므로, 역행렬을 픽셀마다 구하면 낭비다. 그래서 투영 단계에서 카메라·Gaussian 쌍마다 **한 번만** 역행렬을 구해 상삼각 3원소로 넘긴다.

```python
det = σ11*σ22 - σ12*σ21
conics = [σ22/det, -(σ12+σ21)/2/det, σ11/det]   # = [a, b, c]
```

2×2 역행렬 공식 $\Sigma'^{-1} = \frac{1}{\det}\begin{pmatrix}\sigma_{22} & -\sigma_{12}\\ -\sigma_{12} & \sigma_{11}\end{pmatrix}$ 그대로다. 대칭이므로 6개가 아니라 3개면 충분하고, 이름이 "conic"인 이유는 $\Delta^\top\Sigma'^{-1}\Delta = \text{const}$가 **원뿔곡선(타원)** 의 방정식이기 때문이다.

래스터화 쪽(`_torch_impl.py:786`)에서 실제로 이렇게 소비된다.

```python
sigmas = 0.5*(c[:,0]*dx**2 + c[:,2]*dy**2) + c[:,1]*dx*dy
alphas = clamp_max(opacities * exp(-sigmas), MAX_ALPHA)
```

곱셈·덧셈만 남는다. 나눗셈도 역행렬도 없다.

### low-pass filter `eps2d`

역행렬을 구하기 **전에** 한 줄이 끼어든다.

```python
covars2d = covars2d + torch.eye(2) * eps2d      # eps2d 기본값 0.3
det = det.clamp(min=1e-10)
```

이유가 둘이다.

1. **특이행렬 방지.** 카메라를 정면으로 향한 얇은 판(disk) 같은 Gaussian은 $\Sigma'$가 거의 rank-1이 되어 $\det\to 0$, 역행렬이 발산한다.
2. **안티에일리어싱.** $\Sigma' + 0.3 I$는 표준편차 $\sqrt{0.3}\approx 0.55$ 픽셀의 저역통과 필터를 씌운 것과 같다. 픽셀보다 작은 Gaussian이 샘플 격자와 간섭해 반짝이는(aliasing) 현상을 막는, 픽셀 크기의 최소 footprint를 강제한다.

이 팽창은 실제로 불투명도를 희석시킨다. 그래서 `calc_compensations=True`(즉 `rasterize_mode="antialiased"`)면

$$\texttt{compensations} = \sqrt{\frac{\det\Sigma'_{\text{orig}}}{\det(\Sigma'+\epsilon I)}}\ \le 1$$

를 함께 돌려주고, `rendering.py`가 `opacities = opacities * compensations`로 곱해 "번진 만큼 진하게" 보정한다(Mip-Splatting의 2D 보정항).

---

## 7. `radii`와 컬링 — 두 개의 역할

```python
radius_x = ceil(3.33 * sqrt(covars2d[..., 0, 0]))
radius_y = ceil(3.33 * sqrt(covars2d[..., 1, 1]))
```

$\sqrt{\sigma_{11}}, \sqrt{\sigma_{22}}$는 x/y축 방향 화면상 표준편차다. 계수 **3.33**은 임의의 수가 아니다.

$$\exp\!\left(-\tfrac12\cdot 3.33^2\right) = e^{-5.544} \approx 0.0039 \approx \frac{1}{256}$$

즉 3.33σ를 넘으면 기여가 8비트 색 1LSB 아래로 떨어져 **보이지 않는다**. 그 지점에서 자르는 것이다. (x/y 각각 따로 두므로 `radii`는 스칼라가 아니라 `[..., 2]`이고, 축 정렬 바운딩박스를 준다.)

`radii`는 두 가지로 쓰인다.

**(a) 유효성 마스크.** 세 조건 중 하나라도 걸리면 `radii = 0`으로 덮어쓴다.

```python
valid  = (depths > near_plane) & (depths < far_plane)   # 절두체 앞/뒤
radius[~valid] = 0
inside = (means2d[...,0] + r_x > 0) & (means2d[...,0] - r_x < width) & ...  # 화면 밖
radius[~inside] = 0
```

추가로 `radius_clip` 인자보다 작은 반경도 버린다(너무 작아 기여 없는 것). docstring이 명시한다.

> The output `radii` could serve as an indicator, in which zero radii means the corresponding elements are invalid in the output tensors and will be ignored in the next rasterization process.

**중요한 함의**: `packed=False`일 때 반환 텐서들은 `[C, N, ...]` 전체 크기이지만 **원소 대부분이 쓰레기값**이다. `radii > 0`으로 걸러야 한다. asset이 정확히 그렇게 센다.

```python
print(f"화면에 보이는 Gaussian: {(info['radii'] > 0).all(-1).sum().item():,} / {len(splats['means']):,}")
```

`.all(-1)`인 이유는 `radii`가 x/y 두 성분이기 때문이다.

**(b) 타일 개수.** 3단계 `isect_tiles`가 `means2d ± radii`를 타일 크기로 나눠 바운딩박스를 만들고, 걸치는 타일 수를 센다.

```python
tile_mins = floor(means2d/tile - radii/tile);  tile_maxs = ceil(...)
tiles_per_gauss = (tile_maxs - tile_mins).prod(-1)
tiles_per_gauss *= (radii > 0.0).all(-1)     # 컬링된 것은 0개 타일
```

여기서 `radii`가 과대하면 쓸데없는 타일 교차가 폭증해 느려지고, 과소하면 Gaussian 꼬리가 잘려 시각적 이음선이 생긴다. 그래서 최신 gsplat은 `opacities`를 선택 인자로 받아 **더 타이트한 반경**을 계산한다(불투명도가 낮으면 1/256 임계에 더 빨리 도달하므로 3.33σ보다 좁게 잘라도 된다). `isect_tiles`에 `conics`와 `opacities`를 넘기는 것도 축 정렬 박스 대신 실제 타원으로 교차를 판정하기 위한 최적화다.

---

## 8. `packed=True`일 때

컬링 후 살아남는 Gaussian이 전체의 몇 %뿐인 장면(넓은 씬, 많은 카메라)에서는 `[C, N, ...]` 밀집 텐서가 메모리 낭비다. `packed=True`는 유효 원소만 `[nnz, ...]`로 압축하고, COO 희소 포맷처럼 `batch_ids`, `camera_ids`, `gaussian_ids`, `indptr`를 함께 준다. 이때는 모든 원소가 유효하므로 `radii>0` 필터가 필요 없다.

단, asset의 `rasterize_splats()`는 `packed=False`를 명시한다.

```python
packed=False,   # 밀도화 상태 갱신 코드와 맞춤
```

`DefaultStrategy`가 `[C, N]` 밀집 레이아웃을 가정하고 Gaussian별 gradient 통계를 누적하기 때문이다.

---

## 9. 한 줄 정리

`fully_fused_projection`은 $\Sigma = RSS^\top R^\top$ → $R_{cw}\Sigma R_{cw}^\top$ → $J\Sigma_c J^\top$ 를 커널 하나로 처리해, **다음 두 단계가 원하는 형태로 미리 가공한** 값을 내놓는다.

- `isect_tiles`가 원하는 것: `means2d`, `radii`(바운딩박스), `depths`(정렬 키)
- `rasterize_to_pixels`가 원하는 것: `means2d`(Δ 계산), `conics`(역행렬 미리 계산), (선택) `compensations`
- 학습 루프가 원하는 것: `means2d`의 gradient (밀도화 신호), `radii > 0` (가시 Gaussian 카운트)

## 참고 위치

- `gsplat/cuda/_wrapper.py:819` — `fully_fused_projection()` 시그니처/docstring
- `gsplat/cuda/_torch_impl.py:262` — `_fully_fused_projection()` PyTorch 참조 구현
- `gsplat/cuda/_torch_impl.py:53` — `_persp_proj()` (야코비안 $J$)
- `gsplat/cuda/_torch_impl.py:225` — `_world_to_cam()`
- `gsplat/rendering.py:850` — `rasterization()`이 호출하는 지점
- `gsplat/cuda/_torch_impl.py:786` — `conics`가 알파로 소비되는 지점
