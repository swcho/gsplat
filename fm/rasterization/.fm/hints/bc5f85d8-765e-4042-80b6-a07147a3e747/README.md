# `with_eval3d=True` — 2D 근사를 걷어내고 광선 위에서 3D Gaussian을 직접 평가하기

> **Q.** `with_eval3d=True` 옵션은 무엇이 다른가?
> **A.** 2D 근사 없이 픽셀 광선과 3D Gaussian의 응답을 직접 평가한다. `RasterizeToPixelsFromWorld3DGS*.cu` 커널을 사용한다.

---

## 0. 한 줄 요약

기본 3DGS는 **투영 단계에서 3D Gaussian을 2D 타원으로 근사**하고 **래스터화 단계에서 그 2D 타원을 픽셀 중심에서 평가**한다.
`with_eval3d=True`는 **두 번째 근사를 없앤다**. 픽셀마다 카메라 광선 $\mathbf{o}+t\mathbf{d}$를 만들고, 월드 공간의 3D Gaussian이 그 광선 위에서 갖는 **최대 응답**을 닫힌 형태로 계산해 그 값을 $\alpha$로 쓴다.

중요한 건 **전부 바꾸는 게 아니라는 점**이다. 정렬·타일 교차는 여전히 2D 투영 결과(`means2d`/`radii`/`depths`)에 의존하고, 알파 블렌딩 루프도 그대로다. 바뀌는 건 오직 "이 픽셀에서 이 Gaussian의 $\alpha$는 얼마인가"를 계산하는 **한 줄**이다. 그래서 eval3d는 **하이브리드**다.

---

## 1. 기본 3DGS 래스터화의 "두 겹 근사"

워크스루([`.fm/assets/rasterization_walkthrough.py`](../../assets/rasterization_walkthrough.py))의 파이프라인 그림을 다시 보자.

```
③ 원근 투영(EWA)  Σ₂ = J Σ_c Jᵀ + eps2d·I   →  means2d, conics(=Σ₂⁻¹), radii
⑦ 알파 블렌딩     σ = ½(a·dx² + c·dy²) + b·dx·dy,  α = min(0.99, o·e^{−σ})
```

### 근사 ①: 3D Gaussian → 2D 타원 (EWA 투영)

원근 투영 $\pi(x,y,z) = (f_x x/z + c_x,\; f_y y/z + c_y)$는 **비선형**이다. 비선형 사상을 통과한 Gaussian은 더 이상 Gaussian이 아니다. 그래서 EWA(Zwicker et al. 2001)는 Gaussian 중심 $\mu_c$에서의 **1차 테일러 근사(Jacobian $J$)** 로 공분산을 밀어낸다.

$$J = \begin{bmatrix} f_x/z & 0 & -f_x x/z^2 \\ 0 & f_y/z & -f_y y/z^2 \end{bmatrix},
\qquad \Sigma_2 = J\,\Sigma_c\,J^\top + \varepsilon_{2d}\,I$$

이 $\Sigma_2$는 "만약 투영이 $\mu_c$ 근방에서 선형이었다면 나왔을 2D 공분산"이다. 즉 **중심에서만 정확하고, 중심에서 멀어질수록 틀린다.**

### 근사 ②: 그 2D 타원을 픽셀 중심 한 점에서 평가

래스터화 커널은 픽셀 중심 $(j{+}0.5,\,i{+}0.5)$과 `means2d`의 차 $(dx,dy)$를 conic으로 감싼 2차형식

$$\sigma_i = \tfrac12\big(a\,dx^2 + c\,dy^2\big) + b\,dx\,dy,
\qquad \alpha_i = \min(0.99,\; o_i e^{-\sigma_i})$$

를 쓴다 (워크스루 §7, `rasterize_naive`). 여기서 쓰이는 정보는 **2D 평면 위의 (a,b,c)와 (dx,dy)** 뿐이다. 원래 Gaussian이 깊이 방향으로 얼마나 길쭉했는지, 광선이 그 타원체를 어디로 뚫고 지나가는지는 이미 정보가 사라진 뒤다.

### 근사가 눈에 보이게 깨지는 곳

| 상황 | 왜 깨지나 |
|---|---|
| **화면에서 큰 / 카메라에 가까운 Gaussian** | $\mu_c$에서의 1차 근사가 Gaussian의 실제 화면 범위 전체를 못 덮는다. 타원이 실제 실루엣과 어긋난다. |
| **광각 / 화면 주변부** | $J$가 광축에서 멀어질수록 급격히 변한다. 같은 Gaussian이라도 화면 중심과 코너에서 왜곡 정도가 다르다. |
| **어안·F-theta·OpenCV 왜곡** | 투영이 $z$로 나누는 단순 형태가 아니라 다항식 왜곡을 포함한다. 하나의 $J$로 요약하는 것 자체가 무리다. |
| **롤링 셔터** | 픽셀 행마다 카메라 포즈가 다르다 → Gaussian마다 "하나의 2D 타원"이라는 개념이 성립하지 않는다. |
| **깊이 방향으로 길쭉한 Gaussian** | 2D 타원은 시선 방향 두께를 완전히 버린다. 광선이 어디를 지나든 같은 $\alpha$가 나온다. |

3DGUT 논문(Wu et al., CVPR 2025)이 지적하는 지점이 정확히 여기다. 그리고 이 두 근사는 **서로 다른 두 개의 수정**을 요구한다.

- 근사 ①을 고치는 것 → `with_ut=True` (Jacobian 대신 Unscented Transform으로 시그마 포인트를 실제 왜곡 투영에 통과시켜 2D 통계를 추정)
- 근사 ②를 고치는 것 → **`with_eval3d=True`**

그래서 3DGUT는 언제나 **`--with_ut --with_eval3d` 두 개를 같이** 켠다 ([`docs/3dgut.md`](../../../../../docs/3dgut.md)).

---

## 2. eval3d의 아이디어 — 픽셀당 광선을 만든다

`with_eval3d=True`이면 래스터화 커널은 픽셀 $(j,i)$에 대해 **월드 좌표계의 광선** $\mathbf{o} + t\mathbf{d}$를 만든다. 카메라 모델별로 언프로젝션 방법이 다르지만, 인터페이스는 하나로 통일돼 있다 — [`gsplat/cuda/include/Cameras.cuh`](../../../../../gsplat/cuda/include/Cameras.cuh)의 `element_to_world_ray_shutter_pose(j, i, rs_params)`.

```
element_to_image_point(j, i)            →  (j+0.5, i+0.5)      // 2D 경로와 같은 샘플 위치
  → image_point_to_camera_ray(img_pt)   →  카메라 좌표계 방향 (정규화)
  → interpolate_shutter_pose(t_rel)     →  롤링셔터 시각 t_rel의 포즈로 보간
  → camera_ray_to_world_ray(dir)        →  월드 광선 {ray_org, ray_dir, valid_flag}
```

핀홀의 경우 언프로젝션은 지극히 단순하다 ([`Cameras.cuh:714`](../../../../../gsplat/cuda/include/Cameras.cuh#L714)):

```cpp
uv         = (image_point - principal_point) / focal_length;
camera_ray = normalize(vec3{uv.x, uv.y, 1.f});
```

어안·F-theta·LiDAR·OpenCV 왜곡 모델은 각자의 `image_point_to_camera_ray_impl`을 제공한다. **여기가 핵심이다** — 왜곡을 "Gaussian을 어떻게 찌그러뜨릴까"의 문제가 아니라 **"이 픽셀이 바라보는 방향이 어디인가"의 문제로 바꿔 놓는다.** 후자는 카메라 모델의 언프로젝션 공식 그 자체이므로 근사가 없다.

그래서 소스에는 이런 게이트가 붙는다 ([`Rendering.cpp:275-276`](../../../../../gsplat/cuda/csrc/Rendering.cpp)):

```cpp
TORCH_CHECK(!use_hit_distance || with_eval3d, "hit-distance render modes require with_eval3d=True");
TORCH_CHECK(!return_normals   || with_eval3d, "return_normals=True requires with_eval3d=True");
TORCH_CHECK(with_eval3d, "Rays input is only supported with Eval3D");   // rays 인자
```

광선이 명시적으로 존재해야만 의미가 있는 출력들(히트 거리, 법선)과, **광선을 밖에서 직접 주입하는 `rays` 인자**가 전부 eval3d 전용이다.

---

## 3. 광선 위 최대 응답 — 닫힌 형태

3D Gaussian의 (정규화 상수를 뺀) 응답은

$$\rho(\mathbf{x}) = \exp\!\left(-\tfrac12 (\mathbf{x}-\mu)^\top \Sigma^{-1} (\mathbf{x}-\mu)\right)$$

이다. 광선을 따라 이 값을 적분하는 대신, 3DGRT(Moenne-Loccoz et al., SIGGRAPH Asia 2024)는 **광선 위 최댓값**을 응답으로 쓴다. 적분보다 싸고, 무엇보다 **닫힌 형태**로 떨어진다.

### 유도

$\mathbf{e} = \mathbf{o} - \mu$라 두고 지수부의 2차형식을 $t$의 함수로 펼치면

$$g(t) = (\mathbf{e} + t\mathbf{d})^\top \Sigma^{-1} (\mathbf{e} + t\mathbf{d})
= t^2\,\mathbf{d}^\top\Sigma^{-1}\mathbf{d} \;+\; 2t\,\mathbf{d}^\top\Sigma^{-1}\mathbf{e} \;+\; \mathbf{e}^\top\Sigma^{-1}\mathbf{e}$$

$t$에 대한 위로 볼록한 포물선이므로 $g'(t)=0$에서 유일한 최솟값(= $\rho$의 최댓값)을 갖는다:

$$\boxed{\;t^\* = -\frac{\mathbf{d}^\top\Sigma^{-1}\mathbf{e}}{\mathbf{d}^\top\Sigma^{-1}\mathbf{d}}
= \frac{\mathbf{d}^\top\Sigma^{-1}(\mu - \mathbf{o})}{\mathbf{d}^\top\Sigma^{-1}\mathbf{d}}\;}$$

$$\rho_{\max} = \exp\!\left(-\tfrac12\, g(t^\*)\right)$$

### Gaussian 로컬 좌표로 옮기면 훨씬 깔끔하다

$\Sigma = R\,S S^\top R^\top$ (S = diag(scales))이므로 $\Sigma^{-1} = M M^\top$, 단 $M = R\,S^{-1}$. 이제 변환

$$\mathbf{o}_g = M^\top(\mathbf{o} - \mu) = S^{-1}R^\top(\mathbf{o}-\mu),
\qquad \mathbf{d}_g = M^\top \mathbf{d} = S^{-1}R^\top\mathbf{d}$$

를 적용하면 $\mathbf{v}^\top\Sigma^{-1}\mathbf{v} = \lVert M^\top\mathbf{v}\rVert^2$이므로, **Gaussian은 원점 중심의 단위 구가 되고 Mahalanobis 거리는 그냥 유클리드 거리가 된다.** 문제가 "원점에서 광선까지의 거리"로 환원된다.

$$t^\* = \frac{\mathbf{d}_g^\top(\mu_g - \mathbf{o}_g)}{\lVert\mathbf{d}_g\rVert^2}
\;\;\overset{\mu_g = 0}{=}\;\; -\frac{\mathbf{d}_g^\top\mathbf{o}_g}{\lVert\mathbf{d}_g\rVert^2}$$

$\hat{\mathbf{d}}_g = \mathbf{d}_g/\lVert\mathbf{d}_g\rVert$로 정규화해 두면 $t^\* = -\hat{\mathbf{d}}_g\!\cdot\!\mathbf{o}_g$이고, 최소 제곱거리는 **피타고라스**로

$$g(t^\*) = \lVert\mathbf{o}_g\rVert^2 - (\hat{\mathbf{d}}_g\!\cdot\!\mathbf{o}_g)^2
= \lVert\, \hat{\mathbf{d}}_g \times \mathbf{o}_g \,\rVert^2$$

**외적의 노름 제곱** 한 방으로 끝난다. 나눗셈이 사라진다는 게 커널 입장에서 중요하다.

### 최종 알파

$$\alpha = \min\!\big(0.99,\; o \cdot \exp(-\tfrac12 \lVert \hat{\mathbf{d}}_g \times \mathbf{o}_g \rVert^2)\big)$$

2D 경로의 $\alpha = \min(0.99,\, o\,e^{-\sigma})$와 **형태가 완전히 같다.** $\sigma$를 만드는 방법만 바뀌었다. 그래서 뒤따르는 알파 블렌딩은 손댈 필요가 없다.

---

## 4. 커널 코드와 하나씩 대조

[`gsplat/cuda/csrc/RasterizeToPixelsFromWorld3DGS.cuh`](../../../../../gsplat/cuda/csrc/RasterizeToPixelsFromWorld3DGS.cuh)의 `process_fetch_round_blend` 안쪽 (약 690~760행):

```cpp
const vec3 gro    = iscl_rot * (ray_o[p] - xyz);      // o_g = S⁻¹Rᵀ(o − μ)
const vec3 grd    = safe_normalize(iscl_rot * ray_d[p]);  // d̂_g
const float hit_t = -glm::dot(grd, gro);              // t* (정규화 d̂_g 기준)
if (hit_t < 0.f) { continue; }                        // 최근접점이 광선 뒤 → 스킵

const vec3  gcrod    = glm::cross(grd, gro);          // d̂_g × o_g
const float grayDist = glm::dot(gcrod, gcrod);        // ‖d̂_g × o_g‖²
const float power    = -0.5f * grayDist;
float max_response   = __expf(power);                 // ρ_max
float alpha          = min(MAX_ALPHA, opac * max_response);
if (alpha < ALPHA_THRESHOLD) { continue; }

// ── 여기서부터는 2D 경로와 완전히 동일한 앞→뒤 알파 블렌딩 ──
const float next_T = T[p] * (1.0f - alpha);
if (next_T <= transmittance_threshold[p]) { done_mask |= (1u << p); continue; }
const float vis = alpha * T[p];
for (k = 0; k < CDIM; ++k) pix_out[p][k] += c_ptr[k] * vis;
T[p] = next_T;
```

$\Sigma^{-1}$의 절반 $M^\top$은 Gaussian을 shared memory로 실을 때 미리 만들어 둔다 (같은 파일 `cooperative_load_fetch_round`, 약 615행):

```cpp
mat3 R = quat_to_rotmat(quat);
mat3 S = mat3(1/scale[0], 0, 0,  0, 1/scale[1], 0,  0, 0, 1/scale[2]);
iscl_rot_batch[tid] = S * glm::transpose(R);          // = S⁻¹Rᵀ = Mᵀ
normal_batch[tid]   = R[2];                           // 법선 = R·(0,0,1) (return_normals용)
```

순수 PyTorch 참조 구현도 완전히 같다 — [`gsplat/cuda/_torch_impl_eval3d.py`](../../../../../gsplat/cuda/_torch_impl_eval3d.py)의 `_compute_ray_gaussian_distance` / `_compute_gaussian_alphas`. 커널을 못 읽겠으면 이쪽을 먼저 보면 된다.

### 코드에서만 보이는 디테일 3가지

1. **`sigma >= 0` 체크가 사라졌다.** 2D 경로는 수치적으로 망가진 conic 때문에 $\sigma<0$이 나올 수 있어 방어 코드가 있다. eval3d의 `grayDist`는 노름 제곱이라 **항상 $\ge 0$** 이다. 근사를 걷어내니 방어 코드도 하나 줄었다.
2. **`hit_t < 0` per-ray 컬링.** 최근접점이 광선 원점 뒤인 Gaussian을 **픽셀 단위로** 버린다. 2D 경로에는 투영 시점의 `near_plane` 절두체 컬링밖에 없다. 광각/180도 초과 FOV에서 이 차이가 크다.
3. **`hit_distance` = 실제 3D 히트 거리.** `grds = scale * (grd * hit_t)`로 로컬 좌표의 최근접점을 월드 스케일로 되돌린 뒤 노름을 잰다. 2D 경로의 "깊이"는 Gaussian 중심의 $z$일 뿐이지만, eval3d는 **광선이 실제로 그 Gaussian을 스치는 지점까지의 거리**를 낸다. LiDAR 시뮬레이션이 eval3d를 요구하는 이유가 이것이다.

---

## 5. 하이브리드 — 여전히 2D 투영이 필요한 곳

**이게 이 카드에서 가장 오해하기 쉬운 부분이다.** `with_eval3d=True`여도 투영 단계는 없어지지 않는다.

[`Rendering.cpp`](../../../../../gsplat/cuda/csrc/Rendering.cpp)의 흐름:

```
① 투영 (EWA 또는 UT)  →  means2d, radii, depths, conics, projected_opacities
                            │
② intersect_tile        ←──┘   ← means2d / radii / depths 로 타일 교차 + depth 정렬 키
                            │       (with_ut일 땐 conics/opacities를 안 넘겨 AABB 모드로 감)
③ isect_offsets, flatten_ids
                            │
④ rasterize_to_pixels_from_world_3dgs(means, quats, scales, opacities, viewmats, Ks, …,
                                      isect_offsets, flatten_ids)
       ↑ means2d도 conics도 넘기지 않는다. 넘기는 건 "어떤 타일에 어떤 Gaussian이 있고 어떤 순서인가"뿐.
```

| 역할 | 무엇이 담당하나 |
|---|---|
| **어떤 Gaussian이 어떤 타일에 걸리나** | 2D 투영의 `means2d`, `radii` (EWA 또는 UT) |
| **앞→뒤 정렬 순서** | 2D 투영의 `depths` (또는 `global_z_order=False`면 유클리드 거리) |
| **절두체 / near-far 컬링** | 2D 투영 |
| **픽셀별 $\alpha$와 색** | ← **여기만 3D 광선 평가로 바뀐다** |
| **알파 블렌딩·조기 종료** | 2D 경로와 동일 ($T \mathrel{*}= 1-\alpha$, `TRANSMITTANCE_THRESHOLD=1e-4`) |

즉 eval3d는 **"어디를 볼지"는 2D 투영에 맡기고, "무엇이 보이는지"만 3D로 정확히 계산**하는 구조다. 타일 교차는 보수적(over-inclusive)이면 되므로 근사여도 무방하고, 픽셀 값은 근사면 그대로 화질 손실이므로 정확해야 한다 — 계산 예산을 옳은 곳에 쓰는 분업이다.

같은 이유로 정렬은 여전히 **Gaussian 단위 전역 정렬**이다. 광선 추적처럼 픽셀별로 진짜 순서를 구하는 게 아니라, 타일 내 depth 순서를 그대로 쓴다. eval3d는 3DGRT가 아니다 — **래스터화의 뼈대를 유지한 채 응답 평가만 광선 기반으로 바꾼 것**이다.

---

## 6. 파일 이름 `RasterizeToPixelsFromWorld3DGS*.cu`의 "FromWorld"

2D 경로 커널은 `RasterizeToPixels3DGS*.cu`이고 입력이 **`means2d`, `conics`** — 이미 스크린 공간으로 넘어온 값이다.

eval3d 커널은 `RasterizeToPixels**FromWorld**3DGS*.cu`이고 입력이 **`means`, `quats`, `scales`** — **월드 좌표의 원본 Gaussian 파라미터**다. `viewmats`/`Ks`/왜곡 계수도 같이 받아서 커널 안에서 광선을 만든다. "FromWorld" = **스크린 공간 중간 표현을 거치지 않고 월드 파라미터에서 바로 픽셀을 만든다**는 뜻이다.

이 저장소의 실제 파일 구성 (`gsplat/cuda/csrc/`):

| 파일 | 역할 |
|---|---|
| `RasterizeToPixelsFromWorld3DGS.cuh` | 공유 디바이스 코드 — 광선 생성 디스패치, `cooperative_load_fetch_round`, `process_fetch_round_blend` (**응답 공식이 여기 있다**) |
| `RasterizeToPixelsFromWorld3DGS.h` | 런처 선언 |
| `...SerialBatchFwd.cu` | 직렬-배치 forward 커널 (`RendererConfig_MixedBatch`) |
| `...ParallelBatchFwd.cu` | 병렬-배치 forward 커널 (`RendererConfig_ParallelBatch`) |
| `...ParallelBatchBwd.cu` | backward 커널 |

카드의 `{Fwd,Bwd}` 표기는 이 계열 전체를 가리키는 축약이다. 실제로는 batch 전략별로 갈라져 있다.

### backward가 내놓는 것 — 여기서 제약이 파생된다

[`Rasterization.cpp:2900`](../../../../../gsplat/cuda/csrc/Rasterization.cpp)의 `RasterizeBwdResult`:

```cpp
at::Tensor v_means, v_quats, v_scales, v_colors, v_opacities, v_backgrounds;
at::optional<at::Tensor> v_rays;
```

**`v_means2d`가 없다.** 그럴 수밖에 없는 게, eval3d 커널은 `means2d`를 아예 입력으로 받지 않는다. 그래디언트는 월드 파라미터로 **직행**한다. 이게 다음 절 제약의 뿌리다.

---

## 7. 제약 — 소스에서 확인되는 것들

### `absgrad` / `DefaultStrategy` 비호환

[`gsplat/rendering.py:652`](../../../../../gsplat/rendering.py):

```python
if absgrad and not with_eval3d:
    means2d.absgrad = means2d_absgrad
```

`with_eval3d=True`면 `meta["means2d"].absgrad`가 **조용히 붙지 않는다**. `means2d.grad`도 마찬가지다. 그런데 `DefaultStrategy`(원본 3DGS 밀도화)는 densify/split 판단 기준으로 정확히 그 화면공간 그래디언트를 쓴다. 그래서:

```
[Trainer] Note: with_ut=True + with_eval3d=True (full 3DGUT mode).
          DefaultStrategy is incompatible with eval3d; use MCMCStrategy (the `mcmc` subcommand).
```
([`examples/simple_trainer.py:1611`](../../../../../examples/simple_trainer.py))

**gsplat에서 3DGUT는 MCMC 밀도화만 지원한다.** 예외가 아니라 구조적 귀결이다 — 화면공간 그래디언트라는 신호 자체가 존재하지 않는다.

### 나머지 제약 (`Rendering.cpp`)

| 제약 | 이유 |
|---|---|
| `rasterize_mode` 는 `"classic"`만 | `"antialiased"`(Mip-Splatting)의 보정 계수 $\sqrt{\det\Sigma_0/\det\Sigma_{blur}}$는 2D 공분산 개념에 붙어 있다. `_validate_3dgut_rasterize_mode` |
| `packed=False` 필수 | `TORCH_CHECK(!packed, "Packed mode is not supported with Eval3D")` |
| `sparse_grad=False` 필수 | 같은 곳 |
| `quats`/`scales` 필수 (`covars` 불가) | 커널이 $R$, $S$를 따로 필요로 한다. `"UT and Eval3D rasterization require quats and scales, not covars"` |
| `distributed=False` | `"distributed=True does not support with_eval3d=True"` |
| `tile_size ∈ {8, 16}` | 2D 경로는 `{4, 16}`. 기본값은 `_resolve_tile_size`가 결정 — **min(W,H) ≥ 1080이면 16, 아니면 8** |
| `calc_compensations` 불가 | 2D 공분산 보정 개념이 없음 |

### 자주 같이 쓰는 인자 (eval3d를 **요구**하는 것 vs `with_ut`를 요구하는 것)

| 기능 | 요구 사항 |
|---|---|
| `rays=` (외부 광선 주입) | **`with_eval3d=True`** |
| `render_mode`의 hit-distance | **`with_eval3d=True`** |
| `return_normals=True` | **`with_eval3d=True`** |
| `radial/tangential/thin_prism_coeffs` | `with_ut=True` |
| `camera_model="ftheta"` | `with_ut=True` |
| LiDAR 카메라 모델 | `with_ut=True` |
| 롤링 셔터 (`viewmats_rs`) | `with_ut=True` |
| `global_z_order=False` | `with_ut=True` |

**왜곡 파라미터는 `with_ut`가, 광선 기반 출력은 `with_eval3d`가 게이트한다.** 그래서 왜곡 카메라를 제대로 쓰려면 실질적으로 둘 다 필요하다. 전형적 호출:

```bash
python examples/simple_trainer.py mcmc --with_ut --with_eval3d --camera_model fisheye ...
```
```python
rasterization(means, quats, scales, opacities, colors, viewmats, Ks, W, H,
              with_ut=True, with_eval3d=True,          # ← 3DGUT 풀 모드
              camera_model="fisheye", radial_coeffs=..., packed=False)
```

---

## 8. 비용

**연산량.** 픽셀 × Gaussian 하나당:

| | 2D 경로 | eval3d |
|---|---|---|
| 핵심 연산 | 2D 2차형식 — 곱 5회 + 합 (~7 FLOP) | 3×3 matvec 2회 + 정규화(rsqrt) + 외적 + 내적 2회 (~40 FLOP) |
| `expf` | 1회 | 1회 |

**shared memory** — 이쪽이 실은 더 아프다.

| | Gaussian당 shared bytes |
|---|---|
| 2D (`RasterizeToPixels3DGSSerialBatchFwd.cu`) | `int32 id` + `vec3 xy_opacity` + `vec3 conic` = **28 B** |
| eval3d | `int32 id` + `vec4 xyz_opacity` + `mat3 iscl_rot` + `vec3 scale` (+ `vec3 normal`) = **68 B** (법선까지면 80 B) |

**약 2.4배**다. 타일당 shared memory가 늘면 SM에 동시에 올릴 수 있는 CTA 수가 줄어 점유율이 떨어진다. `_resolve_tile_size`의 주석이 정확히 이 트레이드오프를 설명한다 — 1080p 미만에서는 `tile=8`(CTA=32, 픽셀/스레드=2)로 타일당 shared를 줄여 CTA를 많이 띄우고, 1080p 이상에서는 intersect+sort 비용이 지배적이라 `tile=16`(CTA=256, 스레드당 1픽셀)으로 타일 수를 줄이는 게 이긴다.

또 하나: `packed=False` 강제 때문에 대규모 씬/다중 카메라에서 packed 모드의 메모리 절약을 못 쓴다.

**대가로 얻는 것**: 왜곡 카메라에서 undistort 전처리 없이 원본 이미지로 바로 학습, 큰 Gaussian의 실루엣 정확도, 실제 히트 거리와 법선.

---

## 9. 왜 이게 "secondary rays"의 기반인가

2D 경로는 **광선이라는 개념 자체를 갖고 있지 않다.** 픽셀 격자와 스크린 공간 타원만 있다. 그래서 원리적으로 primary ray(카메라에서 곧장 나가는 광선) 외에는 표현할 방법이 없다.

eval3d는 계약을 바꾼다 — 커널이 필요로 하는 건 **광선 하나 $(\mathbf{o}, \mathbf{d})$** 뿐이다. 그러니 그 광선이 어디서 왔는지는 상관이 없다. 그래서 `rays` 인자가 존재한다 ([`_wrapper.py`](../../../../../gsplat/cuda/_wrapper.py) `rasterize_to_pixels_eval3d`):

```python
rays: Optional[Tensor] = None,   # [..., C, H, W, 6] = (o_x,o_y,o_z, d_x,d_y,d_z)
```

커널 안에서는 카메라 모델을 우회하고 이 텐서를 그대로 읽는다 ([`RasterizeToPixelsFromWorld3DGS.cuh:525`](../../../../../gsplat/cuda/csrc/RasterizeToPixelsFromWorld3DGS.cuh)):

```cpp
ray.ray_org = {rays[pix_id*6+0], rays[pix_id*6+1], rays[pix_id*6+2]};
ray.ray_dir = {rays[pix_id*6+3], rays[pix_id*6+4], rays[pix_id*6+5]};
```

여기 들어오는 광선이 **반사·굴절·그림자 광선**이어도 커널은 신경 쓰지 않는다. 응답 공식 $\rho_{\max}$는 광선의 출처와 무관하게 정의되니까. 이것이 3DGRT와의 접점이다 — 3DGRT는 이 **똑같은 최대 응답 공식**을 BVH 광선 추적 위에서 쓰고, gsplat의 eval3d는 타일 래스터라이저 위에서 쓴다. **파티클 표현과 응답 정의를 공유하므로 같은 학습된 씬을 양쪽에서 렌더할 수 있다.** 3DGUT 논문이 강조하는 "rasterization으로 빠르게 primary ray, ray tracing으로 secondary ray"라는 하이브리드가 여기서 나온다.

(다만 `rays` 경로도 타일 교차는 여전히 카메라 투영에서 나온 `isect_offsets`/`flatten_ids`를 쓴다. 완전히 임의의 광선 집합을 다루려면 그 부분의 가속 구조도 같이 바꿔야 한다 — 그게 3DGRT의 BVH다.)

---

## 10. 정리

| | 기본 (`with_eval3d=False`) | `with_eval3d=True` |
|---|---|---|
| 픽셀 $\alpha$ 계산 | 2D conic 2차형식을 픽셀 중심에서 평가 | 광선 $\mathbf{o}+t\mathbf{d}$ 위 3D Gaussian 최대 응답 |
| $\sigma$ | $\tfrac12(a\,dx^2+c\,dy^2)+b\,dx\,dy$ | $\tfrac12\lVert\hat{\mathbf{d}}_g\times\mathbf{o}_g\rVert^2$ |
| 커널 입력 | `means2d`, `conics` | `means`, `quats`, `scales`, `viewmats`, `Ks`, 왜곡 계수 |
| 커널 | `RasterizeToPixels3DGS*.cu` | `RasterizeToPixelsFromWorld3DGS*.cu` |
| 타일 교차·정렬 | 2D 투영 | **2D 투영 (동일)** |
| 알파 블렌딩 | 앞→뒤 $T\mathrel{*}=1-\alpha$ | **동일** |
| backward 출력 | `v_means2d` 포함 | `v_means`/`v_quats`/`v_scales` 직행 (`v_means2d` 없음) |
| 밀도화 전략 | DefaultStrategy / MCMC | **MCMC만** |

한 문장으로: **eval3d는 3DGS의 두 근사 중 "2D 타원을 픽셀 중심에서 평가한다"는 두 번째 근사만 걷어내고, 정렬·타일링·블렌딩이라는 래스터화의 골격은 그대로 둔 하이브리드다.** `with_ut`(첫 번째 근사 수정)와 짝을 이룰 때 비로소 3DGUT가 된다.

---

## 참고

- **3DGUT** — Wu et al., *3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting*, CVPR 2025. UT 투영 + 3D Gaussian 평가의 두 축.
- **3DGRT** — Moenne-Loccoz et al., *3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes*, SIGGRAPH Asia 2024. 광선-Gaussian 최대 응답 공식의 출처.
- **EWA Splatting** — Zwicker et al., 2001. 기본 경로가 쓰는 Jacobian 기반 2D 근사.
- 저장소 문서: [`docs/3dgut.md`](../../../../../docs/3dgut.md)
- 워크스루: [`.fm/assets/rasterization_walkthrough.py`](../../assets/rasterization_walkthrough.py) — §3(EWA 투영), §7(알파 블렌딩), 마지막 "여기서 더 볼 것들"
