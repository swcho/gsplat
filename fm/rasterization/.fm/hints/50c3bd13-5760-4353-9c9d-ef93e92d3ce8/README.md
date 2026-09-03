# `rasterization_2dgs` — 볼륨이 아니라 "표면"을 스플랫하는 버전

> **Q.** `rasterization_2dgs`는 3DGS와 무엇이 다른가?
>
> **A.** 표면 지향 2D Gaussian(디스크) 버전으로, 광선-평면 교차로 σ를 구한다.
> 표면 재구성에 유리한 방식이다.

워크스루 마지막 절 "여기서 더 볼 것들"의 마지막 항목이다. `rasterize_mode="antialiased"`,
`with_ut`, `with_eval3d` 같은 다른 항목들이 **3DGS 파이프라인의 옵션**인 것과 달리,
2DGS는 **프리미티브 자체가 다른 별도의 파이프라인**이다. 그래서 함수도 `rasterization()`의
인자가 아니라 `rasterization_2dgs()`라는 별개 진입점이고, 투영·래스터화 커널도 별도 파일이다
(`Projection2DGS*.cu`, `RasterizeToPixels2DGS*.cu`).

원 논문: Huang et al., *"2D Gaussian Splatting for Geometrically Accurate Radiance Fields"*,
SIGGRAPH 2024.

---

## 1. 프리미티브: 3D 타원체 → 2D 타원 디스크

3DGS의 프리미티브는 3D 공분산 $\Sigma = RSS^\top R^\top$ 를 가진 **부피를 가진 타원체**다.
`scales`가 3개 축 모두 살아 있다.

2DGS는 **세 번째 축을 0으로 죽인 납작한 원반(surfel, disk)** 이다. 커널 코드에서 그대로 보인다.

```cpp
// Projection2DGSFused.cu — scales[2]가 아예 쓰이지 않는다
mat3 RS_camera
    = R * quat_to_rotmat(glm::make_vec4(quats))
        * mat3(scales[0], 0.0, 0.0,
               0.0, scales[1], 0.0,
               0.0, 0.0, 1.0);        // ← 세 번째 열은 스케일 1, 그리고 아래에서 버려진다

mat3 WH = mat3(RS_camera[0], RS_camera[1], mean_c);   // 1열, 2열 + 중심만 사용
```

즉 API가 `scales: [..., N, 3]`을 받는 것은 3DGS와 시그니처를 맞추기 위한 것이고,
**실제로 쓰이는 것은 `scales[..., :2]` 뿐**이다. (Inria 래퍼 쪽에서는 아예
`scales = scales[:, :2]`로 잘라서 넘긴다.)

이 구조에서 회전 행렬 $R$의 세 열이 곧 디스크의 로컬 프레임이 된다.

| 열 | 의미 |
|---|---|
| $t_u = R_{:,0}$ | 디스크 접평면의 첫 번째 축 (길이 $s_u$) |
| $t_v = R_{:,1}$ | 디스크 접평면의 두 번째 축 (길이 $s_v$) |
| $t_w = R_{:,2}$ | **법선** — 접평면에 수직 |

디스크 위의 점은 로컬 좌표 $(u, v)$로 이렇게 매개변수화된다.

$$P(u,v) = p_k + s_u t_u\, u + s_v t_v\, v$$

핵심은 이것이다. **3D 타원체에는 "이 프리미티브의 법선"이라는 개념이 애초에 없다.**
축이 세 개 다 살아 있으므로 어느 축을 법선이라 부를지 정해지지 않고, 얇게 눌린 타원체라
해도 그건 학습이 우연히 만들어 준 결과일 뿐 구조적 보장이 아니다. 2DGS는 세 번째 축을
제거함으로써 **접평면과 법선이 정의상 항상 명확**하다. 커널은 이 법선을 그냥 꺼내 쓴다.

```cpp
vec3 normal = RS_camera[2];                              // = R의 3열 (카메라 공간)
float multipler = glm::dot(-normal, mean_c) > 0 ? 1 : -1; // 카메라 반대편이면 뒤집기
normal *= multipler;                                      // "dual visible": 양면 렌더
```

`multipler`가 하는 일이 재밌다. 디스크는 앞뒤 구분이 없으므로 **항상 카메라를 향하도록
법선을 뒤집어** 준다(dual-visible). 덕분에 학습 중에 디스크가 뒤집혀도 법선 손실이 깨지지 않는다.

## 2. 왜 표면 재구성에 유리한가 — 멀티뷰 불일치 문제

3DGS로 학습한 씬에서 깊이/메시를 뽑아 보면 표면이 뭉개지고 층이 지는데, 원인은 렌더링
품질이 아니라 **프리미티브의 기하가 뷰마다 다른 답을 주기 때문**이다.

```
      3D 타원체(볼륨)                        2D 디스크(표면)

  카메라 A ──────►  ╱▔▔╲                 카메라 A ──────►  ────────
                  │ ●  │  ← A가 보는                        ●   ← A, B가 보는
  카메라 B ──────►  ╲__╱     "가장 진한 점"   카메라 B ──────►      교차점이 동일
                   ↑                                        ↑
             B가 보는 "가장 진한 점"                  평면과 광선의 교차점은
             — 두 점이 다르다                          뷰와 무관하게 유일
```

- **3DGS**: 픽셀 광선이 타원체를 관통하며 3차원 밀도장을 적분한다. 알파가 최대가 되는
  지점(사실상 그 뷰에서의 "표면 위치")은 광선 방향에 따라 달라진다. 게다가 3DGS 실제
  구현은 광선-타원체 적분조차 하지 않고 **화면에서 2D Gaussian으로 근사**해 버리기 때문에
  (EWA), 깊이는 그냥 중심의 $z$ 값으로 퉁쳐진다. 결국 **같은 표면 점을 뷰마다 다른
  깊이로 보고**, 멀티뷰 깊이를 합치면 노이즈가 된다.
- **2DGS**: 프리미티브가 평면이므로 광선과의 교차점이 **정확히 한 점**이고, 그 점은
  광선 방향에 의존하는 근사가 아니다. 어느 뷰에서 보아도 같은 3D 점을 가리킨다.
  법선도 프리미티브에 붙어 있는 고정된 양이라 뷰 간에 일관된다.

여기에 얇은 구조(잎, 종이, 벽, 천)에도 유리하다. 3DGS는 얇은 판을 표현하려면 타원체를
극단적으로 눌러야 하는데 그러면 수치적으로 불안정해지고, 2DGS는 그게 기본형이다.

## 3. 원근 정확(perspective-accurate) 스플래팅 — 논문의 핵심

3DGS의 EWA 스플래팅은 투영을 **아핀 근사**한다. Gaussian 중심에서 투영 함수의 Jacobian $J$를
한 번 계산해 $\Sigma_2 = J W \Sigma W^\top J^\top$ 로 2D 공분산을 만들고, 그 다음부터는
화면 위의 2D Gaussian으로만 논다. 이 근사는 중심에서 멀어질수록, 화각이 넓을수록 틀어진다.
(워크스루 3절의 `project_manually`가 정확히 이 과정이다.)

2DGS는 근사를 아예 하지 않는다. **픽셀 광선과 디스크 평면의 교차점을 닫힌 형태로 정확히 푼다.**

### 3-1. 투영 단계가 만드는 것: `ray_transforms` (M 행렬)

3DGS 투영이 `conics`($\Sigma_2^{-1}$)를 내놓는 자리에서, 2DGS 투영은
**"디스크 로컬 $(u,v,1)$ → 픽셀 동차 좌표"** 로 보내는 $3\times3$ 행렬을 내놓는다.

```cpp
mat3 WH = mat3(RS_camera[0], RS_camera[1], mean_c);   // q_cam = WH · [u, v, 1]
mat3 world_2_pix = mat3(Ks[0], 0.0, Ks[2],
                        0.0, Ks[4], Ks[5],
                        0.0,  0.0,  1.0);             // glm 열우선이라 사실상 K^T
mat3 M = glm::transpose(WH) * world_2_pix;            // M = (KWH)^T
```

`M`의 세 행 $M_u, M_v, M_w$ (코드의 `M0/M1/M2`)이 `ray_transforms` [.., N, 3, 3]로 저장되어
래스터화 커널로 전달된다. 3DGS의 `conics`가 이 자리를 차지하던 것과 정확히 대응된다.

$3\times3$ 로 충분한 이유는 1절에서 본 대로 $RS$의 세 번째 열이 0이라서, 논문의 $4\times4$
동차 변환 $H$가 $[R t_u s_u,\; R t_v s_v,\; p_{cam}]$ 3열로 접히기 때문이다.

투영 단계는 이 밖에 화면 AABB(`radii`), 2D 중심(`means2d`), 깊이(`depths = mean_c.z`),
카메라 공간 법선(`normals`)을 낸다. `means2d`는 σ 계산에는 쓰이지 않고 **타일 교차와
저역통과 필터, 밀도화 기준**에만 쓰인다.

### 3-2. 래스터화 단계: 두 동차 평면의 교차

픽셀 $(p_x, p_y)$에 대해, 커널은 두 개의 **동차 평면(homogeneous plane)** 을 만든다.
커널 주석이 유도를 그대로 담고 있다.

```cpp
// h_u, h_v are the homogeneous plane representations
const vec3 h_u = px * w_M - u_M;      // h_u = p_x·M_w − M_u
const vec3 h_v = py * w_M - v_M;      // h_v = p_y·M_w − M_v

const vec3 ray_cross = glm::cross(h_u, h_v);          // 두 평면의 교선
if (ray_cross.z == 0.0) continue;                     // 광선이 평면과 평행 → 스킵
const vec2 s = vec2(ray_cross.x / ray_cross.z,        // projective flattening
                    ray_cross.y / ray_cross.z);       // → 디스크 로컬 좌표 (u, v)
```

읽는 법:

1. **$h_u$의 의미.** 임의의 $q_{uv} = [u,v,1]$에 대해
   $h_u \cdot q_{uv} = p_x (M_w q_{uv}) - (M_u q_{uv})$ 이다.
   $M q_{uv}$는 그 디스크 점의 픽셀 동차 좌표 $[xz,\,yz,\,z]$ 이므로, 이 식이 0이라는 것은
   곧 **"그 점의 화면 x좌표가 $p_x$이다"** 라는 뜻이다. 즉 $h_u$는 픽셀 열 $p_x$에 대응하는
   평면을, $h_v$는 픽셀 행 $p_y$에 대응하는 평면을 정의한다. 이 두 평면의 교선이 바로
   **픽셀 $(p_x,p_y)$를 지나는 시선(ray)** 이다. (논문 Eq. 9~11.)
2. **외적.** 두 조건을 동시에 만족하는 유일한 해(직교 조건)는 $\zeta = h_u \times h_v$.
3. **투영적 평탄화.** $uv$ 공간도 결국 광선 공간이라 스케일이 무의미하므로
   $(u,v) = (\zeta_1/\zeta_3,\ \zeta_2/\zeta_3)$ 로 나눠서 실제 디스크 로컬 좌표를 얻는다.

여기에 나눗셈이 들어간다는 점이 중요하다. **원근 나눗셈을 근사로 치우지 않고 그대로
수행하기 때문에** "perspective-accurate"라고 부른다.

### 3-3. σ 계산 — 3DGS와 나란히 놓고 보기

디스크 로컬 좌표는 이미 스케일이 정규화되어 있으므로 (스케일 $s_u, s_v$가 $H$ 안에
들어가 있다), Gaussian 응답은 등방 표준정규 그대로다.

$$\mathcal{G}(u,v) = \exp\!\left(-\frac{u^2+v^2}{2}\right),
\qquad \sigma = \tfrac{1}{2}\,(u^2 + v^2)$$

| | 3DGS | 2DGS |
|---|---|---|
| 투영 산출물 | `conics` = $\Sigma_2^{-1}$의 (a,b,c) | `ray_transforms` = $M=(KWH)^\top$ |
| 픽셀당 입력 | $dx, dy$ = 픽셀중심 − `means2d` | $(u,v)$ = 광선∩디스크 (로컬 좌표) |
| σ | $\tfrac12(a\,dx^2 + c\,dy^2) + b\,dx\,dy$ | $\tfrac12(u^2 + v^2)$ |
| 성격 | 화면 위 2D 마할라노비스 거리 (아핀 근사) | 표면 위 정확한 거리 |
| 깊이 | 중심의 $z$ (splat 전체에 상수) | 교차점 기반, median depth 가능 |

이후는 동일하다. $\alpha = \min(0.99,\ o\, e^{-\sigma})$, 앞→뒤 알파 블렌딩,
$T \mathrel{*}= (1-\alpha)$.

## 4. 저역통과 필터 — 디스크가 옆에서 보이면 사라지는 문제

정확한 교차를 하면 새 문제가 생긴다. 디스크를 **정확히 옆에서(edge-on)** 보면 화면 위
투영이 선분(폭 0)이 되어, 픽셀 중심이 그 위에 정확히 얹히지 않는 한 $(u,v)$가 폭발하고
$\alpha \to 0$이 된다. 즉 **한 픽셀보다 얇아진 디스크가 그냥 증발**한다. 학습 중이면
gradient도 같이 사라져 복구가 안 된다.

논문의 해법은 화면 공간에서 최소 크기($\sigma = \sqrt2/2$)를 갖는 2D Gaussian을 하나
더 두고 **둘 중 큰 응답을 쓰는 것**이다.

$$\hat{\mathcal{G}} = \max\Big(\ \mathcal{G}(u,v),\ \ \mathcal{G}\big(\tfrac{x - c}{\sigma}\big)\ \Big)$$

gsplat 구현은 지수부(σ)에서 `min`을 취해 같은 일을 한다 — $e^{-\sigma}$는 감소함수이므로
**σ의 min = G의 max**다.

```cpp
const float gauss_weight_3d = s.x * s.x + s.y * s.y;      // 광선-교차 커널
const vec2  d = {xy_opac.x - px, xy_opac.y - py};         // means2d − 픽셀
// #define FILTER_INV_SQUARE_2DGS 2.0f
const float gauss_weight_2d = FILTER_INV_SQUARE_2DGS * (d.x*d.x + d.y*d.y);

const float gauss_weight = min(gauss_weight_3d, gauss_weight_2d);   // = max(G_3d, G_2d)
const float sigma = 0.5f * gauss_weight;
float alpha = min(MAX_ALPHA, opac * __expf(-sigma));
```

상수 `2.0f`를 풀어 보면 $\sigma_{2d} = |d|^2$, 즉 $G_{2d} = \exp(-|d|^2)
= \exp\!\big(-|d|^2 / (2\cdot(\tfrac{\sqrt2}{2})^2)\big)$ — 논문의 $\sigma=\sqrt2/2$와 정확히 일치한다.
`means2d`가 σ 계산 자체에는 안 쓰인다고 했지만 **이 필터 항에서만은 쓰인다**는 점을 기억하자.

3DGS의 `eps2d` 팽창(2D 공분산 대각에 0.3 더하기)과 목적은 같지만, 3DGS는 공분산을 부풀리고
2DGS는 두 커널의 max를 취한다는 점이 다르다. 2DGS 쪽 방식이 **화면상 최소 크기를 보장하면서도
크게 보일 때는 정확한 커널을 그대로 쓴다**는 장점이 있다.

## 5. 추가 출력 — 여기가 "표면 재구성용"인 이유

`rasterization()`이 `(colors, alphas, meta)` 3개를 돌려주는 데 비해 `rasterization_2dgs()`는
**7개**를 돌려준다.

```python
(render_colors,     # [..., C, H, W, X]  — 3DGS와 동일
 render_alphas,     # [..., C, H, W, 1]  — 3DGS와 동일
 render_normals,    # [..., C, H, W, 3]  ← 알파 블렌딩된 프리미티브 법선 (월드 공간)
 surf_normals,      # [..., C, H, W, 3]  ← 렌더된 깊이맵에서 유한차분으로 뽑은 법선
 render_distort,    # [..., C, H, W, 1]  ← 픽셀별 깊이 왜곡(distortion)
 render_median,     # [..., C, H, W, 1]  ← 중간 깊이 (median depth)
 meta) = rasterization_2dgs(...)
```

### 5-1. `render_normals` — 프리미티브 법선의 알파 블렌딩

색과 완전히 같은 방식으로 누적한다. `vis = alpha * T`.

```cpp
const float *n_ptr = normals + g * 3;
for (uint32_t k = 0; k < 3; ++k) normal_out[k] += n_ptr[k] * vis;
```

커널은 카메라 공간에서 계산하고, `Rendering.cpp`가 마지막에 월드로 회전시킨다.

```cpp
at::Tensor camtoworlds_rotation = at::linalg_inv(viewmats).narrow(-2,0,3).narrow(-1,0,3);
render_normals = at::einsum("...ij,...hwj->...hwi", {camtoworlds_rotation, render_normals});
```

### 5-2. `surf_normals` — 깊이맵에서 유도한 법선

같은 렌더에서 나온 **깊이맵을 3D 점군으로 되돌린 뒤 이웃 픽셀 차분의 외적**으로 법선을 만든다
(`gsplat/utils.py::depth_to_normal`).

```python
points = depth_to_points(depths, camtoworlds, Ks, z_depth=z_depth)
dx = points[..., 2:, 1:-1, :] - points[..., :-2, 1:-1, :]
dy = points[..., 1:-1, 2:, :] - points[..., 1:-1, :-2, :]
normals = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
```

이 두 개(`render_normals` vs `surf_normals`)가 **따로 나오는 것이 핵심**이다. 하나는
프리미티브가 주장하는 법선, 다른 하나는 실제로 렌더된 깊이가 만드는 법선. 둘이 어긋난다는
것은 디스크의 방향이 실제 표면과 어긋나 있다는 뜻이다 → 손실로 만들 수 있다(6절).

### 5-3. `render_distort` — 깊이 왜곡

Mip-NeRF 360의 distortion loss를 알파 블렌딩 루프 안에서 온라인으로 계산한다.
"광선을 따라 가중치가 한 곳에 몰려 있게" 강제하는 항이다.

```cpp
const float depth = c_ptr[CDIM - 1];              // 마지막 채널이 깊이
const float distort_bi_0 = vis * depth * (1.0f - T);   // w_i · d_i · Σ_{j<i} w_j
const float distort_bi_1 = vis * accum_vis_depth;      // w_i · Σ_{j<i} w_j d_j
distort += 2.0f * (distort_bi_0 - distort_bi_1);
accum_vis_depth += vis * depth;
```

docstring이 밝히듯 **논문의 L2 버전이 아니라 L1 버전**이다
("L1 version, different from L2 version in 2DGS paper"). 깊이 채널이 필요하므로
`distloss=True`는 깊이가 포함된 `render_mode`를 요구한다.

```cpp
TORCH_CHECK(!distloss || append_depth, "distloss requires a depth render mode");
```

### 5-4. `render_median` — 중간 깊이

투과율 $T$가 0.5를 지나기 직전의 Gaussian 깊이. 즉 "이 광선의 에너지 절반이 소진되는 지점".

```cpp
if (T > 0.5) { median_depth = c_ptr[CDIM - 1]; median_idx = batch_start + t; }
```

기대 깊이(expected depth, 가중 평균)와 달리 **여러 표면에 걸쳐 평균내지 않으므로 경계에서
번지지 않는다**. `depth_mode="median" | "expected"`로 고르고, 이 선택은
`surf_normals`를 계산할 때 어느 깊이를 쓸지도 바꾼다.

```cpp
at::Tensor depth_for_normal = depth_mode_is_median
    ? render_median
    : render_colors.narrow(-1, render_colors.size(-1) - 1, 1);
render_normals_from_depth = depth_to_normal_2dgs(depth_for_normal, camtoworlds, Ks).squeeze(0);
```

Inria 래퍼의 주석이 실무 기준을 정확히 말해 준다.
**경계가 있는(bounded) 씬이면 median, 무한(unbounded) 씬이면 expected** — 후자는 disk aliasing을 줄인다.

## 6. 정규화 손실이 학습 파이프라인에서 하는 일

추가 출력만으로는 기하가 좋아지지 않는다. **손실에 넣어야** 효과가 난다.
`examples/simple_trainer_2dgs.py`가 그 참조 구현이다.

```python
# 법선 일관성(normal consistency): 두 법선의 코사인 정렬
normals_from_depth *= alphas.squeeze(0).detach()
normal_error = (1 - (normals * normals_from_depth).sum(dim=0))[None]
loss += curr_normal_lambda * normal_error.mean()

# 왜곡(distortion): 광선을 따라 가중치를 한 점에 모은다
loss += render_distort.mean() * curr_dist_lambda
```

기본 하이퍼파라미터와 **warm-up 스케줄**이 특징적이다.

| 옵션 | 기본값 | 시작 iteration |
|---|---|---|
| `--normal_loss` (`normal_lambda`) | `5e-2` | `normal_start_iter = 7_000` |
| `--dist_loss` (`dist_lambda`) | `1e-2` | `dist_start_iter = 3_000` |

시작 전에는 람다를 그냥 0으로 둔다.

```python
curr_normal_lambda = cfg.normal_lambda if step > cfg.normal_start_iter else 0.0
```

이유는 직관적이다. 초반에는 디스크들이 아직 아무 데나 떠 있어서, 법선 일관성이나 왜곡을
일찍 강하게 걸면 **아직 존재하지도 않는 표면에 맞추려다 최적화가 나쁜 국소해에 갇힌다.**
색(photometric) 손실로 대략적인 배치가 잡힌 뒤에 기하를 조이는 순서다.

또 `normals_from_depth`에 `alphas.detach()`를 곱하는 것도 눈여겨볼 것 — 알파가 낮은
(= 표면이 확실하지 않은) 픽셀의 기여를 자동으로 줄인다.

## 7. gsplat API 차이 요약

| | `rasterization` (3DGS) | `rasterization_2dgs` |
|---|---|---|
| 반환값 | `(colors, alphas, meta)` | `(colors, alphas, normals, surf_normals, distort, median, meta)` |
| `scales` | 3축 전부 사용 | `[..., :2]`만 사용 (3번째는 무시) |
| 전용 인자 | `rasterize_mode`, `with_ut`, `with_eval3d`, `covars` | `distloss`, `depth_mode` |
| `meta` 특이 키 | `conics` | `ray_transforms`, `normals`, `render_distort`, `gradient_2dgs` |
| `radii` | `[..., C, N, 2]` | `[..., C, N, 2]` (동일 — 축정렬 AABB) |
| 미분 가능성 | — | **`Ks`(내부 파라미터)에 대해 미분 불가** (docstring 경고) |

`meta["gradient_2dgs"]`는 밀도화(densification) 기준으로 쓰는 별도 gradient 누적 텐서다
(`_wrapper.py`의 `densify` 인자 — "Dummy variable to keep track of gradient for densification").
2DGS는 splat 크기 정의가 3DGS와 달라서 밀도화 기준도 따로 관리한다.

`packed=True`, `sparse_grad`, `absgrad`, 타일 교차/오프셋 인코딩은 3DGS와 **완전히 공유**된다
(`isect_tiles`, `isect_offset_encode`는 같은 커널). 즉 파이프라인의 골격은 워크스루에서 본
그대로고, ②③(투영)과 ⑦(블렌딩)만 2DGS 전용 커널로 갈아 끼운 형태다.

## 8. `rasterization_2dgs_inria_wrapper`는 왜 같이 있는가

같은 파일에 원저자(hbb1)의 CUDA 백엔드 `diff-surfel-rasterization`을 감싼 래퍼가 함께 있다.

```python
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
assert eps2d == 0.3, "This is hard-coded in CUDA to be 0.3"
```

용도는 세 가지다.

1. **레퍼런스 대조.** gsplat의 재구현이 원본과 같은 결과를 내는지 벤치마크로 확인한다.
   품질 회귀가 났을 때 "우리 구현 버그인가, 알고리즘 특성인가"를 가른다.
2. **논문 수치 재현.** 원본 구현으로 보고된 숫자를 그대로 재현해야 할 때.
3. **마이그레이션 경로.** 기존 2DGS 코드를 gsplat 인터페이스로 옮기는 중간 단계.

인터페이스가 gsplat 쪽으로 맞춰져 있음을 코드에서 볼 수 있다 — 쿼터니언 정규화를 대신 해 주고
(`F.normalize`, 원본은 내부 정규화를 하지 않는다), `viewmats`/`Ks`를 Inria식
`world_view_transform` / `full_proj_transform` / `FoV`로 변환해 주고, 출력 `allmap`의
채널들을 이름 있는 텐서로 풀어 준다.

```python
render_depth_expected = allmap[..., 0:1]
render_alphas         = allmap[..., 1:2]
render_normal         = allmap[..., 2:5]
render_depth_median   = allmap[..., 5:6]
render_dist           = allmap[..., 6:7]
```

단점도 분명하다. 별도 패키지 설치가 필요하고, `eps2d`가 하드코딩되어 있으며, 카메라 루프를
Python에서 돌기 때문에(`for cid in range(C)`) 배치 처리가 안 된다. 그래서 실사용은
`rasterization_2dgs` 쪽이다.

## 9. 트레이드오프 — 언제 2DGS를 쓰는가

**2DGS가 유리한 경우**

- 메시/포인트클라우드 추출, TSDF 융합, 표면 정확도가 목표일 때
- 얇은 구조(잎사귀, 종이, 벽, 천) — 3DGS는 타원체를 극단적으로 눌러야 해서 불안정
- 뷰 간 깊이 일관성이 필요한 다운스트림(재조명, 물리 시뮬, 충돌 판정)
- 넓은 화각/왜곡 렌즈 — EWA 아핀 근사가 약한 영역인데 2DGS는 근사 자체를 안 한다

**3DGS가 유리한 경우**

- 순수 외관 품질(PSNR/SSIM/LPIPS). 2DGS는 **보통 3DGS보다 약간 낮다.**
  표현력을 1 자유도 줄인 대가이고, 볼륨감이 있는 것(연기, 털, 반투명 재질)이나 뷰 의존 효과를
  볼륨으로 "속여서" 맞추던 여지가 사라진다.
- 기하가 필요 없고 새 시점 합성만 하면 될 때

바꿔 말해 2DGS는 **외관 품질을 조금 내주고 기하 정확도를 사는 교환**이다.
정규화 손실(6절)까지 켜면 이 교환은 더 뚜렷해진다 — 손실이 색 맞추기를 방해하는 방향으로
작용하기 때문이다. 그래서 warm-up 스케줄과 작은 람다가 기본값인 것이다.

## 10. 코드 대응표

| 단계 | Python 래퍼 | CUDA 소스 | 3DGS 대응 |
|---|---|---|---|
| 투영 | `fully_fused_projection_2dgs` | `Projection2DGSFused.cu` / `Projection2DGSPacked.cu` | `fully_fused_projection` |
| 투영 VJP 보조 | — | `Projection2DGS.cuh` `compute_ray_transforms_aabb_vjp` | — |
| 타일 교차 | `isect_tiles` / `isect_offset_encode` | `IntersectTile.cu` | **동일 (공유)** |
| 블렌딩 | `rasterize_to_pixels_2dgs` | `RasterizeToPixels2DGSSerialBatch{Fwd,Bwd}.cu` | `rasterize_to_pixels` |
| 인덱스 추출 | `rasterize_to_indices_in_range_2dgs` | `RasterizeToIndices2DGSSerialBatch.cu` | `rasterize_to_indices_in_range` |
| 전체 | `rendering.rasterization_2dgs()` | `Rendering.cpp` `rasterization_2dgs()` | `rasterization()` |
| 학습 예제 | `examples/simple_trainer_2dgs.py` | — | `examples/simple_trainer.py` |
| 뷰어 | `examples/simple_viewer_2dgs.py`, `gsplat_viewer_2dgs.py` | — | `examples/simple_viewer.py` |

## 한 줄 정리

3DGS가 **부피 있는 타원체를 화면에서 2D Gaussian으로 근사해** $\sigma$를 conic으로 재는 반면,
2DGS는 **법선이 정의된 납작한 디스크를 두고 픽셀 광선과 그 평면의 교차점을 두 동차 평면의
외적으로 정확히 풀어** $\sigma = \tfrac12(u^2+v^2)$ 를 잰다.
그 결과 뷰에 일관된 깊이와 법선이 공짜로 나오고, 여기에 법선 일관성·왜곡 손실을 얹어
표면 재구성 품질을 얻는다 — 외관 품질을 조금 내주는 대가로.
