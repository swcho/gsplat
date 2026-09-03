# 축정렬 반경 `radii` — 3.33σ 사각형은 어디서 오는가

**Q.** 축정렬 반경 `radii`는 어떻게 계산되는가?

**A.** `radii = ceil(3.33 * sqrt(diag(Σ₂)))`, 즉 **3.33σ 사각 반경**이다. 상수 3.33은 CUDA 쪽 `Common.h`의 `GAUSSIAN_EXTEND = 3.33f`에 대응한다.

---

## 1. 코드에서의 위치

워크스루(`rasterization_walkthrough.py` §3, `project_manually`)의 한 줄이 정의 그 자체다.

```python
radii = torch.ceil(3.33 * cov2d.diagonal(dim1=-2, dim2=-1).sqrt())   # 3.33σ 사각 반경
```

순수 PyTorch 참조 구현(`gsplat/cuda/_torch_impl.py`, `_fully_fused_projection`)도 축을 풀어 쓴 같은 식이다.

```python
radius_x = torch.ceil(3.33 * torch.sqrt(covars2d[..., 0, 0]))
radius_y = torch.ceil(3.33 * torch.sqrt(covars2d[..., 1, 1]))
radius   = torch.stack([radius_x, radius_y], dim=-1)   # [..., C, N, 2]
```

CUDA 커널(`gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu`, `projection_ewa_3dgs_fused_fwd_kernel`)도 완전히 동일하다.

```cuda
float extend = GAUSSIAN_EXTEND;          // 3.33f
...
float radius_x = ceilf(extend * sqrtf(covar2d[0][0]));
float radius_y = ceilf(extend * sqrtf(covar2d[1][1]));
```

상수는 `gsplat/cuda/include/Common.h`에 정의되어 있고, 주석이 의미를 그대로 말해준다.

```c
#define ALPHA_THRESHOLD         (1.f / 255.f)
// GAUSSIAN_EXTEND determines where the gaussian is truncated in standard deviations.
#define GAUSSIAN_EXTEND         3.33f
```

파이썬 쪽에는 `gsplat/cuda/_constants.py`에 `ALPHA_THRESHOLD = 1.0 / 255.0`이 같은 값으로 미러링되어 있다(3.33은 `_torch_impl.py`에 리터럴로 박혀 있다).

**입력 Σ₂는 EWA 근사의 결과물**이다. 즉 Σ₂ = J Σ_c Jᵀ + eps2d·I (`eps2d=0.3 px²`의 최소 블러 포함). 따라서 `radii`는 "3D Gaussian이 화면에 만든 2D 타원의 크기"를 픽셀 단위로 잰 것이다.

---

## 2. 왜 σ의 배수로 잘라내는가

Gaussian은 수학적으로 **무한한 지지집합(support)**을 갖는다. 어떤 점에서도 밀도가 정확히 0이 되지 않으므로, 자르지 않으면 모든 Gaussian이 모든 픽셀에 기여해야 한다 — 즉 O(N × 픽셀 수). 래스터화가 성립하려면 "여기 밖은 안 봐도 된다"는 유한한 경계가 반드시 필요하다.

그 경계를 σ의 배수로 잡는 이유는 정규분포의 꼬리가 지수적으로 죽기 때문이다. 마할라노비스 거리 제곱을 q라 하면 커널 값은 exp(−q/2)이고, 반경 kσ에서 q = k²이다.

| k | exp(−k²/2) | 1/값 |
|---|---|---|
| 1σ | 0.6065 | ≈ 1.65 |
| 2σ | 0.1353 | ≈ 7.4 |
| 3σ | **0.011109** ≈ e^(−4.5) | ≈ 90 |
| 3.33σ | **0.0039091** ≈ e^(−5.544) | ≈ 256 |

### 3.33이라는 숫자의 정확한 출처

래스터화 커널은 `α < ALPHA_THRESHOLD = 1/255`인 Gaussian을 아예 건너뛴다(워크스루 §6). 불투명도가 최대(o = 1)일 때 알파가 정확히 1/255로 떨어지는 지점을 풀면:

$$\alpha = e^{-k^2/2} = \frac{1}{255}
\;\Longrightarrow\;
k = \sqrt{2\ln 255} = 3.32904\ldots$$

- `ln 255 = 5.541264`, `2 × = 11.082527`, `sqrt = 3.3290430`
- 여기서 반올림한 값이 **3.33**이다.
- 검산: exp(−3.33²/2) = exp(−5.54445) = **0.0039091**, 한편 1/255 = **0.0039216**. 3.33σ에서의 밀도가 임계값보다 **약간 더 작으므로**, 3.33은 정확한 해보다 아주 조금 넉넉한(보수적인) 쪽이다 — 잘라도 절대 보이는 픽셀을 잃지 않는다.

즉 **3.33σ는 "8비트 색 정밀도에서 더 이상 보이지 않게 되는 거리"**다. 3σ(≈1/90)는 8비트로 여전히 2~3 계조를 만들어 잘리는 경계가 눈에 띌 수 있고, 4σ 이상은 잘려도 어차피 커널이 버리는 영역을 쓸데없이 타일링한다. 3.33σ는 `ALPHA_THRESHOLD`와 정확히 짝을 이루는 값이다.

> 이 일관성이 중요한 이유: 만약 반경이 임계값보다 작으면 **보여야 할 기여가 잘려** 사각형 테두리가 보이고, 반대로 너무 크면 **커널이 어차피 버릴 픽셀·타일을 정렬하고 순회**하느라 비용만 든다. 3.33은 두 상수를 한 지점에 묶는다.

---

## 3. 왜 원(최대 고유값)이 아니라 축정렬 사각형(대각 성분)인가

### 원본 3DGS(Inria)는 원을 썼다

`graphdeco-inria/diff-gaussian-rasterization`의 `cuda_rasterizer/forward.cu`는 이렇게 계산한다(원문 그대로):

```cuda
float mid = 0.5f * (cov.x + cov.z);
float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
```

- 2×2 공분산의 **최대 고유값** λ_max를 구해 `3 * sqrt(λ_max)`라는 **스칼라 반경 하나**를 만든다 → 타원을 감싸는 **원**.
- 상수도 3.0(≈1/90)이라 gsplat의 3.33보다 짧다.

### gsplat은 축별 사각형을 쓴다

gsplat은 x, y **각각** `3.33 * sqrt(Σ₂[0,0])`, `3.33 * sqrt(Σ₂[1,1])`을 쓴다. 이유는 세 가지다.

**(a) 뒤따르는 소비자가 AABB만 요구한다.**
`radii`가 실제로 쓰이는 곳은 (i) 타일 교차, (ii) 화면 밖 컬링인데, 둘 다 **축정렬 경계상자(AABB)** 판정이다. 타일은 16×16 축정렬 격자이고, 이미지 경계도 축정렬 사각형이다. 원의 반경으로부터 AABB를 만들려면 결국 정사각형 [±r, ±r]로 바꿔야 하는데, 이는 타원을 감싸는 원을 다시 감싸는 정사각형 — **두 번 헐거워진다**. 애초에 축별로 재면 한 번도 헐거워지지 않는다.

**(b) 축정렬 사각형이 3.33σ 타원의 정확한 최소 AABB다.**
이것이 핵심이고, 우연이 아니다. 타원 {x : xᵀΣ₂⁻¹x ≤ t}에서 x 좌표의 최댓값은

$$\max_{x^\top \Sigma_2^{-1} x \le t} x_1 = \sqrt{t\,\Sigma_2[0,0]}$$

이다. t = 3.33²을 넣으면 정확히 `3.33 * sqrt(Σ₂[0,0])`. 즉 대각 성분으로 만든 이 사각형은 **loose bound가 아니라 tight(최소) AABB**다. 코드 주석이 "compute tight rectangular bounding box"라고 쓴 이유가 이것이다.

이 사실은 `IntersectTile.cu`의 AccuTile 경로(SNUGBOX)가 conic으로 같은 값을 다시 유도하는 데서도 확인된다. `disc = B² − AC = −det(Σ₂⁻¹) = −1/det(Σ₂)`, `C = Σ₂⁻¹[1,1] = Σ₂[0,0]/det(Σ₂)`이므로

```cuda
float x_extent = sqrtf(-t / disc * C);   // = sqrt(t * det(Σ₂) * Σ₂[0,0]/det(Σ₂)) = sqrt(t * Σ₂[0,0])
```

정확히 같은 식으로 환원된다.

**(c) 길쭉한(anisotropic) Gaussian에서 차이가 크다.**
PSD 행렬에서는 항상 λ_max ≥ max(Σ₂[0,0], Σ₂[1,1])이다. 따라서 원 반경 `sqrt(λ_max)`는 축별 반경보다 **결코 작을 수 없고**, 얇고 긴 splat(바닥·벽·나뭇가지처럼 실제 씬에 아주 흔한 형태)에서는 몇 배까지 커진다. 예를 들어 σ = (30 px, 2 px) 축정렬 splat이면 사각형은 100×7 px, 원은 100×100 px 상당 — 타일 수가 십수 배 늘어난다. `tiles_per_gauss`는 그대로 정렬/블렌딩 비용이므로 이 차이가 곧 성능이다.

> 참고 — 두 방식을 섞은 경로도 있다. 언센티드 변환(UT) 투영(`ProjectionUT3DGSFused.cu`, `_torch_impl_ut.py`)은
> `radius_x = ceil(min(extend*sqrt(Σ₂[0,0]), extend*sqrt(λ_max)))`처럼 축별 값과 고유값 원 반경의 **min**을 쓴다.
> 수학적으로는 λ_max ≥ 대각이므로 min은 항상 축별 값을 고르며, 고유값 항은 비정상 공분산(음수 대각 등)에 대한
> **수치 안전장치**에 가깝다. 결론적으로 gsplat 전 경로가 실질적으로 축별 사각형을 쓴다.

---

## 4. `diag(Σ₂)`가 정확히 무엇인가 — marginal 분산

$$\Sigma_2 = \begin{bmatrix}\sigma_x^2 & \sigma_{xy}\\ \sigma_{xy} & \sigma_y^2\end{bmatrix}$$

에서 `diag(Σ₂) = (σ_x², σ_y²)`이고, 이것은 2D Gaussian을 **x축으로 사영(적분)했을 때의 1D marginal 분산**이다. 즉

- x의 주변분포는 N(μ_x, Σ₂[0,0]) — 상관계수 σ_xy와 무관하게 성립.
- 그래서 `sqrt(Σ₂[0,0])`은 "x 방향으로 이 splat이 얼마나 퍼져 있는가"의 자연스러운 척도이고, 앞서 본 대로 이것이 곧 타원의 x 방향 최대 반경이다.

**비대각 성분 σ_xy는 `radii`에 전혀 들어가지 않는다.** 회전한 타원의 기울기 정보는 `radii`가 아니라 `conics`(Σ₂⁻¹의 (a,b,c))가 들고 가고, 실제 알파 계산에서 쓰인다:

$$\sigma_i = \tfrac12(a\,dx^2 + c\,dy^2) + b\,dx\,dy,\qquad \alpha_i=\min(0.99,\ o_i e^{-\sigma_i})$$

역할 분담이 명확하다 — **`radii`는 "어디를 볼지"(보수적·축정렬·정수), `conics`는 "얼마나 진하게 칠할지"(정확·연속·미분가능)**. 그래서 코드 주석도 `radii` 계산을 "non differentiable"이라고 명시한다. 반경은 gradient가 흐르지 않는 순수 스케줄링용 양이다.

---

## 5. `ceil`과 정수 픽셀

`ceil`은 두 가지를 한다.

1. **정수화**: 반경이 픽셀 인덱스 산술(타일 인덱스, 화면 경계 비교)에 쓰이므로 정수여야 한다. `radii`의 dtype은 `int32`다.
2. **올림(내림이 아님)**: 잘라내는 방향이 **바깥쪽**이어야 안전하다. `floor`였다면 3.33σ 경계 바로 안쪽 픽셀이 커버되지 않아 눈에 띄는 절단이 생길 수 있다. 올림은 최대 1px 여유를 더할 뿐이다.

부수 효과로 `ceil` 때문에 0이 나오려면 Σ₂ 대각이 정확히 0이어야 하는데, `eps2d = 0.3 px²` 블러가 항상 더해지므로 실제로는 `3.33 * sqrt(0.3) ≈ 1.82 → ceil = 2`가 하한이다. 즉 **자연 발생적으로 `radii = 0`이 되는 일은 없고, 0은 오직 명시적 컬링의 표식**이다(§7).

---

## 6. 왜 `[C, N, 2]`인가 — x/y를 따로 두는 이유

`radii`의 모양은 `[C, N, 2]`(현재 CUDA 시그니처는 배치를 포함해 `[B, C, N, 2]`, packed 모드에서는 `[nnz, 2]`)이다.

- **마지막 축 2 = (radius_x, radius_y)**. §3에서 본 대로 사각형이 축별로 다르기 때문에 스칼라 하나로는 표현할 수 없다. Inria 구현이었다면 `[C, N]`로 충분했을 것이다. 이 shape 자체가 "gsplat은 원이 아니라 사각형을 쓴다"는 사실의 직접적인 증거다.
- **C 축(카메라별)**: 같은 Gaussian이라도 카메라마다 깊이 z와 Jacobian J가 다르므로 Σ₂가 다르고, 따라서 반경도 다르다. 가까운 카메라에서는 크고 먼 카메라에서는 작다. 컬링 여부도 카메라마다 독립이다.

워크스루의 `meta` 표에도 이렇게 적혀 있다: `radii | [C,N,2] | 화면에서의 x/y 반경(px, int). 0이면 컬링됨(절두체 밖, 너무 작음, 너무 투명)`.

---

## 7. `radii`가 이후 단계에서 쓰이는 곳

### (a) 컬링 플래그 (`radii = 0`)

투영 커널은 아래 조건에서 반경을 0으로 덮어써서 "이 (카메라, Gaussian) 쌍은 없는 셈 친다"고 표시한다. 별도의 mask 텐서를 만들지 않고 `radii` 하나로 신호를 겸한다.

| 조건 | 코드 |
|---|---|
| 절두체 밖 (z ≤ near 또는 z ≥ far) | `_torch_impl.py`: `radius[~valid] = 0.0` |
| 너무 작음 (`radius_clip`) | `if (radius_x <= radius_clip && radius_y <= radius_clip) radii[..] = 0` |
| 너무 투명 (`opacity < ALPHA_THRESHOLD`) | 투영 커널의 opacity 분기에서 조기 return |
| 화면 밖 (사각형이 이미지와 안 겹침) | `mean2d.x + radius_x <= 0 \|\| mean2d.x - radius_x >= image_width \|\| ...` |

주의할 점: **컬링된 항목의 `means2d`/`conics`/`depths`는 커널이 early-return해서 쓰지 않으므로 쓰레기값(`torch.empty` 잔여)이다.** 워크스루 §8이 명시적으로 `valid = (m["radii"] > 0).all(dim=-1)` 마스크로 걸러 비교하는 이유가 이것이다. 그리고 화면 밖 판정 자체가 `radii` 사각형으로 이뤄지므로, **반경이 컬링의 입력이자 컬링의 출력**이다.

`packed=True` 모드에서는 아예 `radii > 0`인 쌍만 `[nnz, ...]`로 압축해서 반환한다.

### (b) 타일 교차 AABB (`isect_tiles`)

기본(AABB) 경로는 반경 사각형을 타일 단위로 나눠 걸치는 타일 범위를 구한다. PyTorch 참조(`_torch_impl.py::_isect_tiles`):

```python
tile_radii = radii / tile_size
tile_mins  = torch.floor(tile_means2d - tile_radii).int()
tile_maxs  = torch.ceil(tile_means2d + tile_radii).int()
...
tiles_per_gauss *= (radii > 0.0).all(dim=-1)      # 컬링된 것은 0 타일
```

CUDA(`IntersectTile.cu`)의 AABB fallback도 동일하다(`floor(tile_x - tile_radius_x)` … `ceil(tile_x + tile_radius_x)`를 [0, tile_width]로 clamp). 커널 맨 앞에서 `if (radius_x <= 0 || radius_y <= 0) { tiles_per_gauss[idx] = 0; return; }`로 컬링을 재확인한다.

여기서 `radii`의 사각형성이 왜 중요한지 다시 보인다 — **타일 격자가 축정렬이라 사각형 반경이 그대로 타일 인덱스 범위로 나눗셈 한 번에 변환된다.** 원 반경이었다면 같은 산술을 쓰되 훨씬 넓은 범위를 쓸어담게 된다.

### (c) AccuTile (더 조인 경로)

`rasterization()`은 `conics`와 `opacities`도 `isect_tiles`에 넘긴다. 그러면 사각형이 아니라 **알파 1/255 등고선 타원**과 실제로 겹치는 타일만 고른다(`accutile_process_tiles`). 결과 이미지는 같고(버려지는 타일의 기여는 어차피 임계값 아래) 정렬·블렌딩 비용만 준다. 이때도 등고선 레벨은 같은 상수쌍으로 정의된다:

```cuda
float t = fminf(GAUSSIAN_EXTEND * GAUSSIAN_EXTEND, 2.0f * __logf(opacity / ALPHA_THRESHOLD));
```

`t`가 `GAUSSIAN_EXTEND²`로 캡되므로 **AccuTile이 `radii` 사각형 밖으로 나가는 일은 없다** — 반경이 상한, AccuTile이 그 안에서의 정밀화다.

---

## 8. 불투명도 인지 반경 (opacity-aware radius)

`opacities`를 투영 커널에 넘기면 상수 3.33 대신 **Gaussian마다 다른 extend**를 쓴다.

```cuda
extend = min(GAUSSIAN_EXTEND, sqrt(2.0f * __logf(opacity / ALPHA_THRESHOLD)));
```

유도는 §2와 같지만 봉우리 높이가 o < 1이라는 점만 다르다:

$$\alpha = o\,e^{-q/2} \ge \frac{1}{255}
\;\Longleftrightarrow\;
q \le 2\ln\!\left(\frac{o}{1/255}\right)
\;\Longrightarrow\;
k_{\text{eff}} = \min\!\left(3.33,\ \sqrt{2\ln(255\,o)}\right)$$

직관: **흐릿한 Gaussian은 더 일찍 보이지 않게 된다.** 최고점이 이미 낮으니 1/255로 떨어지는 지점이 중심에 더 가깝다.

| opacity o | k_eff | 사각형 넓이 비 (k_eff/3.33)² |
|---|---|---|
| 1.00 | 3.329 | 1.00 |
| 0.90 | 3.297 | 0.98 |
| 0.60 | 3.172 | 0.91 |
| 0.30 | 2.945 | 0.78 |
| 0.10 | 2.545 | 0.58 |
| 0.05 | 2.256 | **0.46** |
| < 1/255 | — | 아예 컬링(`radii = 0`) |

핵심 포인트:

- 3.33은 **o = 1일 때의 값**이자 상한이다. `min(...)`이 있으므로 불투명도 인지 반경은 절대 3.33σ보다 커지지 않는다.
- 학습 중 자연스럽게 생기는 저불투명도 Gaussian(잘려나가기 직전 것들)에서 타일 수가 절반 가까이 줄어든다.
- antialiased 모드에서는 보정계수를 먼저 곱한(`opacity *= compensation`) **유효 불투명도**로 계산한다.
- `o < ALPHA_THRESHOLD`면 반경을 재볼 것도 없이 `radii = 0`으로 컬링한다.
- 워크스루 §3 마지막 셀이 이걸 직접 검증한다: `ext = torch.sqrt(2 * torch.log(opac / ALPHA_THRESHOLD)).clamp(max=3.33)`를 CUDA가 낸 `radii`와 비교.

주의 — `rasterization()`은 내부적으로 `opacities`를 넘기므로 **실사용 경로의 `radii`는 사실 3.33σ가 아니라 opacity-aware 반경**이다. 워크스루의 `project_manually` / `_fully_fused_projection` 기본 호출은 `opacities`가 없어 순수 3.33σ를 낸다. 두 값을 비교할 때 이 차이를 잊으면 "왜 안 맞지?"가 된다.

---

## 9. 한 줄 요약 / 암기 포인트

1. `radii = ceil(3.33 * sqrt(diag(Σ₂)))` — **대각 성분(축별 marginal 표준편차) × 3.33, 올림, int32**.
2. **3.33 = sqrt(2 ln 255)** — 알파가 `ALPHA_THRESHOLD = 1/255`로 떨어지는 σ 배수. `Common.h`의 `GAUSSIAN_EXTEND`.
3. **원이 아니라 사각형**인 이유: 소비자(타일 격자, 이미지 경계)가 AABB이고, 대각 성분 사각형이 그 타원의 **정확한 최소 AABB**이기 때문. 원본 Inria는 `3.0 * sqrt(λ_max)` 원이라 길쭉한 splat에서 훨씬 헐겁다.
4. **`[C,N,2]`의 2**는 x/y 반경이 서로 다르다는 뜻, **C**는 카메라마다 Σ₂가 다르다는 뜻.
5. **`radii = 0` = 컬링 표식**(절두체 밖 / 너무 작음 / 너무 투명 / 화면 밖). 이 항목의 다른 meta 값은 쓰레기값.
6. `opacities`를 넘기면 `extend = min(3.33, sqrt(2 ln(255·o)))`로 **줄어든다** (커지지는 않는다).

## 관련 파일

- `fm/rasterization/.fm/assets/rasterization_walkthrough.py` §3(투영), §5(타일 교차), §6(블렌딩·임계값)
- `gsplat/cuda/include/Common.h` — `GAUSSIAN_EXTEND`, `ALPHA_THRESHOLD`, `MAX_ALPHA`, `TRANSMITTANCE_THRESHOLD`
- `gsplat/cuda/_constants.py` — 파이썬 쪽 미러
- `gsplat/cuda/_torch_impl.py` — `_fully_fused_projection`, `_isect_tiles`
- `gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu` — `extend` / `radius_x` / `radius_y` / 컬링
- `gsplat/cuda/csrc/IntersectTile.cu` — AABB 경로와 AccuTile(SNUGBOX) 경로
- `gsplat/cuda/csrc/ProjectionUT3DGSFused.cu`, `gsplat/cuda/_torch_impl_ut.py` — 고유값 항을 곁들인 UT 변형
