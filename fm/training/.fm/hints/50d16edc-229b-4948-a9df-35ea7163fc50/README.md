# `rasterization()`의 1단계: SH 평가 (Spherical Harmonics evaluation)

> **Q.** `rasterization()`의 1단계 SH 평가는 무엇을 하는가?
>
> **A.** `spherical_harmonics`가 카메라→Gaussian 시선 방향으로 SH 계수를 평가해 뷰 의존적 RGB를 얻는다. `sh_degree` 인자로 활성 차수를 제한할 수 있다.

---

## 1. 파이프라인 속 위치

`gsplat/rendering.py`의 `rasterization()`이 미분 가능한 렌더러 전체이고, 내부는 4개의 CUDA 커널 단계로 구성된다 (워크스루 3단계 설명).

| 단계 | 커널/함수 | 하는 일 | 산출물 |
|---|---|---|---|
| **1. SH 평가** | `spherical_harmonics` | 시선 방향으로 SH 계수 → RGB | `colors` `[..., C, N, 3]` |
| 2. 투영 | `fully_fused_projection` | $\Sigma = RSS^\top R^\top$를 EWA splatting으로 2D 투영 | `means2d`, `conics`, `depths`, `radii` |
| 3. 타일 교차 | `isect_tiles` | 16×16 타일별 (tile_id, depth) 키 정렬 | `isect_ids`, `flatten_ids` |
| 4. 픽셀 래스터화 | `rasterize_to_pixels` | 앞→뒤 알파 블렌딩 | `render`, `alpha` |

1단계가 담당하는 것은 **오직 "이 Gaussian은 이 카메라에서 무슨 색으로 보이는가"** 하나다. 위치·모양·불투명도는 전혀 건드리지 않는다. SH 평가는 Gaussian 하나당(그리고 카메라 하나당) 완전히 독립적인 계산이라, `N × C × 3`개 스레드로 임베러싱하게 병렬화된다.

### 왜 SH가 필요한가

3DGS의 Gaussian이 색을 **상수 RGB 하나**로 들고 있으면, 어느 각도에서 봐도 같은 색이다. 그러면 표현할 수 없는 것들이 있다.

- 유리·금속의 **정반사(specular) 하이라이트** — 카메라가 움직이면 하이라이트도 움직인다
- 잎사귀·물의 **광택**, 젖은 도로의 반사
- 뷰마다 미세하게 다른 노출/화이트밸런스가 만드는 색 편차

SH는 이 "방향에 따라 달라지는 색"을 **저주파 함수로 압축 표현**하는 장치다. 방향 $\mathbf{d}$(단위벡터)의 함수 $c(\mathbf{d})$를 구면조화함수 기저로 전개한다.

$$c(\mathbf{d}) \;=\; \sum_{l=0}^{L}\sum_{m=-l}^{l} c_{lm}\, Y_{lm}(\mathbf{d})$$

- $Y_{lm}$: 구면 위에 정의된 **직교 기저함수** (실수형 SH). 구면에서의 푸리에 급수라고 보면 된다.
- $c_{lm}$: 학습되는 계수. Gaussian마다, RGB 채널마다 하나씩.
- 차수 $L$까지 쓰면 계수 개수 $K = (L+1)^2$.

| `sh_degree` $L$ | $K=(L+1)^2$ | Gaussian당 계수(RGB) | 표현력 |
|---|---|---|---|
| 0 | 1 | 3 | 상수색 (뷰 무관, DC만) |
| 1 | 4 | 12 | 아주 느슨한 방향 변화 |
| 2 | 9 | 27 | 부드러운 광택 |
| 3 | 16 | **48** | 3DGS 표준 |

워크스루의 `SH_DEGREE = 3`이 바로 이것이며, 주석대로 "계수 $(3+1)^2 = 16$개"다. 파라미터는 `sh0` `[N,1,3]`(DC, $l=0$)과 `shN` `[N,15,3]`($l\ge 1$)로 **쪼개서** 저장한다 — 학습률이 다르기 때문이다(아래 4절).

---

## 2. "카메라→Gaussian 시선 방향"이 정확히 어떻게 계산되는가

이게 카드에서 가장 자주 틀리는 지점이다. SH 평가에 넘어가는 방향은 **픽셀→Gaussian 방향이 아니고, 카메라 광학중심에서 Gaussian 중심을 향하는 방향**이다. Gaussian 하나에 방향 하나(카메라당)만 쓴다 — 픽셀별로 다시 계산하지 않는다.

`spherical_harmonics()`는 `means`(world 좌표)와 `viewmats`(world→camera)를 받아서 내부에서 카메라 위치를 복원한다. `gsplat/cuda/_wrapper.py:436`의 docstring:

> The camera position is recovered from each view matrix as `-R^T t`.

view matrix가 $\begin{bmatrix} R & \mathbf{t} \\ 0 & 1\end{bmatrix}$ (world→cam)이면, 카메라 중심의 world 좌표는

$$\mathbf{p}_{\text{cam}} = -R^\top \mathbf{t}$$

이고 시선 방향은

$$\mathbf{d}_i = \frac{\boldsymbol{\mu}_i - \mathbf{p}_{\text{cam}}}{\lVert \boldsymbol{\mu}_i - \mathbf{p}_{\text{cam}}\rVert} = \frac{\boldsymbol{\mu}_i + R^\top\mathbf{t}}{\lVert\cdot\rVert}$$

CUDA 커널(`csrc/SphericalHarmonics.cuh`)이 실제로 하는 것이 정확히 이 식인데, 부호를 미리 접어 두어 덧셈 한 번으로 끝낸다.

```cpp
// camera_offset_from_world_to_camera(): R^T t 를 계산
vec3 camera_offset(
    viewmat[0]*tx + viewmat[4]*ty + viewmat[8]*tz,   // R^T의 1행 · t
    viewmat[1]*tx + viewmat[5]*ty + viewmat[9]*tz,
    viewmat[2]*tx + viewmat[6]*ty + viewmat[10]*tz);

// view_direction_from_world_to_camera(): mean + R^T t  ==  mean - (-R^T t)
return vec3(mean[0], mean[1], mean[2]) + camera_offset_from_world_to_camera(viewmat);
```

주의할 점 세 가지.

1. **정규화는 커널 안에서**, 그리고 **degree ≥ 1일 때만** 한다 (`sh_coeffs_to_color_fast`에서 `rsqrtf`로). degree 0은 기저가 상수라 방향이 필요 없으므로 정규화도 생략한다.
2. $-R^\top\mathbf{t}$는 **$R$이 직교(orthonormal)라는 가정**에 의존한다. docstring이 경고하듯 view matrix에 scale/shear가 섞이거나, 4×4를 raw로 최적화해서 직교성이 깨지면 결과가 근사값이 된다.
3. rolling shutter를 쓰면(`viewmats_rs`) 두 끝점 카메라 위치의 **평균**을 쓴다.

워크스루가 `viewmats=torch.linalg.inv(camtoworlds)`를 넘기는 이유도 이것이다 — 데이터셋의 `camtoworld`를 뒤집어 world→cam을 만들어 줘야 커널이 카메라 중심을 제대로 복원한다.

---

## 3. `sh_degree` 인자의 두 가지 역할

`rasterization()`의 docstring (`gsplat/rendering.py:418`):

> `sh_degree`: The SH degree to use, which can be smaller than the total number of bands. If set, the `colors` should be `[N, K, D]` SH coefficients (shared across batch/camera dims), else the `colors` should be `[..., (C,) N, D]` post-activation color values. Default is None.

즉 이 인자는 **스위치 겸 다이얼**이다.

### (a) 스위치: `None`이면 SH 평가를 아예 건너뛴다

`_maybe_evaluate_sh()` (함수 이름의 *maybe*가 이걸 뜻한다):

```python
if sh_degree is None:
    # colors가 이미 활성화된 RGB → shape만 [..., C, N, D]로 브로드캐스트
    ...
else:
    masks = (radii > 0).all(dim=-1)              # [..., C, N]
    features = spherical_harmonics(sh_degree, means, viewmats, features, masks=masks)
    if clamp:
        features = torch.clamp_min(features + 0.5, 0.0)   # Inria CUDA 백엔드와 동일하게
    else:
        features = features + 0.5
```

C++ 오케스트레이터(`rasterization_3dgs`)로는 `sh_degree_value = sh_degree if sh_degree is not None else -1`로 넘어가고, `-1`이 "SH 없음" 신호다. 그래서 SH 없이 색을 직접 최적화하거나(RGB 파라미터화), 렌더러를 feature splatting(임의의 $D$채널)으로 쓰는 것도 같은 함수로 가능하다.

### (b) 다이얼: 저장된 밴드보다 **작은** 차수만 평가

계수는 항상 $K=16$개를 들고 있으면서 `sh_degree=1`로 부르면, $l\ge2$ 기저는 계산하지 않고 앞 4개만 합산한다. 순수 함수 관점에서는 "고차항을 0으로 취급"과 같지만 **연산량도 실제로 줄어든다**(`degree >= 2` 블록을 아예 실행하지 않음). 검증 조건은 C++에서 걸린다.

```cpp
TORCH_CHECK((sh_degree + 1) * (sh_degree + 1) <= colors_tensor.size(-2),
            "sh_degree requires more color SH coefficients than provided");
```

Python 참조 구현도 동일하다: `assert (degrees_to_use + 1) ** 2 <= coeffs.shape[-2]`.

### `+ 0.5` 오프셋은 무엇인가

SH 기저의 $l=0$ 값은 $Y_{00} = \tfrac{1}{2\sqrt{\pi}} \approx 0.2820947917738781$이다 (커널 첫 줄의 매직넘버가 바로 이 값). 3DGS 관례는 SH 출력을 **0 중심**으로 두고 마지막에 $0.5$를 더해 $[0,1]$ RGB로 옮긴다. 그래서 초기화도 역함수 꼴이다 (워크스루 `init_splats_with_optimizers`):

```python
C0 = 0.28209479177387814
colors[:, 0, :] = (rgbs - 0.5) / C0      # DC만 SfM 색으로, 고차항은 0
```

검산: DC 계수 $c_{00} = (\text{rgb}-0.5)/C_0$ 를 평가하면 $C_0 \cdot c_{00} + 0.5 = \text{rgb}$. 초기 렌더가 정확히 SfM 포인트 색으로 나오는 이유다. `clamp=True`일 때 `clamp_min(·, 0)`으로 음수를 자르는 것은 Inria 원본 CUDA 백엔드와 수치를 맞추기 위한 것이다(상한 클램프는 없다 — 1을 넘는 값은 그대로 통과해 손실이 눌러 준다).

---

## 4. 학습에서 SH가 다뤄지는 방식

### 차수 워밍업 스케줄

워크스루 5단계:

```python
sh_degree_to_use = min(step // 1000, SH_DEGREE)   # 0 → 1 → 2 → 3
```

처음엔 DC만 학습해 **색부터 안정화**하고, 1000 step마다 밴드를 하나 열어 준다. 이유는 최적화 관점에서 명확하다. 초기에는 Gaussian의 위치·모양이 엉망이라 재구성 오차가 크고, 고차 SH는 **그 오차를 "방향 의존 색"으로 흡수해 버릴** 자유도가 있다. 즉 기하로 풀어야 할 문제를 색으로 위조(overfit)한다. 저차부터 여는 coarse-to-fine이 이 함정을 막는다.

이 스케줄 덕분에 `sh_degree`가 **매 스텝 바뀌는 인자**라는 점도 기억할 만하다 — 계수 텐서는 그대로 `[N,16,3]`이고 인자만 오르내린다.

### 학습률

```python
"sh0": 2.5e-3,
"shN": 2.5e-3 / 20,     # 고차 SH는 천천히
```

고차항 lr을 20배 낮추는 것도 같은 철학이다. 고차 SH는 표현력이 크고 뷰별로 자유롭게 움직일 수 있어, 빠르게 학습시키면 뷰 보간 품질(novel view)이 나빠진다.

### 워크스루가 `sh_degree=0`으로 첫 렌더를 하는 이유

3단계에서 초기 상태를 그려 볼 때 `rasterize_splats(..., sh_degree=0)`을 쓴다. 고차항은 어차피 0으로 초기화되어 있으니 결과는 같고, "초기 상태 렌더 (SfM 색만)"라는 의도를 코드로 못 박은 셈이다.

---

## 5. gsplat 구현 디테일 (문서 순서 ≠ 실행 순서)

카드/워크스루는 SH를 "1단계"로 소개하지만, **실제 실행 순서는 투영 뒤**다. `_rasterization()`의 순서:

```
fully_fused_projection  →  radii  →  _maybe_evaluate_sh(masks = radii > 0)  →  isect_tiles  →  rasterize_to_pixels
```

C++ 오케스트레이터(`csrc/Rendering.cpp`)도 동일하다: projection(≈L892–1000) → feature assembly/SH(≈L1056–1180) → `intersect_tile`(≈L1309) → rasterize. 이유는 최적화다. 투영이 near/far·화면 밖 Gaussian을 `radii=0`으로 컬링해 주므로, **보이지 않는 Gaussian의 SH는 계산하지 않는다**. `masks` 인자가 그 역할이고, 커널은 `if (masks != nullptr && !masks[output_id]) return;`으로 즉시 빠져나온다. 마스킹된 항목의 색은 쓰이지 않는다(타일에 걸치지 않으므로).

논리적 데이터 흐름(색을 먼저 정해야 블렌딩할 수 있다)에서는 SH가 앞이라 "1단계"라는 설명이 맞고, 커널 스케줄링에서는 뒤다 — 둘 다 알아 두면 코드를 읽을 때 헷갈리지 않는다.

그 밖의 구현 포인트:

- **`spherical_harmonics_l0` / `spherical_harmonics_l1_plus` 분리**: DC만 쓸 때는 `sh0 → 0.2820948 * sh0` 하나로 끝나므로 방향 계산조차 필요 없다. `l1_plus`는 `shN`이 **degree-1 기저에서 시작**한다(DC 계수를 의도적으로 뺀 배열) — `sh0`/`shN`을 나눠 저장하는 파라미터 레이아웃과 그대로 맞는다.
- **K=16, D=3 고속 경로**: `spherical_harmonics_fwd_kernel_k16_3channel`이 `DEGREE`를 템플릿 파라미터로 받아 계수 배열 크기를 컴파일 타임에 확정하고, `uint4`/`ushort4` wide load로 48개 계수를 한 스레드가 몰아 읽는다. 3DGS 표준 설정이 이 경로를 탄다.
- **`assemble_proj_features`**: SH 색 + depth 채널 + extra signal을 하나의 버퍼로 조립하는 fused 커널. `needs_dirs = (has_color && sh_degree > 0) || ...` 로 방향 계산 필요 여부를 먼저 판정한다 — `sh_degree == 0`이면 방향을 아예 만들지 않는다.
- **역전파**: `spherical_harmonics_bwd_kernel`이 `v_coeffs`(계수 gradient)와 `v_means`(위치 gradient)를 함께 낸다. **SH를 통해서도 `means`에 gradient가 흐른다** — 시선 방향이 $\boldsymbol{\mu}$의 함수이기 때문이다. 스레드당 K개 결과를 레지스터에 모아 한 번만 global memory에 쓰는 구조(atomics 회피).
- **채널 수 $D$는 3에 묶이지 않는다**: docstring대로 "can be any positive integer (e.g. 3 for RGB, 1 for scalar features)". semantic feature splatting 같은 응용이 같은 커널을 쓴다.
- **packed 모드**: `[nnz, K, D]`로 계수를 미리 gather해서 넘기고, `batch_ids`/`camera_ids`/`gaussian_ids`로 인덱싱한다.

---

## 6. 자주 나오는 오해 정리

| 오해 | 사실 |
|---|---|
| SH를 픽셀마다 평가한다 | Gaussian×카메라마다 **한 번**. 픽셀 단계(4단계)는 이미 나온 RGB를 블렌딩만 한다. |
| 방향은 Gaussian→카메라 | **카메라→Gaussian** ($\boldsymbol{\mu} - \mathbf{p}_\text{cam}$). 부호를 뒤집으면 홀수 차수 항의 부호가 전부 반대가 된다. |
| `sh_degree=3`이면 계수가 3개 | $(3+1)^2 = 16$개 (RGB 곱하면 48개 float). |
| `sh_degree`는 텐서 크기를 정한다 | 텐서는 항상 `[N,16,3]`. 인자는 **몇 밴드까지 합산할지**만 정한다. |
| SH 출력이 곧 RGB | $+0.5$ 오프셋(+ 옵션 `clamp_min`)을 지나야 $[0,1]$ RGB. |
| SH가 반사·굴절을 물리적으로 모델링한다 | 아니다. 방향의 **저주파 함수**를 fitting할 뿐이다. 날카로운 거울 반사는 $L=3$으로 표현 못 하고, 그래서 3DGS가 강한 specular에서 약하다. |
| `sh_degree=None`은 에러 | 정상 사용법. `colors`를 활성화된 RGB로 직접 넘기는 모드. |

## 7. 관련 소스

- `gsplat/rendering.py:234` — `rasterization()` 시그니처와 docstring
- `gsplat/rendering.py:693` — `_maybe_evaluate_sh()`
- `gsplat/cuda/_wrapper.py:436` — `spherical_harmonics()` (+ `:493` l0, `:508` l1_plus)
- `gsplat/cuda/_torch_impl.py:968` — `_eval_sh_bases_fast()` (Sloan, JCGT 2013), `:1052` `_spherical_harmonics()`
- `gsplat/cuda/csrc/SphericalHarmonics.cuh:40` — `camera_offset_from_world_to_camera()` ($R^\top t$)
- `gsplat/cuda/csrc/SphericalHarmonicsCUDA.cu:49` — `sh_coeffs_to_color_fast()`
- `gsplat/cuda/csrc/Rendering.cpp` — 오케스트레이터의 실제 단계 순서
- `examples/simple_trainer.py:288` `create_splats_with_optimizers`, `:649` `rasterize_splats`
