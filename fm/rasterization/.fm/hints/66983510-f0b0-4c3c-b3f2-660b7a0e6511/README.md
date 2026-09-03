# `with_ut=True` — Jacobian 대신 Unscented Transform으로 투영하기

> **Q.** `with_ut=True` 옵션의 의미는?
> **A.** Jacobian 1차 근사 대신 Unscented Transform으로 공분산을 투영한다. 어안·F-theta·롤링셔터 카메라를 지원하는 3DGUT 방식이다.

---

## 1. 출발점: 기본 경로가 쓰는 EWA Jacobian 근사

노트북 3장(`## 3. ②③ 카메라 변환 + 원근 투영 (EWA splatting)`)이 설명하는 기본 경로는 이렇다.

1. 카메라 좌표로: μ_c = R μ + t, Σ_c = R Σ Rᵀ
2. 원근 투영 π(x,y,z) = (f_x·x/z + c_x, f_y·y/z + c_y) 는 **비선형**이다. 비선형 함수는 Gaussian을 Gaussian으로 보내지 않는다. 그래서 μ_c에서 **1차 테일러 전개**를 해서 그 자리를 선형으로 취급한다.

$$J = \begin{bmatrix} f_x/z & 0 & -f_x x/z^2 \\ 0 & f_y/z & -f_y y/z^2 \end{bmatrix},\qquad
  \Sigma_{2D} = J\,\Sigma_c\,J^\top + \epsilon\,I$$

이게 EWA splatting의 핵심이고, `ProjectionEWA3DGSFused.cu`의 `projection_ewa_3dgs_fused_fwd_kernel`이 이 한 줄을 Gaussian×카메라당 스레드 1개로 처리한다.

### 이 근사가 깨지는 지점

Jacobian 근사는 "μ_c 근처에서 π가 거의 선형"이라는 가정 위에 서 있다. 이 가정이 무너지는 경우가 세 가지 있고, 셋 다 실무에서 자주 만난다.

| 상황 | 왜 문제인가 |
|---|---|
| **화면 가장자리 / 넓은 FOV** | z가 작거나 x/z, y/z가 크면 J의 −f·x/z² 항이 폭주한다. 노트북도 언급하듯 참조 구현 `_persp_proj`는 J를 만들기 전에 x/z, y/z를 시야각의 1.3배로 **clamp**한다 — Inria 원본의 땜질이다. 즉 기본 경로는 이미 "J가 못 버티는 영역"을 손으로 막고 있다. |
| **어안 / F-theta / 강한 렌즈 왜곡** | 투영이 r = f·θ 형태거나 6차 다항식 왜곡이 붙는다. 하나의 Gaussian이 화면에서 눈에 띄게 **휘어진** 모양으로 찍히는데, J Σ Jᵀ는 여전히 타원 하나만 만들 수 있다. 게다가 OpenCV pinhole의 undistort는 뉴턴 반복(`compute_undistortion_newton`)이라 닫힌 형태의 미분이 없다. |
| **롤링셔터** | 픽셀 행마다 **카메라 포즈가 다르다**. 투영 함수가 π(x; pose(t)) 인데 t 자체가 투영 결과(어느 행에 찍히느냐)에 의존한다 — 암시적 방정식이다. gsplat은 이걸 고정점 반복(기본 10회)으로 푼다. 이런 함수의 해석적 Jacobian은 사실상 불가능하다. |

3DGUT 논문(Wu et al., NVIDIA, CVPR 2025, *"3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting"*)의 문제의식이 정확히 여기다: **왜곡 영상을 미리 undistort 하지 않고 그대로 학습하고 싶다**. 어안 영상을 pinhole로 펴면 시야가 잘리거나 가장자리가 심하게 늘어난다.

---

## 2. Unscented Transform: "미분 대신 표본"

UT는 원래 **Unscented Kalman Filter**(Julier & Uhlmann, 그리고 Wan & van der Merwe 2000)에서 나왔다. EKF가 상태 전이 함수를 Jacobian으로 선형화하다가 강한 비선형에서 발산하는 문제를 풀려고 나온 기법이다. 문제 구조가 3DGS의 투영과 **똑같다**: "Gaussian을 비선형 함수에 통과시킨 뒤의 평균·공분산을 알고 싶다."

핵심 아이디어 한 줄:

> 확률분포를 근사하는 것이, 임의의 비선형 함수를 근사하는 것보다 쉽다.

절차:

1. 평균 m과 공분산 C를 **결정론적으로** 대표하는 2n+1개의 **시그마 포인트**를 뽑는다. 3D니까 n=3 → **7개**.
2. 각 시그마 포인트를 비선형 함수 f에 **그냥 통과**시킨다. 미분 없음. 근사 없음. 진짜 forward 투영 함수를 7번 부를 뿐이다.
3. 투영된 7개 점의 **가중 평균**과 **가중 공분산**을 다시 계산한다. 그게 출력 Gaussian.

> **중요**: 몬테카를로가 아니다. 시그마 포인트는 랜덤 샘플이 아니라 평균과 공분산을 정확히 재현하도록 배치한 결정론적 점들이다. 그래서 7개면 충분하다.

**정확도**: UT는 임의의 비선형 함수에 대해 결과 분포의 평균·공분산을 **테일러 전개 2차항까지 정확**하게 잡는다(가우시안 입력이면 α, β 튜닝으로 3차 일부까지). Jacobian 근사는 **1차까지만** 정확하다. 정확히 이 한 차수 차이가 어안/F-theta에서 벌어진다.

**비용**: 투영 함수 호출이 1번 → **7번**. 대신 Jacobian을 유도·구현·미분할 필요가 없어진다. 그리고 이 7번은 스레드 1개 안에서 `#pragma unroll`로 도는 고정 7회 루프라, 전체 파이프라인(정렬 + 블렌딩)에서 투영이 차지하는 비중을 생각하면 감내할 만한 값이다.

---

## 3. gsplat 구현 뜯어보기

### 3.1 시그마 포인트 생성 — 공분산 분해가 공짜다

`gsplat/cuda/include/Cameras.cuh` 의 `world_gaussian_sigma_points()`:

```cpp
const auto lambda = ut_lambda(params);          // λ = α²(D+κ) − D
glm::fmat3 R = glm::mat3_cast(gaussian_world_rot);

ret.points[0] = gaussian_world_mean;            // 중심점
for (auto i = 0u; i < D; ++i) {
    const auto delta      = std::sqrt(D + lambda) * gaussian_world_scale[i] * R[i];
    ret.points[i + 1]     = gaussian_world_mean + delta;   // m + sqrt((n+λ)C)_i
    ret.points[i + 1 + D] = gaussian_world_mean - delta;   // m − sqrt((n+λ)C)_i
}
```

일반적인 UT는 여기서 C의 촐레스키 분해나 행렬 제곱근을 구해야 한다. 그런데 3DGS는 공분산을 애초에 **C = (S·R)ᵀ(S·R)** 로 인수분해해 들고 있다(quat + scale). 코드 주석이 짚듯 이건 **닫힌 형태의 SVD**다: C = U Σ Uᵀ, U = Rᵀ, Σ = diag(sᵢ²). 즉 행렬 제곱근이 **회전축 방향 × scale**로 그냥 나온다. 3DGS 파라미터화와 UT의 궁합이 좋은 이유.

결과: 시그마 포인트는 **월드 공간에서** 중심 1개 + 3개 주축 방향으로 ±스프레드 6개 = 7개. (주의: 노트북 ①단계에서 만든 Σ를 쓰는 게 아니라 quats/scales를 직접 받는다 — 그래서 `with_ut=True`는 `covars`를 직접 넘기는 걸 거부한다.)

### 3.2 α, β, κ 와 가중치

```cpp
λ  = α²·(D + κ) − D
W₀^m = λ / (D + λ)                    // 중심점의 평균 가중치
W₀^c = W₀^m + (1 − α² + β)            // 중심점의 공분산 가중치
Wᵢ   = 1 / (2(D + λ))                 // 나머지 6개 (평균/공분산 공용)
```

각 파라미터의 뜻:

| 파라미터 | 의미 | gsplat 기본값 |
|---|---|---|
| **α** (alpha) | 시그마 포인트가 평균에서 얼마나 **멀리** 퍼지는지. 작을수록 평균 근처에 밀집 → 고차 비선형 효과(멀리 있는 이상치)를 덜 탄다. 보통 1e-4 ~ 1 사이. | **0.1** |
| **β** (beta) | 입력 분포에 대한 **사전 지식**. 입력이 가우시안이면 β=2가 4차 모멘트 기준 최적. 중심점의 공분산 가중치에만 들어간다. | **2.0** |
| **κ** (kappa) | 보조 스케일링. 보통 0 또는 3−n. | **0.0** |

`UnscentedTransformParameters` 생성자(`Cameras.cuh:59`)는 `α²(D+κ) > 0`을 강제한다 — 아니면 √(D+λ)가 NaN이 되고 가중치 분모가 발산한다.

**기본값을 실제로 넣어보면** (D=3, α=0.1, β=2, κ=0):

```
λ = 0.01·3 − 3 = −2.97
D + λ = 0.03           →  √(D+λ) ≈ 0.1732
W₀^m = −2.97 / 0.03 = −99
W₀^c = −99 + (1 − 0.01 + 2) = −96.01
Wᵢ   = 1 / 0.06 ≈ 16.667   (i = 1..6)

검산: 평균 가중치 합 = −99 + 6×16.667 = 1.0  ✓
```

두 가지가 눈에 띈다.

- **스프레드가 0.1732σ 로 아주 좁다.** 시그마 포인트가 1σ가 아니라 0.17σ 지점에 놓인다. 강한 왜곡에서 멀리 있는 점이 화면 밖으로 튀는 걸 막는 보수적 선택이다.
- **가중치가 음수다.** W₀ ≈ −99. 이게 UT의 알려진 부작용이고, gsplat 커널이 이걸 명시적으로 방어한다:

```cpp
// The UT center covariance weight can be very negative (e.g. ≈ -96 with
// default alpha=0.1).  This means the UT covariance estimate is not
// guaranteed positive-semidefinite ...
if (covar2d[0][0] < 0.f || covar2d[1][1] < 0.f) { radii = 0; return; }  // 컬링
```

즉 **UT 결과 공분산이 PSD 보장을 못 한다**. 행렬식이 양수여도 대각 원소가 음수일 수 있고, 그러면 `sqrtf`가 NaN을 낸다. gsplat은 그런 Gaussian을 그냥 버린다(radii=0).

### 3.3 `require_all_sigma_points_valid`

7개 시그마 포인트 각각은 투영될 때 valid/invalid 판정을 받는다 — 카메라 뒤에 있거나, 왜곡 다항식의 신뢰 구간 밖이거나(`icD > 0.8f` 체크), 이미지 경계 + 마진 밖이거나(`in_image_margin_factor`, 기본 **0.1** = 화면 밖 10%까지는 봐준다).

이 플래그가 "그럼 이 Gaussian 전체는 유효한가?"를 정한다:

```cpp
bool valid = params.require_all_sigma_points_valid;
...
if (params.require_all_sigma_points_valid) {
    valid &= point_valid;            // 7개 전부 유효해야 함 (AND)
    if (!point_valid) return {..., false};   // 하나라도 실패하면 즉시 탈출
} else {
    valid |= point_valid;            // 하나라도 유효하면 통과 (OR)
}
```

- **기본값 `false` (관대함, OR)**: 화면 경계에 걸친 Gaussian도 살린다. 시야 가장자리에서 갑자기 팝핑되는 걸 막는다. 대신 일부 시그마 포인트가 이상한 값을 내면 공분산 추정이 오염될 수 있다.
- **`true` (엄격, AND)**: 7개가 모두 멀쩡할 때만 그린다. 공분산 추정 품질은 좋아지지만 경계에서 Gaussian이 사라진다. 대신 조기 탈출이 가능해 약간 빠르다.

가중치 누적 루프의 트릭도 볼 만하다: `mean` 변수를 첫 반복에 W₀^m 로 쓰고 루프 끝에서 `mean = rest`로 갈아끼운다. 공분산 루프도 똑같이 `covariance → rest`. 분기 없이 "0번은 특별, 나머지는 동일" 을 처리한다.

### 3.4 카메라 모델과 롤링셔터

`with_ut=True`가 열어주는 것들:

**카메라 모델** (`CameraModel = Literal["pinhole", "ortho", "fisheye", "ftheta", "lidar"]`):

| 모델 | 왜곡 파라미터 | 비고 |
|---|---|---|
| `pinhole` | `radial_coeffs` [C,6], `tangential_coeffs` [C,2], `thin_prism_coeffs` [C,4] | OpenCV 모델. undistort는 뉴턴 반복 |
| `ortho` | — | 정사영 |
| `fisheye` | `radial_coeffs` [C,4] | OpenCV fisheye |
| `ftheta` | `ftheta_coeffs` (`FThetaCameraDistortionParameters`, 6차 다항식, PIXELDIST_TO_ANGLE / ANGLE_TO_PIXELDIST) | **`with_ut=True` 없이는 아예 못 쓴다** (`Rendering.cpp`에서 TORCH_CHECK로 막음). 180° 넘는 FOV는 `global_z_order=False`도 필요 |
| `lidar` | `lidar_coeffs` | 역시 `with_ut=True` 필수 |

C++ 쪽 카메라 모델 타입 리스트(`Cameras.cuh`)는 `PerfectPinhole / Orthographic / OpenCVPinhole / OpenCVFisheye / FTheta` × 외부 왜곡 모델의 **직교곱**으로 컴파일 타임에 생성된다.

**롤링셔터** (`RollingShutterType`):

```python
ROLLING_TOP_TO_BOTTOM = 0
ROLLING_LEFT_TO_RIGHT = 1
ROLLING_BOTTOM_TO_TOP = 2
ROLLING_RIGHT_TO_LEFT = 3
GLOBAL = 4   # 기본값 = 롤링셔터 없음
```

GLOBAL이 아니면 `viewmats_rs`(프레임 끝 포즈)를 함께 줘야 하고, `with_ut=True`가 **강제**된다. 각 시그마 포인트마다 이런 고정점 반복이 돈다(`world_point_to_image_point_shutter_pose`, 기본 `N_ROLLING_SHUTTER_ITERATIONS = 10`):

1. 시작 포즈로 투영 → 후보 픽셀 위치
2. 그 픽셀 위치에서 셔터 상대 시각 t ∈ [0,1] 계산 (`shutter_relative_frame_time`)
3. t로 포즈 보간: 위치는 lerp, 회전은 **slerp**
4. 새 포즈로 재투영 → 2번으로. t가 안 변하면 조기 탈출

Gaussian 하나당 slerp 엔드포인트는 동일하므로 `QuaternionSlerper`를 7개 시그마 포인트가 **공유**한다(LiDAR 경로). 이 반복 구조를 보면 왜 해석적 Jacobian이 사실상 불가능한지 바로 보인다 — **UT는 forward 함수만 있으면 되므로 이 루프를 그냥 7번 돌리면 끝난다.**

---

## 4. 3DGUT의 두 축: `with_ut` + `with_eval3d`

`docs/3dgut.md`가 명확히 한다:

> Setting `with_ut=True` and `with_eval3d=True` to enable 3DGUT (which is consist of two parts: using unscented transform to estimate the camera projection **and** evaluate Gaussian response in 3D space.)

| 축 | 플래그 | 담당 |
|---|---|---|
| **UT** | `with_ut` | **투영**. Jacobian → 시그마 포인트. 왜곡·롤링셔터 카메라를 지원 |
| **3D 평가** | `with_eval3d` | **래스터화**. 2D conic으로 σ를 재는 대신 **픽셀 광선 × 3D Gaussian**의 응답을 월드 공간에서 직접 평가 (`RasterizeToPixelsFromWorld3DGS*.cu`) |

논문 제목의 "and Secondary Rays"가 두 번째 축이다. 3D 공간에서 광선-Gaussian 응답을 계산하면 **카메라에서 나온 1차 광선이 아닌 광선**(반사, 굴절, 그림자)도 그대로 처리할 수 있다. 2D splat 근사에 갇혀 있으면 불가능한 일이다. 레이트레이싱 기반 3DGRT 대비 3DGUT는 래스터라이저 속도를 유지하면서 광선 질의를 열어준다.

**왜 둘을 같이 켜야 하나 — gradient 때문이다.** `Rendering.cpp`에서 UT 투영은 이렇게 호출된다:

```cpp
ProjectionUT3DGSFusedResult projection = [&]() {
    at::AutoGradMode no_grad(false);          // ← 미분 끔
    return call_torch_op<&projection_ut_3dgs_fused>(...);
}();
```

`ProjectionUT3DGSFused.cu`에는 **backward 커널이 없다**(파일 자체가 fwd 하나뿐). UT 투영은 순전히 forward-only, 타일 교차용 2D bounding 정보를 만드는 용도다. 그래서 `with_eval3d=True`로 3D 공간에서 응답을 평가해야 gradient가 means/quats/scales로 흘러간다. `with_ut`만 켜면 학습이 안 된다.

---

## 5. 다른 옵션과의 상호작용 (소스 확인)

`Rendering.cpp`의 검증 블록과 실제 호출부에서 뽑은 제약들:

| 상호작용 | 내용 | 근거 |
|---|---|---|
| **AccuTile 비활성화** | `with_ut=True`면 `intersect_tile`에 conics/opacities를 **빈 텐서로** 넘긴다 → 정밀한 타원 교차(AccuTile/SNUGBOX) 대신 **AABB 폴백** | `Rendering.cpp:1307-1308` `at::optional<at::Tensor> intersect_conics = as_optional_tensor(with_ut ? at::Tensor{} : kernel_conics);` (순수 PyTorch 경로도 동일: `rendering.py:899-900`) |
| **rasterize_mode** | `"classic"`만 허용. `"antialiased"` 불가 | `_validate_3dgut_rasterize_mode`, `"Antialiased rasterization is only supported for classic 3DGS"` |
| **packed** | 불가 (`"Packed mode is not supported with UT"`) | 노트북 9장의 `packed=True` 희소 표현과 배타적 |
| **sparse_grad** | 불가 | 위와 같은 이유 |
| **covars 직접 입력** | 불가. quats/scales가 **필수** (시그마 포인트를 만들려면 R과 s가 따로 필요) | `"UT and Eval3D rasterization require quats and scales, not covars"` |
| **distributed** | 불가 (`"distributed=True does not support with_ut=True"`) | |
| **global_z_order** | `False`는 `with_ut=True`일 때만 허용. 180° 넘는 FTheta FOV를 그리려면 `False` 필요(유클리드 거리 컬링) | `"global_z_order can be false only if with_ut=True"` |
| **absgrad** | `with_eval3d=True`면 `means2d.absgrad`가 안 붙는다 (`rendering.py:652`) → 밀도화 기준이 달라짐. gsplat이 3DGUT에서 **MCMC 전략만** 지원하는 이유 | `docs/3dgut.md`: "we only support MCMC densification strategy for 3DGUT" |
| **eps2d / opacity-aware radii** | UT 이후에도 그대로 적용. `add_blur(eps2d, covar2d, ...)`, `extend = min(3.33, sqrt(2·ln(α/(1/255))))` — 노트북 3장의 불투명도 인지 반경과 동일 | `ProjectionUT3DGSFused.cu` |

**AccuTile이 꺼지는 게 왜 자연스러운가**: `with_ut`는 보통 `with_eval3d`와 함께 쓰이고, 그때 실제 Gaussian 기여는 **3D에서** 계산된다. 커널 주석이 직접 설명한다 — 2D 기반 컬링/타일링은 "투영된 Gaussian이 어떻게 생겼을지에 대한 근사"에 불과하므로, 강한 비선형 왜곡에서는 3D에서 기여가 있는 Gaussian을 잘라낼 수도 있다. 성능과 품질의 타협.

---

## 6. Jacobian vs UT 한눈에

| | Jacobian (EWA, 기본) | Unscented Transform (`with_ut=True`) |
|---|---|---|
| 필요한 것 | 투영의 **해석적 미분** J | 투영의 **forward 함수만** |
| 정확도 | 1차까지 정확 | **2차까지 정확** |
| 투영 함수 호출 | 1번 (+ J 조립) | **7번** (2n+1, n=3) |
| 카메라 모델 | pinhole 전용 (`assert camera_model == "pinhole"`) | pinhole/ortho/fisheye/ftheta/lidar + 왜곡 계수 |
| 롤링셔터 | 불가 | 지원 (고정점 반복 10회) |
| 새 카메라 추가 | J를 유도하고 CUDA로 구현 | forward 투영만 구현하면 끝 |
| 수치 안정성 | x/z, y/z clamp 필요 | 공분산 PSD 보장 없음 → 음수 대각 컬링 필요 |
| 미분 가능 | O (bwd 커널 있음) | **X** (no_grad, bwd 커널 없음) → `with_eval3d` 필요 |
| AccuTile | 사용 | AABB 폴백 |
| 커널 | `ProjectionEWA3DGSFused.cu` | `ProjectionUT3DGSFused.cu` |

---

## 7. 언제 쓰나

- **자율주행 카메라 리그**: 서라운드뷰 어안 렌즈 + 롤링셔터 CMOS. 차가 움직이는 동안 한 프레임 안에서도 포즈가 변한다. 3DGUT의 원저자가 NVIDIA인 이유. gsplat의 `lidar` 카메라 모델과 spinning LiDAR 지원도 같은 맥락(회전하는 LiDAR는 본질적으로 롤링셔터다).
- **광각/액션캠 캡처**: GoPro, 어안 렌즈로 찍은 실내 스캔. undistort 하면 시야가 잘리고 가장자리 화질이 뭉개진다. `with_ut`면 **원본 왜곡 영상 그대로 학습**한다 (다만 COLMAP 등으로 **왜곡 파라미터 캘리브레이션은 여전히 필요**).
- **VR/파노라마**: 180° 넘는 FOV. `camera_model="ftheta"` + `global_z_order=False`.
- **반사·굴절이 필요한 장면**: `with_eval3d`가 열어주는 secondary ray.

반대로 **일반 pinhole 데이터셋(MipNeRF360 등)에서는 켤 이유가 없다**. 투영 7번 + AccuTile 손실 + MCMC 강제라는 비용만 지불하고, 정확도 이득은 거의 없다. 기본값이 `False`인 이유.

```bash
# 3DGUT 학습 (MCMC 전략 필수)
python examples/simple_trainer.py mcmc --with_ut --with_eval3d --camera_model fisheye ...
```

```python
rasterization(
    means, quats, scales, opacities, colors, viewmats, Ks, W, H,
    with_ut=True, with_eval3d=True,          # 3DGUT 두 축
    camera_model="ftheta",
    ftheta_coeffs=ftheta_params,
    rolling_shutter=RollingShutterType.ROLLING_TOP_TO_BOTTOM,
    viewmats_rs=viewmats_end,                # 프레임 끝 포즈
    ut_params=UnscentedTransformParameters(  # 기본값: 보통 건드릴 필요 없음
        alpha=0.1, beta=2.0, kappa=0.0,
        in_image_margin_factor=0.1,
        require_all_sigma_points_valid=False,
    ),
    global_z_order=False,                    # 180° 초과 FOV일 때
)
```

---

## 8. 기억할 한 문장

**Jacobian은 "함수를 근사"하고, UT는 "분포를 근사"한다.** 앞의 것은 미분이 필요하고 1차까지만 맞지만, 뒤의 것은 forward 호출 7번이면 되고 2차까지 맞는다. 어안·F-theta·롤링셔터처럼 미분이 어렵거나 비선형이 강한 카메라에서 이 교환이 이긴다.

## 참고 파일

- `gsplat/cuda/include/Cameras.h:50` — `UnscentedTransformParameters` (α/β/κ 기본값, `require_all_sigma_points_valid`)
- `gsplat/cuda/include/Cameras.cuh:1781~1935` — `ut_lambda`, `ut_weights`, `world_gaussian_sigma_points`, `world_gaussian_to_image_gaussian_unscented_transform_shutter_pose`
- `gsplat/cuda/include/Cameras.cuh:549~` — `world_point_to_image_point_shutter_pose` (롤링셔터 고정점 반복)
- `gsplat/cuda/csrc/ProjectionUT3DGSFused.cu` — UT 투영 커널 (forward only)
- `gsplat/cuda/csrc/Rendering.cpp:238~262, 885~930, 1300~1330` — 제약 검증, no_grad 호출, AccuTile 비활성화
- `gsplat/rendering.py:453~485, 815~905` — 파라미터 문서와 순수 PyTorch 참조 경로
- `gsplat/cuda/_torch_impl_ut.py` — `_fully_fused_projection_with_ut` (읽기 쉬운 PyTorch 참조 구현)
- `docs/3dgut.md` — 사용법
