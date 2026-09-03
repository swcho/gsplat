# gsplat이 지원하는 카메라 모델 / 기능

> **Q.** gsplat이 지원하는 카메라 모델·기능에는 어떤 것들이 있는가?
> **A.** pinhole 외에 fisheye/ftheta, 왜곡계수, rolling shutter, 3DGUT(`with_ut`)를 지원한다.

---

## 1. 이 카드가 나온 맥락

워크스루(`training_walkthrough.py`)는 **가장 단순한 pinhole 경로만** 쓴다.
1단계 데이터 준비 표(L84)에 이렇게 적혀 있다.

| 데이터 | 용도 |
|---|---|
| 내부 파라미터 `Ks` [3,3] (+왜곡계수) | 투영 모델. **왜곡이 있으면 이미지를 미리 undistort** |

즉 노트북은 "왜곡은 전처리로 지워버리고, 렌더러는 순수 pinhole만 본다"는 전략이다.
실제로 `examples/datasets/colmap.py`의 `Parser`가 COLMAP 카메라 타입을 보고
(`SIMPLE_RADIAL` / `OPENCV` / `OPENCV_FISHEYE` 등) `cv2.initUndistortRectifyMap`으로
리맵 테이블을 미리 만들어 두고, 학습 때 이미지를 왜곡 없는 상태로 넘긴다
(`colmap.py:366~420`). 로드 시 `"Warning: COLMAP Camera is not PINHOLE. Images have distortion."`
경고가 뜨는 지점이 바로 여기다.

그래서 마지막 "여기서 더 볼 것들" 절(L472)에 **"pinhole 외 fisheye/ftheta, 왜곡계수,
rolling shutter, 3DGUT(`with_ut`)"** 가 남는다. 이 카드는 "노트북이 생략한, 진짜 카메라를
그대로 다루는 경로"의 목록이다.

---

## 2. 카메라 모델 — `camera_model` 인자

`gsplat/cuda/_wrapper.py:38`이 단일 소스다.

```python
CameraModel = Literal["pinhole", "ortho", "fisheye", "ftheta", "lidar"]
```

`rasterization(..., camera_model="pinhole")`이 기본값이며, 문자열이
`CameraModelType.PINHOLE` 같은 CUDA enum으로 변환되어 커널로 내려간다
(`gsplat/rendering.py:571`).

| 값 | 모델 | 파이썬 참조 구현 | 비고 |
|---|---|---|---|
| `pinhole` | OpenCV 핀홀 (원근 투영) | `_OpenCVPinholeCameraModel` | 기본값. radial/tangential/thin-prism 왜곡 지원 |
| `ortho` | 정사 투영 | `_OrthographicCameraModel` | 깊이에 무관한 스케일 |
| `fisheye` | OpenCV 어안 (등거리) | `_OpenCVFisheyeCameraModel` | 9차 홀수 다항식 forward, 역변환은 Newton 20회 |
| `ftheta` | NVIDIA F-Theta | `_FThetaCameraModel` | 초점거리가 다항식에 흡수됨. **`with_ut=True` 필수** |
| `lidar` | 회전형 LiDAR 그리드 | — | `lidar_coeffs` 필수, `with_ut=True` 필수 |

참조 구현은 `gsplat/cuda/_torch_cameras.py`에 있고(테스트/디버그용 순수 torch),
실제 렌더링은 `gsplat/cuda/csrc/`의 CUDA 커널이 담당한다.

### fisheye 모델의 핵심

핀홀은 `r = f·tan(θ)`라 θ가 90°에 가까워지면 발산한다. 어안은 각도 자체를
반지름으로 매핑한다.

```
θ + k1·θ³ + k2·θ⁵ + k3·θ⁷ + k4·θ⁹     # forward: 입사각 → 정규화 거리
```

`radial_coeffs`가 `[k1,k2,k3,k4]` (shape `[..., C, 4]`)이고, 역방향(픽셀→광선)은
닫힌 해가 없어 Newton 반복으로 푼다. 180° 이상의 화각을 한 장에 담을 수 있는 이유다.

### ftheta 모델의 핵심

F-Theta는 forward(`angle_to_pixeldist`)와 backward(`pixeldist_to_angle`)
**두 개의 6계수 다항식**을 들고 있고, 둘 중 어느 쪽을 기준(reference)으로 볼지
`reference_poly_type`으로 고른다. 나머지 방향은 Newton 3회로 역산한다.
추가로 왜곡 좌표에 `[c d; e 1]` 선형 변환(`linear_cde`)을 곱해 픽셀 비대칭을 흡수한다.
`max_angle`은 **캘리브레이션에서 온 값**이며, 화각을 넓히려고 임의로 올리면
다항식이 신뢰 구간(`[0, max_angle]`) 밖에서 발산한다 —
`rasterization` docstring이 명시적으로 경고하는 부분이다.

180°를 넘는 밴드를 렌더링하려면 `global_z_order=False`가 필요하다
(카메라 z 기준 컬링 대신 유클리드 거리 기준 near/far 컬링으로 전환).

---

## 3. 왜곡 계수 — 이미지를 안 펴고 그대로 학습하기

`rasterization()` 시그니처의 왜곡 관련 인자 4종:

| 인자 | shape | 적용 모델 | 의미 |
|---|---|---|---|
| `radial_coeffs` | `[..., C, 6]` (pinhole) / `[..., C, 4]` (fisheye) | pinhole, fisheye | 반경 방향 왜곡 k1..k6 |
| `tangential_coeffs` | `[..., C, 2]` | pinhole | 렌즈-센서 비평행에서 오는 접선 왜곡 p1,p2 |
| `thin_prism_coeffs` | `[..., C, 4]` | pinhole | thin-prism 항 s1..s4 |
| `ftheta_coeffs` | `FThetaCameraDistortionParameters` | ftheta | 다항식 2쌍 + `linear_cde` (카메라 전체 공유) |

pinhole 쪽 `radial_coeffs`는 내부에서 항상 6개로 zero-pad 되므로 4개만 줘도 된다
(`_torch_cameras.py:963`). 왜곡 → 정규화 좌표 역변환은 고정 5회 반복
(`max_undistortion_iterations=5`).

**핵심 차이**: 노트북/`colmap.py` 방식은 *이미지를 왜곡 없이 리샘플링*하고
(→ 보간 손실 + 어안은 화면 가장자리를 통째로 버림), 이 방식은 *렌더러가 왜곡을 그대로
재현*한다. 그래서 원본 픽셀을 손대지 않고 학습할 수 있다.
단, **캘리브레이션(COLMAP 등)으로 왜곡 파라미터를 얻는 것 자체는 여전히 필요**하다.

---

## 4. 롤링 셔터

CMOS 센서는 한 프레임을 한 순간에 노출하지 않고 행(또는 열)을 순차 스캔한다.
카메라가 움직이면 위/아래 행이 서로 다른 포즈에서 찍힌다(젤로 현상).

`gsplat/cuda/_wrapper.py:194`:

```python
class RollingShutterType(IntEnum):
    ROLLING_TOP_TO_BOTTOM = 0
    ROLLING_LEFT_TO_RIGHT = 1
    ROLLING_BOTTOM_TO_TOP = 2
    ROLLING_RIGHT_TO_LEFT = 3
    GLOBAL = 4          # 기본값 = 롤링셔터 없음
```

쓰는 법: `rolling_shutter=<타입>` 과 함께 **두 번째 뷰 행렬** `viewmats_rs`를 넘긴다.
`viewmats`는 셔터 시작 시점 포즈, `viewmats_rs`는 셔터 종료 시점 포즈다.
픽셀의 상대 시각 `t∈[0,1]`에 대해 포즈를 보간한다
(`_interpolate_shutter_pose`, `_torch_cameras.py:2163`):

- 평행이동: 선형 보간 `(1-α)·t_start + α·t_end`
- 회전: 쿼터니언 **slerp** 후 정규화

즉 "각 픽셀이 자기 시각의 카메라 포즈로 투영된다". 자율주행/핸드헬드 영상처럼
카메라가 빠르게 움직이는 데이터에서 기하 오차를 크게 줄여준다.

---

## 5. 3DGUT — 이 모든 걸 가능하게 하는 엔진

### 왜 필요한가

기존 3DGS의 투영은 **EWA splatting** — 가우시안 중심 주변에서 투영 함수를
1차 테일러 전개(야코비안 `J`)해서 3D 공분산을 2D로 옮긴다: `Σ₂ᴰ ≈ J Σ₃ᴰ Jᵀ`.
이 국소 아핀 근사는 **투영이 거의 선형일 때만** 맞는다. 어안·강한 왜곡·롤링셔터는
심하게 비선형이라 근사가 깨진다. 그리고 픽셀 시각마다 포즈가 다른 롤링셔터에는
애초에 하나의 `J`를 정의할 수 없다.

### 어떻게 푸는가

NVIDIA [3DGUT](https://research.nvidia.com/labs/toronto-ai/3DGUT/)
([arXiv:2412.12507](https://arxiv.org/abs/2412.12507))는 야코비안 대신
**Unscented Transform**을 쓴다. 칼만 필터의 UKF에서 온 아이디어다.

1. 3D 가우시안마다 시그마 포인트 `2·D+1 = 7`개를 생성 (중심 1 + 주축 방향 ±3)
   — `_world_gaussian_sigma_points`, `_torch_impl_ut.py:111`
2. 각 시그마 포인트를 **실제 비선형 카메라 함수 그대로** 통과시킴
   (어안이든 ftheta든 롤링셔터 보간이든 상관없음)
3. 투영된 7개 점의 가중 평균·공분산으로 2D 가우시안을 복원
   (가중치는 `alpha, beta, kappa`로 결정, `lambda = α²(D+κ) - D`)

`with_ut=True`가 이 **투영** 부분이고, `with_eval3d=True`는 짝이 되는 **셰이딩** 부분이다
— 가우시안 응답(alpha)을 2D 화면 공간이 아니라 **3D 월드 공간에서 광선을 따라** 평가한다.
docs/3dgut.md의 표현대로 "3DGUT은 이 두 부분으로 구성"되므로 보통 **둘 다 켠다**.

### 쓰는 법

```bash
# gsplat은 3DGUT에 대해 MCMC densification만 지원한다
python examples/simple_trainer.py mcmc --with_ut --with_eval3d --camera_model fisheye ...
```

API 레벨:

```python
rasterization(
    ..., camera_model="fisheye",
    radial_coeffs=k,               # [C,4]
    with_ut=True, with_eval3d=True,
    rolling_shutter=RollingShutterType.ROLLING_TOP_TO_BOTTOM,
    viewmats_rs=viewmats_end,
)
```

`simple_trainer.py`는 `cfg.camera_model`(L106), `cfg.with_ut`/`cfg.with_eval3d`(L260-261)를
받아 `Runner.rasterize_splats`(L690~750)에서 카메라별 왜곡 계수와 함께 그대로 전달한다.

---

## 6. 제약 조건 — 시험에 나오는 부분

코드가 실제로 `TORCH_CHECK`/`ValueError`로 막는 조합들:

| 제약 | 근거 |
|---|---|
| `ftheta`는 `with_ut=True` 없이 못 씀 | `Rendering.cpp:252`, `Projection.cpp:49` — *"ftheta camera is only supported via UT"* |
| `lidar`도 `with_ut=True` 필수 (+`lidar_coeffs` 필수) | `Rendering.cpp:258` |
| 3DGUT는 `rasterize_mode="classic"`만 — `antialiased` 불가 | `_validate_3dgut_rasterize_mode` (`rendering.py:162`), `Rendering.cpp:241` |
| `global_z_order=False`는 `with_ut=True`일 때만 | `Rendering.cpp:249` |
| `with_eval3d`면 `tile_size ∈ {8,16}`, 아니면 `{4,16}` | `Rendering.cpp:260~272` |
| 3DGUT + `DefaultStrategy` 조합 금지 → **MCMCStrategy** 사용 | `simple_trainer.py:1611`, docs/3dgut.md |
| `rasterization()`은 `Ks`에 대해 **미분 불가** | `rendering.py` docstring `.. warning::` |

반대로 `pinhole` / `ortho` / `fisheye`는 UT 없는 기존 EWA 경로
(`ProjectionEWA3DGSFused.cu`, `ProjectionEWA3DGSPacked.cu`)에서도 지원된다 —
**왜곡 계수 없는 순수 투영에 한해서**다. 왜곡·롤링셔터를 실제로 적용하려면 UT 경로가 필요하다.

---

## 7. 한 줄 정리

> `camera_model`(pinhole/ortho/fisheye/ftheta/lidar) + 왜곡 계수 3종(+ftheta_coeffs) +
> `rolling_shutter`/`viewmats_rs` — 이 비선형 카메라들을 **선형 EWA 근사 없이** 렌더링하기
> 위한 엔진이 `with_ut`(UT 투영) + `with_eval3d`(3D 응답 평가) = **3DGUT**이고,
> 덕분에 undistort 전처리 없이 원본 이미지로 바로 학습할 수 있다.

---

### 더 읽을 곳

- `/home/sungwoo/projects/swcho/gsplat/docs/3dgut.md` — 사용법 요약
- `/home/sungwoo/projects/swcho/gsplat/gsplat/rendering.py:260~500` — `rasterization()` 시그니처와 docstring
- `/home/sungwoo/projects/swcho/gsplat/gsplat/cuda/_torch_cameras.py` — 5종 카메라 모델의 순수 torch 참조 구현
- `/home/sungwoo/projects/swcho/gsplat/gsplat/cuda/_torch_impl_ut.py` — 시그마 포인트 생성/투영
- `/home/sungwoo/projects/swcho/gsplat/examples/datasets/colmap.py:366~420` — 대조군: 사전 undistort 방식
