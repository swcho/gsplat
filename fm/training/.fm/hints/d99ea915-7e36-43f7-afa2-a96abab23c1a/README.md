# 카메라 내부 파라미터에 왜곡계수가 있으면?

**답: 이미지를 미리 undistort(왜곡 보정)해서 학습에 넣는다. 그래야 렌더러가 순수한 pinhole 투영 모델을 그대로 쓸 수 있다.**

워크스루의 1단계 표에 그대로 적혀 있는 항목이다.

| 데이터 | 용도 |
|---|---|
| 카메라 포즈 `camtoworlds` [N,4,4] | 각 학습 이미지의 시점 |
| **내부 파라미터 `Ks` [3,3] (+왜곡계수)** | **투영 모델. 왜곡이 있으면 이미지를 미리 undistort** |
| sparse 3D 포인트 + RGB | Gaussian 초기 위치/색 |

---

## 1. 왜 "미리" 보정해야 하는가 — 렌더러가 pinhole만 안다

3DGS의 기본(classic) 렌더 경로는 3D Gaussian을 2D로 투영할 때 **투영 변환의 국소 선형화(야코비안 `J`)** 를 써서
3D 공분산 Σ를 2D 공분산 `JWΣWᵀJᵀ`로 밀어 보낸다(EWA splatting). 이 닫힌 형식은 투영이
`x/z, y/z` + `K` 라는 **매끄러운 사영 변환**일 때만 성립한다. 렌즈 왜곡처럼 반경 방향으로
비선형하게 픽셀을 끌어당기는 항이 끼면 "3D Gaussian → 2D Gaussian"이라는 전제 자체가 깨진다.

그래서 gsplat의 기본 투영 경로에는 아예 단정문이 박혀 있다
(`gsplat/rendering.py`, `_fully_fused_projection` 로 가는 else 분기):

```python
if with_ut:
    ... _fully_fused_projection_with_ut(...)   # UT 경로: 왜곡 처리 가능
else:
    assert camera_model == "pinhole", camera_model   # 기본 경로: pinhole 강제
```

즉 **왜곡을 렌더러가 흉내낼 수 없으니, 데이터 쪽에서 왜곡을 없애 버린다.** 왜곡이 제거된
이미지는 정의상 pinhole 카메라가 찍은 이미지와 동등하므로, 이후 파이프라인은 `K`(3x3) 하나만
알면 된다. 워크스루가 `rasterize_splats(splats, c2w, K, W, H, ...)`처럼 `K`만 넘기고
왜곡계수를 아예 인자로 갖지 않는 이유이기도 하다("`rasterize_splats` ← `Runner.rasterize_splats`,
비고: appearance/post-processing/**왜곡 카메라 생략**").

## 2. gsplat이 실제로 하는 일 — `examples/datasets/colmap.py`

### (a) COLMAP 카메라 모델 → 왜곡계수 추출

`_camera_distortion()`이 COLMAP 카메라 모델명을 OpenCV 왜곡계수 벡터로 정규화한다.

| COLMAP 모델 | 반환 계수 | camtype |
|---|---|---|
| `SIMPLE_PINHOLE`, `PINHOLE` | 빈 배열 (왜곡 없음) | perspective |
| `SIMPLE_RADIAL` | `[k1, 0, 0, 0]` | perspective |
| `RADIAL` | `[k1, k2, 0, 0]` | perspective |
| `OPENCV` | `params[4:8]` = `[k1, k2, p1, p2]` | perspective |
| `OPENCV_FISHEYE` | `params[4:8]` = `[k1..k4]` | fisheye |
| 그 외 | `ValueError` (지원 안 함) | — |

계수가 하나라도 있으면 로드 시 경고를 띄운다:
`"Warning: COLMAP Camera is not PINHOLE. Images have distortion."`

### (b) Parser에서 **한 번만** remap 테이블을 만든다

`Parser.__init__`의 `# undistortion` 블록. `len(params) == 0`이면 `continue`(할 일 없음).

- **perspective**:
  ```python
  K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(K, params, (width, height), 0)
  mapx, mapy = cv2.initUndistortRectifyMap(K, params, None, K_undist,
                                           (width, height), cv2.CV_32FC1)
  ```
  `alpha=0`이므로 "유효 픽셀만 남는 최대 사각형" ROI를 얻는다(검은 테두리 없음, 시야는 조금 잘림).
- **fisheye**: OpenCV 함수 대신 직접 격자를 만들어
  `r = 1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸` 를 곱해 `mapx/mapy`를 계산하고, 원본 범위 안에 떨어지는
  픽셀만 `mask`로 표시한 뒤 그 bounding box를 ROI로 잡는다. `K_undist`는 ROI 오프셋만큼
  주점을 이동(`cx -= x_min`, `cy -= y_min`).

그리고 **파서의 상태를 "보정된 카메라"로 덮어쓴다**:

```python
self.mapx_dict[camera_id] = mapx
self.mapy_dict[camera_id] = mapy
self.Ks_dict[camera_id]   = K_undist          # ← K가 교체된다
self.roi_undist_dict[camera_id] = roi_undist
self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])   # ← W,H도 줄어든다
self.mask_dict[camera_id] = mask
```

이게 핵심이다. 이 시점 이후로 파이프라인 전체(포즈 정규화, `scene_scale`, 렌더 해상도, 평가)는
**왜곡이 없는 가상의 pinhole 카메라**만 상대한다.

### (c) Dataset에서 이미지마다 remap 적용

`Dataset.__getitem__`:

```python
K = self.parser.Ks_dict[camera_id].copy()   # undistorted K
params = self.parser.params_dict[camera_id]
if len(params) > 0:
    # Images are distorted. Undistort them.
    mapx, mapy = self.parser.mapx_dict[camera_id], self.parser.mapy_dict[camera_id]
    image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
    x, y, w, h = self.parser.roi_undist_dict[camera_id]
    image = image[y : y + h, x : x + w]      # ROI 크롭
```

정리하면 **왜곡계수는 이미지 픽셀을 재배치하는 데 소진되고, 학습 텐서에는 남지 않는다.**
`data["K"]`는 이미 `K_undist`이고, `data["image"]`는 이미 보정+크롭된 이미지다. 워크스루의

```python
K = data0["K"][None].to(DEVICE)   # [1,3,3]
```

한 줄이 왜 왜곡 정보 없이도 안전한지가 여기서 나온다. (fisheye의 `mask`는 유효 픽셀 표시용으로
따로 넘어가 손실 계산에서 무효 영역을 제외하는 데 쓰인다.)

## 3. 대가(trade-off)

undistort는 공짜가 아니다. 카드의 답을 외울 때 같이 기억해 둘 부작용:

1. **리샘플링 흐림**: `cv2.remap` + `INTER_LINEAR`은 보간이므로 고주파 디테일이 약간 뭉개진다.
   3DGS는 디테일을 재현하는 것이 목표이므로 이 손실은 상한선을 조금 낮춘다.
2. **시야/해상도 손실**: `alpha=0` ROI 크롭 때문에 이미지 주변부가 잘려 나간다. 광각/어안일수록 크게 잘린다.
3. **어안은 원리적으로 손해**: 180° 급 FOV를 pinhole 평면에 펼치면 주변부가 극단적으로 늘어나
아예 표현이 불가능한 영역이 생긴다. 그래서 어안 데이터는 undistort보다 아래의 네이티브 경로가 낫다.
4. **전처리 비용/디스크**: COLMAP의 `image_undistorter`로 미리 파일을 만들어 두는 방식도 흔하다
(gsplat은 런타임 remap 방식).

## 4. 대안 — 왜곡을 렌더러가 직접 처리하는 길 (3DGUT)

워크스루 마지막 "여기서 더 볼 것들"에 힌트가 있다:
`카메라 모델: pinhole 외 fisheye/ftheta, 왜곡계수, rolling shutter, 3DGUT(with_ut)`.

`rasterization()`은 다음 인자들을 갖고 있다.

```python
camera_model: CameraModel = "pinhole",   # "pinhole" | "ortho" | "fisheye" | "ftheta"
with_ut: bool = False,                   # Unscented Transform 투영
with_eval3d: bool = False,               # 3D 공간에서 Gaussian 응답 평가
radial_coeffs: Optional[Tensor] = None,      # [..., C, 6] (pinhole) / [..., C, 4] (fisheye)
tangential_coeffs: Optional[Tensor] = None,  # [..., C, 2]
thin_prism_coeffs: Optional[Tensor] = None,  # [..., C, 4]
ftheta_coeffs: ...
```

docstring이 밝히듯 이것은
[3DGUT (arXiv:2412.12507)](https://arxiv.org/abs/2412.12507) 방식이다. 야코비안 선형화를
버리고 **Unscented Transform**(Gaussian에서 sigma point 몇 개를 뽑아 실제 비선형 투영을
통과시킨 뒤 2D 분포를 다시 적합)으로 투영하기 때문에, 왜곡·어안·롤링셔터 같은
**비선형 카메라 모델을 그대로** 처리할 수 있다. `simple_trainer.py`의
`Runner.rasterize_splats`도 데이터에 계수가 있으면 이 인자들로 흘려보낸다:

```python
if cam.radial_coeffs is not None:
    radial_coeffs = torch.from_numpy(cam.radial_coeffs).to(means.device).unsqueeze(0)
...
rasterization(..., camera_model=camera_model, with_ut=with_ut,
              radial_coeffs=radial_coeffs, tangential_coeffs=..., thin_prism_coeffs=...)
```

단, 이 계수들은 `with_ut=True` 경로에서만 의미가 있고(기본 경로는 위의 `assert`에 걸린다),
3DGUT 모드는 `rasterize_mode="classic"`만 지원하며 속도 비용도 있다.

## 5. 한 줄 요약

- **기본 답**: 왜곡계수가 있으면 → `cv2.initUndistortRectifyMap` + `cv2.remap`으로 **이미지를 먼저 펴고**,
  `K`를 `getOptimalNewCameraMatrix`의 `K_undist`로 바꾸고, ROI만큼 크롭한다.
  그 뒤 렌더러는 왜곡을 모른 채 pinhole 투영만 한다.
- **왜**: classic 3DGS 투영은 EWA 야코비안 근사라 비선형 왜곡을 표현할 수 없다.
- **예외**: 3DGUT(`with_ut=True` + `radial_coeffs`/`fisheye`/`ftheta`)를 쓰면 왜곡을
  보정하지 않고 렌더 단계에서 직접 다룰 수 있다.

## 참고 파일

- `/home/sungwoo/projects/swcho/gsplat/examples/datasets/colmap.py`
  — `_camera_distortion()`, `Parser.__init__`의 `# undistortion` 블록, `Dataset.__getitem__`
- `/home/sungwoo/projects/swcho/gsplat/gsplat/rendering.py`
  — `rasterization()` 시그니처/docstring("Camera Distortion and Rolling Shutter"), `assert camera_model == "pinhole"`
- `/home/sungwoo/projects/swcho/gsplat/examples/simple_trainer.py`
  — `Runner.rasterize_splats`의 왜곡계수 전달부, `cfg.camera_model` / `cfg.with_ut`
