# COLMAP `Parser`가 읽어오는 세 가지 데이터와 각각의 용도

> **Q.** COLMAP `Parser`가 읽어오는 세 가지 데이터와 각각의 용도는?
>
> **A.** 카메라 포즈 `camtoworlds` [N,4,4]는 각 학습 이미지의 시점, 내부 파라미터 `Ks` [3,3](+왜곡계수)는 투영 모델, sparse 3D 포인트+RGB는 Gaussian의 초기 위치/색(`init_type="sfm"`)으로 쓰인다.

관련 소스: `examples/datasets/colmap.py` (`Parser`, `Dataset`), `examples/simple_trainer.py` (`create_splats_with_optimizers`, `Runner.__init__`), 워크스루 1단계(`training_walkthrough.py` L76~120).

---

## 0. 큰 그림 — 왜 "세 가지"인가

3DGS 학습은 **카메라를 고정하고 Gaussian만 최적화**하는 문제다. 미분가능 래스터라이저는 "이 시점·이 렌즈로 보면 이렇게 보인다"를 계산해서 실제 사진과 비교하는데, 그러려면 학습 전에 **씬의 기하 좌표계가 이미 확정**되어 있어야 한다. 그 확정 작업을 COLMAP(Structure-from-Motion)이 미리 해 두고, `Parser`는 그 결과를 세 덩어리로 꺼내 온다.

| 데이터 | `Parser` 속성 | 무엇에 쓰이나 | 학습 중 최적화되나 |
|---|---|---|---|
| 카메라 **외부** 파라미터 (포즈) | `camtoworlds` [N,4,4] | 각 학습 이미지의 시점 → `viewmats = inv(camtoworlds)` | ❌ 고정 (옵션 `pose_opt`로만 미세보정) |
| 카메라 **내부** 파라미터 | `Ks_dict[camera_id]` [3,3], `params_dict`(왜곡계수) | 투영 모델. 왜곡이 있으면 이미지를 미리 undistort | ❌ 고정 |
| sparse **3D 포인트 + 색** | `points` [M,3], `points_rgb` [M,3] | Gaussian 초기 위치·색 (`init_type="sfm"`) | ✅ 초기값으로만 쓰이고 이후 최적화·밀도화 대상 |

읽는 진입점은 하나다 — `pycolmap.Reconstruction(colmap_dir)` (`colmap.py:144`, `data_dir/sparse/0/` 또는 `data_dir/sparse/`). 이 안에 `cameras`(내부), `images`(외부 + 2D 트랙), `points3D`(3D 포인트)가 들어 있고, 세 가지 데이터는 정확히 이 세 테이블에 1:1 대응한다.

```
sparse/0/cameras.bin   → Ks_dict, params_dict, imsize_dict     (내부 파라미터)
sparse/0/images.bin    → camtoworlds, image_names, camera_ids   (외부 파라미터=포즈)
sparse/0/points3D.bin  → points, points_rgb, points_err,        (sparse 포인트클라우드)
                          point_indices
```

---

## 1. 카메라 포즈 `camtoworlds` [N,4,4] — "각 학습 이미지의 시점"

### 어떻게 만들어지나

COLMAP은 포즈를 **world→camera** 로 저장한다. `Parser`는 이를 뒤집어서 camera→world로 바꾼다.

```python
w2c_mats.append(_image_w2c(im))          # colmap.py:160  (image.cam_from_world)
w2c_mats = np.stack(w2c_mats, axis=0)
camtoworlds = np.linalg.inv(w2c_mats)    # colmap.py:190
```

- `_image_w2c` (`colmap.py:110`)는 `image.cam_from_world.matrix()` [3,4]를 4x4 homogeneous로 패딩한다.
- 이미지 목록은 **파일명 정렬**로 재배열된다(`inds = np.argsort(image_names)`, `colmap.py:198`). 예전 NeRF 결과와 같은 test set을 쓰기 위한 조치이며, 덕분에 `test_every` 기반 train/val 분리가 재현 가능해진다.

### 세 가지 용도

1. **렌더 시점** — 래스터라이저는 world→cam을 원하므로 다시 역행렬을 취한다. 워크스루 `rasterize_splats`(L237)와 `simple_trainer.py`의 동일 지점:
   ```python
   viewmats=torch.linalg.inv(camtoworlds)   # world→cam
   ```
   즉 `camtoworlds`는 "카메라가 월드 어디에, 어떤 자세로 있는가"를 담고, 렌더링 시엔 그 역변환으로 Gaussian을 카메라 좌표계로 끌어온다.

2. **월드 정규화 (`normalize=True`)** — 카메라 배치를 기준으로 similarity 변환을 구해 포즈와 포인트를 **함께** 옮긴다 (`colmap.py:277~307`):
   ```python
   T1 = similarity_from_cameras(camtoworlds)   # 카메라 분포 기준 정규화
   T2 = align_principal_axes(points)           # 포인트 주축 정렬
   # + z 뒤집힘 보정 T3 (median(z) > mean(z)이면 x축 180° 회전)
   self.transform = T3 @ T2 @ T1
   ```
   포즈와 포인트가 같은 변환을 받으므로 상대 기하는 보존되고, 좌표 스케일만 학습에 좋은 범위로 옮겨진다.

3. **`scene_scale` 산출** — 카메라 위치들의 중심에서 가장 먼 카메라까지의 거리 (`colmap.py:437~440`):
   ```python
   camera_locations = camtoworlds[:, :3, 3]
   scene_center = np.mean(camera_locations, axis=0)
   self.scene_scale = np.max(np.linalg.norm(camera_locations - scene_center, axis=1))
   ```
   이 스칼라가 **learning rate와 밀도화 임계값의 기준 단위**가 된다. `simple_trainer.py:458`에서 `scene_scale * 1.1 * global_scale`로 쓰이고, `means`의 lr에 곱해지며(`simple_trainer.py:336`), `DefaultStrategy`의 `grow_scale3d`/`prune_scale3d` 판정도 이 값에 상대적이다. 그래서 포즈는 "시점"이면서 동시에 **씬의 물리적 단위를 정의하는 자**이기도 하다.

---

## 2. 내부 파라미터 `Ks` [3,3] + 왜곡계수 — "투영 모델"

### 저장 형태

카드는 `Ks` [3,3]이라고 적지만, `Parser`가 들고 있는 실체는 **카메라별 딕셔너리**다 — 한 씬에 여러 대의 카메라(서로 다른 intrinsic)가 섞여 있을 수 있기 때문이다.

```python
self.Ks_dict      # camera_id -> K [3,3]
self.params_dict  # camera_id -> 왜곡계수 (없으면 길이 0 배열)
self.imsize_dict  # camera_id -> (width, height)
self.mask_dict    # camera_id -> undistort ROI 마스크 (fisheye)
```

[3,3] 텐서 하나로 등장하는 지점은 `Dataset.__getitem__`이 뱉는 `data["K"]`이고, 배치 차원이 붙은 `Ks` [B,3,3]는 학습 루프에서 `rasterization(..., Ks=Ks, ...)`로 그대로 들어간다(워크스루 L360, L238).

### factor / 실제 이미지 크기 보정

다운샘플 배율만큼 K의 앞 두 행(fx, fy, cx, cy)을 나눈다:

```python
K = np.asarray(cam.calibration_matrix(), dtype=np.float64)
K[:2, :] /= factor                        # colmap.py:169
imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)
```

그 뒤 **실제 파일을 한 장 읽어서** COLMAP 기록과 해상도가 어긋나면 다시 스케일을 맞춘다(`colmap.py:357~364`). Tanks&Temples처럼 COLMAP intrinsic이 2x 업샘플 이미지 기준인 데이터셋을 구제하는 코드다.

### 왜곡(distortion) 처리 — "왜곡이 있으면 이미지를 미리 undistort"

`_camera_distortion`(`colmap.py:88`)이 COLMAP 카메라 모델을 두 부류로 접는다:

| COLMAP 모델 | 반환 계수 | camtype |
|---|---|---|
| `SIMPLE_PINHOLE`, `PINHOLE` | 빈 배열 → 왜곡 없음 | perspective |
| `SIMPLE_RADIAL` | `[k1,0,0,0]` | perspective |
| `RADIAL` | `[k1,k2,0,0]` | perspective |
| `OPENCV` | `params[4:8]` | perspective |
| `OPENCV_FISHEYE` | `params[4:8]` | fisheye |
| 그 외 | `ValueError` | — |

핀홀이 아니면 경고를 찍고(`colmap.py:185`), **미리 undistort 맵을 만들어 둔다** (`colmap.py:368~434`):

- perspective: `cv2.getOptimalNewCameraMatrix` → `K_undist`, `cv2.initUndistortRectifyMap` → `mapx/mapy`
- fisheye: `theta` 급수 `1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸`로 맵을 직접 구성하고, 유효 영역 마스크로 ROI를 잡아 `K_undist`의 주점(cx, cy)을 ROI만큼 평행이동

그리고 **`Ks_dict`와 `imsize_dict`를 undistort 후 값으로 덮어쓴다.** 실제 remap은 이미지 로드 시점에 일어난다 (`Dataset.__getitem__`, `colmap.py:477~483`):

```python
image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
x, y, w, h = self.parser.roi_undist_dict[camera_id]
image = image[y:y+h, x:x+w]
```

**핵심 이유**: gsplat의 기본 래스터라이저는 **순수 핀홀 투영**만 계산한다. 렌즈 왜곡을 렌더러 안에서 모방하지 않고, 대신 **GT 이미지를 핀홀처럼 보이게 펴서** 모델과 데이터의 투영 모델을 일치시킨다. 그래서 "미리 undistort"가 선택이 아니라 정합성 조건이다.

---

## 3. sparse 3D 포인트 + RGB — "Gaussian 초기 위치/색"

### 읽는 부분

```python
points3D  = _as_dict(reconstruction.points3D)
points     = np.array([points3D[i].xyz   for i in ids], dtype=np.float32)  # [M,3]
points_err = np.array([points3D[i].error for i in ids], dtype=np.float32)  # [M,]
points_rgb = np.array([points3D[i].color for i in ids], dtype=np.uint8)    # [M,3] 0~255
```
(`colmap.py:245~258`) — `point3D_id`를 **정렬**해서 인덱싱하므로 실행마다 순서가 안정적이다.

덤으로 `point_indices`(`colmap.py:262~273`)는 `image_name -> [해당 이미지에서 관측된 포인트 인덱스]` 매핑이다. 각 포인트의 `track.elements`를 뒤집어 만들며, `Dataset(load_depths=True)`가 이걸로 sparse depth를 투영해 `depth_loss`(disparity L1)의 supervision으로 쓴다(`colmap.py:513~532`).

### Gaussian 초기화로의 변환

`simple_trainer.py:311~346` (워크스루 `init_splats_with_optimizers`, L157~200):

```python
if init_type == "sfm" or init_type == "lidar":
    points = torch.from_numpy(parser.points).float()
    rgbs   = torch.from_numpy(parser.points_rgb / 255.0).float()
elif init_type == "random":
    points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
    rgbs   = torch.rand((init_num_pts, 3))
```

| Gaussian 파라미터 | SfM에서 오는 값 | 변환 |
|---|---|---|
| `means` [N,3] | `points` | 그대로 |
| `scales` [N,3] | `points`의 이웃 밀도 | `log(knn(3) 평균거리 * init_scale)` — 이웃이 멀면 큰 Gaussian |
| `sh0` (DC) | `points_rgb / 255` | `rgb_to_sh(rgbs)`; 고차 SH 계수는 0 |
| `quats` [N,4] | (SfM 무관) | 랜덤 후 내부 normalize |
| `opacities` [N] | (SfM 무관) | `logit(init_opa=0.1)` |

즉 SfM 포인트는 **위치·색·"대략적 크기"까지 세 가지를 동시에 제공**한다. 크기는 포인트가 직접 주는 게 아니라 포인트 *분포*(3-최근접 이웃 거리)에서 유도된다는 점이 포인트다.

`init_type="random"`으로도 학습은 되지만, SfM 초기화가 훨씬 빨리 수렴하고 최종 품질도 좋다 — 3DGS의 밀도화(densification)는 **이미 Gaussian이 있는 곳 주변만** 늘리고 쪼개므로, 초기 포인트가 없는 영역은 씬을 덮기 어렵다. 워크스루 L290의 표현대로 "SfM 포인트만으로는 씬을 다 덮지 못하므로 학습 중에 Gaussian을 늘리고 정리한다" — 밀도화는 SfM 초기화를 **보완**하는 것이고, 대체하는 게 아니다.

---

## 4. 세 가지 외에 `Parser`가 같이 챙기는 것들

카드의 "세 가지"는 학습에 직접 먹히는 축이고, 실무적으로는 아래도 같이 나온다:

- `image_paths` / `image_names` — 다운샘플 폴더(`images_4`, `images_8`)와 COLMAP이 쓴 원본 파일명을 정렬 순서로 짝지어 매핑한다(`colmap.py:232~240`). `.jpg` 원본이면 필요 시 PNG로 리사이즈 폴더를 생성(`_resize_image_folder`).
- `scene_scale` — 위 1번 참조. lr·임계값의 단위.
- `transform` [4,4] — 정규화에 쓴 누적 변환. 나중에 원래 COLMAP 좌표계로 되돌릴 때 필요.
- `camera_indices` / `num_cameras` — camera_id를 0-based 연속 인덱스로 바꾼 것. appearance/camera embedding용.
- `exposure_values` — `load_exposure=True`일 때 EXIF에서 EV를 읽고 평균을 빼서 상대 노출로 저장.
- `bounds` — `poses_bounds.npy`가 있으면 로드(forward-facing 씬용).

**train/val 분리**는 `Parser`가 아니라 `Dataset`이 한다. 규칙은 아주 단순하다 (`colmap.py:456~461`):

```python
indices = np.arange(len(parser.image_names))
train: indices[indices % test_every != 0]
val  : indices[indices % test_every == 0]
```

---

## 5. 기억용 한 줄 정리

> **포즈(외부) = "어디서 봤나", K+왜곡(내부) = "어떤 렌즈로 봤나", sparse 포인트+RGB = "무엇이 있었나(의 첫 추측)".**
> 앞의 둘은 학습 중 **고정된 관측 조건**이고, 마지막 하나만 **최적화의 출발점**이다.

## 6. 흔한 함정

- `Parser.Ks`라는 속성은 없다 — `Ks_dict[camera_id]`다. `Ks` [3,3]/[B,3,3]는 `Dataset` 출력(`data["K"]`)과 `rasterization` 인자 이름이다.
- `Ks_dict`/`imsize_dict`는 undistort 이후 값으로 **덮어써진다.** undistort 전 K를 기대하면 안 된다.
- `init_type="random"`이라도 `Parser`는 여전히 필요하다 — `scene_scale`이 랜덤 포인트의 범위(`init_extent * scene_scale`)와 lr을 정하기 때문.
- `normalize=True`는 포즈와 포인트에 **같은** 변환을 적용한다. 한쪽만 변환하면 씬이 깨진다.
- `factor > 1`이면 `images_{factor}` 폴더가 실제로 있어야 하고, 없으면 `ValueError`. 반대로 COLMAP intrinsic과 실제 파일 해상도가 다르면 `Parser`가 K를 자동 재보정한다는 점도 기억할 만하다.
