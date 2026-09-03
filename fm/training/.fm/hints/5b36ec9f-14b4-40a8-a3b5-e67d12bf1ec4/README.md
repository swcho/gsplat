# `means`의 학습률에 `scene_scale`을 곱하는 이유

**Q.** `means`의 학습률에 `scene_scale`을 곱하는 이유는?

**A.** 위치 이동량이 씬의 물리적 크기에 비례해야 하기 때문이다. 이렇게 하면 씬 크기에 무관하게 동일한 학습 거동을 얻는다.

---

## 1. 코드에서 어디에 나오나

워크스루의 `init_splats_with_optimizers`에서 파라미터별 학습률을 만드는 부분:

```python
lrs = {
    "means":     1.6e-4 * scene_scale,  # 위치는 씬 크기에 비례
    "scales":    5e-3,
    "quats":     1e-3,
    "opacities": 5e-2,
    "sh0":       2.5e-3,
    "shN":       2.5e-3 / 20,            # 고차 SH는 천천히
}
```

원본은 `examples/simple_trainer.py:336`이고, 딱 `means`만 `scene_scale`이 붙는다.

```python
("means",  torch.nn.Parameter(points), means_lr * scene_scale),   # ← 여기만
("scales", torch.nn.Parameter(scales), scales_lr),
("quats",  torch.nn.Parameter(quats),  quats_lr),
("opacities", torch.nn.Parameter(opacities), opacities_lr),
```

`scene_scale` 자체는 데이터셋 파서가 **카메라 배치의 반경**으로 계산한다 (`examples/datasets/colmap.py:436-440`).

```python
# size of the scene measured by cameras
camera_locations = camtoworlds[:, :3, 3]
scene_center = np.mean(camera_locations, axis=0)
dists = np.linalg.norm(camera_locations - scene_center, axis=1)
self.scene_scale = np.max(dists)          # 카메라 중심에서 가장 먼 카메라까지의 거리
```

그리고 트레이너에서 여유분 1.1배와 사용자 배율을 곱해 최종값으로 쓴다 (`simple_trainer.py:458`, 워크스루 106행도 동일).

```python
self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
```

즉 `scene_scale`은 **"이 씬의 1 단위 길이"** 를 나타내는 물리적 스케일이다.

## 2. 핵심 논리: `means`만 길이 차원(dimension)을 가진다

Gaussian의 5종류 파라미터를 차원 관점에서 보면 `means`가 유일하게 특별하다.

| 파라미터 | 저장 공간 | 차원 | 씬을 $s$배 확대하면 |
|---|---|---|---|
| `means` | 월드 좌표 그대로 | **길이 \[L\]** | $\mu \to s\,\mu$ (값이 같이 커진다) |
| `scales` | log-space | 길이의 로그 | $\log\sigma \to \log\sigma + \log s$ (**덧셈 상수만 이동**) |
| `quats` | 미정규화 quaternion | 무차원(방향) | 불변 |
| `opacities` | logit-space | 무차원(확률) | 불변 |
| `sh0`/`shN` | SH 계수 | 무차원(색) | 불변 |

- `scales`는 로그 공간에 저장되므로 씬을 100배 키워도 값은 `+log 100 ≈ +4.6`만 평행이동한다. *상대적* 크기 변화를 만드는 데 필요한 로그 증분은 씬 크기와 무관하므로 학습률도 상수여야 한다.
- 반면 `means`는 값 자체가 미터/월드유닛이다. "이 Gaussian을 조금 옆으로 옮긴다"라는 **같은 의미의 이동**이 씬 크기에 따라 다른 숫자를 요구한다. 방 하나짜리 씬에서 0.001은 유의미한 이동이지만, 도시 규모 씬에서 0.001은 아무 일도 일어나지 않는 값이다.

따라서 `means`의 학습률은 무차원 상수가 아니라 **길이 차원을 가진 양**이고, 그 길이 단위를 씬에서 가져와야 한다 — 그게 `scene_scale`이다. `1.6e-4`는 "씬 반경의 약 0.016%"라는 **상대 스텝 크기**로 읽어야 한다.

## 3. Adam이 이 논리를 정확히 성립시킨다

이 트릭이 "대략 맞는 근사"가 아니라 거의 정확하게 성립하는 이유는 옵티마이저가 Adam이기 때문이다. Adam의 갱신량은

$$\Delta\theta = -\eta \cdot \frac{\hat m}{\sqrt{\hat v}+\epsilon},\qquad \left|\frac{\hat m}{\sqrt{\hat v}}\right| \approx O(1)$$

로, gradient의 **크기가 아니라 부호·일관성만** 반영하는 정규화된 스텝이다. 즉 **한 스텝의 실제 이동 거리 ≈ 학습률 $\eta$** 그 자체다. `means`의 lr은 "gradient에 곱하는 계수"가 아니라 **"매 스텝 Gaussian이 월드 좌표에서 몇 미터 움직일지"** 를 직접 지정하는 값인 셈이다. 그래서 씬 크기에 비례해야 한다는 요구가 그대로 lr에 걸린다.

조금 더 형식적으로: 월드를 $s$배 확대한 씬($\mu' = s\mu$)에서 동일한 이미지를 렌더링하면 손실은 같으므로 $\nabla_{\mu'}L = \frac{1}{s}\nabla_\mu L$이다. Adam은 $\hat m/\sqrt{\hat v}$에서 이 $1/s$ 인자를 **약분해 버린다** — 그러므로 lr을 그대로 두면 확대된 씬에서 Gaussian은 원래 씬과 **똑같은 절대 거리**만 움직인다. 즉 상대적으로는 $s$배 느려진다. lr에 $s$를 곱해 주면 갱신량도 $s$배가 되어 학습 궤적이 원본 궤적의 정확한 $s$배 확대판이 된다 → **학습이 스케일 등변(scale-equivariant)** 이 된다.

(SGD였다면 gradient가 이미 $1/s$로 줄어들므로 보정 방향이 반대가 된다. `scene_scale` 곱셈은 Adam 특유의 "정규화된 스텝" 성질과 짝을 이루는 처방이다.)

## 4. 곱하지 않으면 무슨 일이 생기나

| 씬 | `scene_scale` | 실효 lr (`1.6e-4 × scene_scale`) | 곱하지 않았다면 |
|---|---|---|---|
| 작은 오브젝트 스캔 | ≈ 0.5 | 8e-5 | lr이 씬 대비 과도 → 위치가 흔들리고 발산·아티팩트 |
| 일반 mipnerf360 씬 | ≈ 4~5 | 6~8e-4 | (기준값이 튜닝된 지점) |
| 대규모 항공/도시 씬 | ≈ 100+ | 1.6e-2 | 스텝이 씬 대비 1/20 이하 → 위치가 SfM 초기값에서 거의 못 벗어나 30k 스텝으로 수렴 불가 |

요약하면, 곱하지 않으면 **하이퍼파라미터를 씬마다 다시 튜닝**해야 한다. 곱해 두면 `1.6e-4`라는 하나의 기본값이 온갖 규모의 씬에서 통한다 — 이것이 "씬 크기에 무관하게 동일한 학습 거동"의 실제 의미다.

## 5. 같은 원리가 적용되는 다른 곳들

`scene_scale`은 lr뿐 아니라 **길이 차원을 가진 모든 하이퍼파라미터**의 기준 단위다. 워크스루 300-302행의 densification 표를 보면:

| 조건 | 임계값 | `scene_scale` 곱하나? |
|---|---|---|
| duplicate / split 구분 | 3D 크기 ≤ / > `0.01 × scene_scale` (`grow_scale3d`) | **곱한다** — 길이 |
| prune (비대한 Gaussian) | 3D 크기 > `0.1 × scene_scale` (`prune_scale3d`) | **곱한다** — 길이 |
| grow 판정 | 화면 grad 평균 > `2e-4` (`grow_grad2d`) | **안 곱한다** — 정규화된 화면(NDC) 좌표계 양 |
| prune (기여 없음) | opacity < `0.005` | **안 곱한다** — 무차원 |

구현은 `gsplat/strategy/default.py`에서 `initialize_state(scene_scale=...)`로 값을 상태에 저장해 두고(`default.py:116-127`), 판정할 때 `self.grow_scale3d * state["scene_scale"]`처럼 곱해 쓴다(`default.py:301`, `355`). 워크스루 313행의 `strategy.initialize_state(scene_scale=scene_scale)`가 바로 이 값을 넘기는 곳이다.

`init_type="random"`일 때 초기 포인트를 뿌리는 범위도 같다 — `init_extent * scene_scale * (rand*2-1)` (`simple_trainer.py:315`).

## 6. 함께 기억할 lr 관련 디테일

- **`means`만 스케줄러가 붙는다** (`simple_trainer.py:808-812`). `ExponentialLR(gamma=0.01**(1/max_steps))`로 학습 종료 시 초기값의 **1%** 까지 감쇠한다. 즉 실효 lr은 `1.6e-4 × scene_scale` → `1.6e-6 × scene_scale`으로 줄어들며, 초반에는 크게 움직여 구조를 잡고 후반에는 미세 조정만 한다. 다른 파라미터는 상수 lr.
- **배치 크기 보정**: 최종 lr은 다시 `lr * math.sqrt(BS)`로 곱해진다 (`BS = batch_size * world_size`). SDE 스케일링 규칙에 따른 보정으로, `scene_scale` 보정과는 독립적인 축이다.
- **`normalize=True`와의 관계**: 파서가 카메라 기준 similarity 변환으로 월드를 정규화하더라도(`similarity_from_cameras` + `align_principal_axes`) 씬 크기가 정확히 1이 되지는 않는다. 그래서 정규화 *이후*의 카메라 분포로 `scene_scale`을 다시 재고, lr·임계값에 곱한다. "씬을 단위 크기로 리스케일한다" 대신 "하이퍼파라미터를 씬 단위로 표현한다"를 택한 것이고, 두 방식은 수학적으로 등가지만 후자가 카메라 포즈·깊이 등 다른 데이터를 건드리지 않아 덜 침습적이다.
- **`cfg.global_scale`** 은 사용자가 이 기준 단위를 통째로 키우거나 줄여 densification 강도와 위치 lr을 한 번에 조절할 수 있는 손잡이다.
- MCMC 전략(`gsplat/strategy/mcmc.py`)은 위치를 gradient로만 움직이지 않고 relocate/노이즈 주입을 쓰지만, 그 노이즈 크기도 Gaussian의 실제 공분산에서 파생되므로 같은 "스케일에 비례" 정신을 따른다.

## 7. 한 줄 정리

`means`는 파라미터 중 유일하게 **월드 길이 단위**를 갖고, Adam의 스텝 크기는 사실상 lr 값 자체다. 따라서 lr도 길이 단위여야 하며, 그 단위를 씬에서 가져오는 것이 `× scene_scale`이다. `1.6e-4`는 절대 거리가 아니라 **씬 반경에 대한 상대 스텝 비율**로 읽어야 한다.
