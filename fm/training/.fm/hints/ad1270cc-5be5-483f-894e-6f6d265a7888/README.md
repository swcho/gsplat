# `rasterization()`이 반환하는 `info` dict의 역할

## 카드 내용

> **Q.** `rasterization()`이 반환하는 `info` dict의 역할은?
> **A.** 밀도화 전략의 입력이 된다. 특히 `info["means2d"]`는 화면공간 위치 텐서로, 그 gradient가 "이 Gaussian을 더 쪼개야 하는가"의 신호가 된다.

## 한 줄 요약

`rasterization()`은 3개를 돌려준다 — `render_colors`, `render_alphas`, 그리고 `meta`(워크스루에서는 `info`로 받는다). 앞의 둘은 **손실 계산용**, 세 번째는 **밀도화(densification) 전략용**이다. `info`는 "렌더 결과"가 아니라 렌더 도중에 계산된 **중간 산물(intermediate results)의 창구**다.

```python
renders, alphas, info = rasterize_splats(splats, camtoworlds, Ks, width, height, sh_degree=...)

strategy.step_pre_backward(splats, optimizers, strategy_state, step, info)   # ← info
loss = torch.lerp(l1, ssim, 0.2); loss.backward()
for opt in optimizers.values(): opt.step(); opt.zero_grad(set_to_none=True)
strategy.step_post_backward(splats, optimizers, strategy_state, step, info, packed=False)  # ← info
```

손실은 `renders`만 쓴다. `info`는 **오직** 두 개의 전략 훅에만 들어간다.

## `info`에 실제로 뭐가 들어 있나

`gsplat/rendering.py`의 `rasterization()` 끝부분(`meta = {...}`)에서 조립되며, 앞서 설명한 4단계 CUDA 파이프라인 각 단계의 부산물이 그대로 노출된다.

| 키 | shape (packed=False) | 나온 단계 | 무엇 |
|---|---|---|---|
| `means2d` | `[C, N, 2]` | 투영 | 화면(픽셀) 좌표. **핵심** |
| `radii` | `[C, N, 2]` | 투영 | 화면상 반경. `0`이면 컬링(안 보임) |
| `depths` | `[C, N]` | 투영 | 카메라 좌표계 깊이 |
| `conics` | `[C, N, 3]` | 투영 | 2D 공분산의 역행렬 상삼각 성분 |
| `opacities` | `[C, N]` | 투영 | 투영 후 opacity |
| `batch_ids` / `camera_ids` / `gaussian_ids` | `[nnz]` 또는 `None` | 투영 | packed 모드의 희소 인덱스 |
| `tile_width` / `tile_height` / `tiles_per_gauss` | 스칼라·`[C,N]` | 타일 교차 | 16×16 타일 그리드 정보 |
| `isect_ids` / `flatten_ids` / `isect_offsets` | — | 타일 교차 | 정렬된 (tile, depth) 키와 오프셋 |
| `width` / `height` / `tile_size` / `n_batches` / `n_cameras` | 스칼라 | — | 렌더 해상도·배치 메타 |

옵션에 따라 `render_extra_signals`, `normals`도 추가된다.

즉 대부분은 **디버깅·시각화·커스텀 후처리용 배관 노출**이고, 학습 로직이 실제로 소비하는 것은 소수다. 워크스루 3단계가 딱 이 용도로 찍어 본다.

```python
print("info keys:", sorted(info.keys()))
print("means2d:", tuple(info["means2d"].shape), "| radii:", tuple(info["radii"].shape))
print(f"화면에 보이는 Gaussian: {(info['radii'] > 0).all(-1).sum().item():,} / {len(splats['means']):,}")
```

`DefaultStrategy._update_state()`가 요구하는 키는 정확히 이 6개다 (없으면 `assert`로 죽는다).

```python
for key in ["width", "height", "n_cameras", "radii", "gaussian_ids", self.key_for_gradient]:
    assert key in info, f"{key} is required but missing."
```

## 왜 하필 `means2d`의 gradient가 "쪼개라" 신호인가

`means2d[c, i]`는 카메라 `c`에서 Gaussian `i`의 **화면 중심 위치**다. 여기에 대한 gradient는

$$\frac{\partial \mathcal{L}}{\partial \text{means2d}_i} = \text{“이 Gaussian을 화면에서 어느 방향으로 얼마나 밀면 손실이 줄어드는가”}$$

이다. 이 벡터의 **크기(norm)가 크다**는 것은 해당 Gaussian이 담당하는 화면 영역이 아직 **잘 재현되지 않아, 렌더러가 그 blob을 계속 끌어당기고 있다**는 뜻이다. 그런데 blob 하나는 타원 하나밖에 표현하지 못하므로, 밀어서 해결되지 않는 오차는 **표현력 부족**이다 → 개수를 늘려라(duplicate/split).

원논문의 통찰이 바로 이것이고, 3DGS 밀도화가 opacity나 loss map이 아니라 **화면공간 위치 gradient**를 기준으로 삼는 이유다.

```
∂L/∂means2d 큼  ─┬─ 크기 작음 (≤ 1%·scene_scale) → duplicate : 같은 크기로 복제해 빈틈 채우기
                 └─ 크기 큼   (>  1%·scene_scale) → split     : 2개로 쪼개고 크기 /1.6
```

`DefaultStrategy._grow_gs()`가 그대로 이 분기다.

```python
grads = state["grad2d"] / count.clamp_min(1)          # 보인 횟수로 평균
is_grad_high = grads > self.grow_grad2d               # 기본 2e-4
is_small = torch.exp(params["scales"]).max(-1).values <= self.grow_scale3d * state["scene_scale"]
is_dupli = is_grad_high & is_small
is_split = is_grad_high & ~is_small
```

## 두 훅이 나뉜 이유 — `retain_grad()`

`step_pre_backward()`는 한 줄뿐이다.

```python
info[self.key_for_gradient].retain_grad()    # key_for_gradient = "means2d" (2DGS는 "gradient_2dgs")
```

`means2d`는 `fully_fused_projection`의 출력, 즉 **leaf가 아닌 중간 텐서(non-leaf)** 다. PyTorch는 non-leaf 텐서의 `.grad`를 backward 후 **보관하지 않는다**(메모리 절약). 따라서 `backward()` **전에** `retain_grad()`를 걸어 두지 않으면 `step_post_backward()`에서 `info["means2d"].grad`가 `None`이 되어 밀도화가 불가능하다.

그래서 순서가 강제된다.

```
forward → step_pre_backward(retain_grad)  →  loss.backward()  →  opt.step()  →  step_post_backward(밀도화)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^
          이걸 빼면 .grad == None                                               같은 info 객체를 재사용
```

두 훅에 **같은 `info` 객체**를 넘기는 것도 중요하다. `backward()`가 그 텐서에 `.grad`를 채워 주므로, `info`는 forward와 backward를 잇는 **살아 있는 핸들** 역할을 한다.

## `step_post_backward()`가 `info`로 하는 일

`_update_state()`에서 gradient를 해상도로 정규화한 뒤 Gaussian별 러닝 통계에 누적한다.

```python
grads = info["means2d"].grad.clone()          # absgrad=True면 info["means2d"].absgrad
grads[..., 0] *= info["width"]  / 2.0 * info["n_cameras"]   # 픽셀 → [-1,1] NDC 스케일
grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

sel = (info["radii"] > 0.0).all(dim=-1)       # [C,N] 보이는 것만
gs_ids = torch.where(sel)[1]
grads = grads[sel]                            # [nnz,2]

state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))   # 크기 누적
state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))  # 몇 번 보였나
state["radii"][gs_ids] = torch.maximum(state["radii"][gs_ids],
                                       radii / float(max(info["width"], info["height"])))
```

여기서 `info`의 각 키가 하는 역할이 드러난다.

- `means2d.grad` → 분할 신호 본체
- `width`/`height`/`n_cameras` → 해상도·카메라 수에 무관한 **정규화**. 이 정규화가 있어야 `grow_grad2d=2e-4`라는 하나의 임계값이 모든 해상도에서 통한다
- `radii` → `> 0`인 것만 골라 **보이는 Gaussian만** 통계에 반영(안 보이면 gradient가 0인데, 그걸 평균에 넣으면 신호가 희석된다). 동시에 화면상 크기가 과도한 것(`grow_scale2d=0.05`, `prune_scale2d`)을 판정
- `gaussian_ids` → `packed=True`일 때 희소 텐서를 원래 Gaussian 인덱스로 되돌리는 매핑

`count`로 나누는 이유는 Gaussian마다 **보인 횟수가 다르기** 때문이다. 여러 뷰에서 자주 보인 것과 한 번만 보인 것을 합계로 비교하면 불공평하다.

## `packed` 인자가 얽히는 지점

워크스루의 `rasterize_splats()`가 `packed=False`를 고정한 이유가 여기 있다.

| | `means2d` shape | `info["gaussian_ids"]` | 전략 호출 |
|---|---|---|---|
| `packed=False` | `[C, N, 2]` (조밀) | `None` | `step_post_backward(..., packed=False)` |
| `packed=True` | `[nnz, 2]` (보이는 것만) | `[nnz]` | `step_post_backward(..., packed=True)` |

`_update_state()`가 두 경로를 분기해 처리하므로, **`rasterization()`의 `packed`와 `step_post_backward()`의 `packed`가 일치해야** 한다. 어긋나면 인덱싱이 조용히 틀어진다. 워크스루 주석 `packed=False,  # 밀도화 상태 갱신 코드와 맞춤`이 그 뜻이다.

## `absgrad`가 바꾸는 것

`rasterization(absgrad=True)`를 주면 backward가 `means2d`에 `.grad`와 별도로 `.absgrad` 속성을 붙인다(`gsplat/cuda/_wrapper.py`: `means2d.absgrad = means2d_absgrad`).

- `.grad` — 픽셀별 기여의 **부호 있는 합**. 한 blob이 왼쪽 픽셀에서 +x, 오른쪽에서 −x로 당겨지면 **상쇄되어 0에 가까워진다**. 실제로는 "찢어야 하는" 상태인데 신호가 죽는다
- `.absgrad` — 픽셀별 기여의 **절댓값 합**. 상쇄가 없어 그런 갈등 상태를 잡아낸다(AbsGS, arXiv:2404.10484). 대신 값 자체가 커지므로 임계값을 `grow_grad2d=0.0008` 정도로 올려야 한다

주의: `DefaultStrategy(absgrad=True)`만 켜고 `rasterization(absgrad=True)`를 빼먹으면 `.absgrad` 속성이 없어 터진다. **양쪽을 같이 켜야 한다.**

## 모든 전략이 `info`를 쓰는 것은 아니다

`MCMCStrategy`는 `step_pre_backward()`가 **주석 처리되어 있고**(즉 no-op), `step_post_backward()`도 `info`를 실제로 참조하지 않는다. MCMC는 화면공간 gradient 대신 **opacity 기반 relocation + 위치 노이즈**로 개수를 조절하기 때문이다(대신 `lr`을 인자로 받는다). 즉 `info["means2d"].grad`는 **원논문 계열 밀도화 전용 신호**다.

## 정리

- `info`(= `meta`)는 렌더 파이프라인 중간 산물을 밖으로 내보내는 dict. 손실 계산에는 안 쓰인다
- 학습이 실제로 소비하는 것은 `means2d`, `radii`, `width`, `height`, `n_cameras`, `gaussian_ids`의 6개
- `info["means2d"]`는 non-leaf 텐서이므로 `retain_grad()`가 필수 → 그래서 `pre`/`post` 두 훅으로 쪼개져 있다
- `∂L/∂means2d`의 크기 = 표현력 부족 신호 → 크기가 작으면 duplicate, 크면 split
- `radii > 0`은 가시성 마스크, `width`/`height`/`n_cameras`는 임계값을 해상도 독립으로 만드는 정규화 인자
