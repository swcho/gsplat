# `meta["means2d"]` — 렌더 중간값이자 밀도화의 유일한 신호원

> **Q.** `meta["means2d"]`가 학습에서 특별히 중요한 이유는?
>
> **A.** 투영된 화면 좌표(px)인데, 이 텐서의 **grad 크기**가 밀도화(densification) 시
> split/duplicate 판정 기준이 되기 때문이다. `DefaultStrategy`가 이 값을 사용한다.

---

## 1. means2d는 원래 무엇인가

래스터화 파이프라인 ③단계(원근 투영, EWA)의 산출물이다. 워크스루의 개요도에서:

```
▼  ② 카메라 좌표     μ_c = R μ + t,  Σ_c = R Σ Rᵀ          ┐
▼  ③ 원근 투영(EWA)  Σ₂ = J Σ_c Jᵀ + eps2d·I              ├ fully_fused_projection
     → means2d, conics(=Σ₂⁻¹), radii, depths, 절두체 컬링 ┘
```

| 키 | 모양 | 뜻 |
|---|---|---|
| `means2d` | `[C,N,2]` (packed면 `[nnz,2]`) | 투영된 Gaussian 중심, **픽셀 단위** |

수식으로는 `means2d = (fx·x/z + cx, fy·y/z + cy)`, 즉 카메라 좌표 μ_c의 핀홀 투영 π(μ_c)다.
⑦ 알파 블렌딩에서 픽셀 중심(px+0.5)과의 차 `dx, dy`를 만드는 데 쓰이고, 그것이
σ = ½(a·dx² + c·dy²) + b·dx·dy → α로 이어진다.

여기까지만 보면 means2d는 그저 "지나가는 중간 텐서"다. `radii`, `conics`, `depths`와
동급으로 `meta` 딕셔너리에 담겨 있을 뿐이다. **그런데 딱 하나, 이 텐서만 학습 알고리즘이
값이 아니라 `.grad`를 읽어 간다.** 그래서 특별하다.

---

## 2. 원 3DGS 논문의 밀도화 휴리스틱

3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023) 4.1절 "Adaptive Control of Gaussians"는
"기하가 부족한 곳"을 두 가지로 나눈다.

- **under-reconstruction (과소 복원)** — 있어야 할 곳에 Gaussian이 아예 부족하다.
  남아 있는 Gaussian이 작고, 자기가 못 덮는 영역을 향해 계속 끌려간다.
- **over-reconstruction (과대 복원)** — Gaussian 하나가 너무 넓은 영역을 뭉개서 덮고 있다.
  디테일을 못 내니 역시 계속 흔들린다.

논문은 두 경우 모두 **"view-space positional gradient의 평균 크기가 임계값 τ_pos를 넘는다"**
는 공통 증상을 보인다고 관찰하고, 이를 하나의 판정식으로 삼는다.

> 논문 값: **τ_pos = 0.0002**. gsplat의 `DefaultStrategy.grow_grad2d` 기본값이 정확히 `0.0002`다.

그리고 둘을 **3D 스케일 크기**로 갈라 처리한다.

| 증상 | 크기 조건 | 조치 |
|---|---|---|
| under-reconstruction | 작다 | **clone(duplicate)** — 똑같이 복제하고 gradient 방향으로 이동 |
| over-reconstruction | 크다 | **split** — 2개로 쪼개고 스케일을 φ=1.6으로 나눔 |

핵심은 "**한 개의 스칼라 신호(화면 grad 평균) + 크기 한 개**로 늘릴지/쪼갤지를 결정한다"는
것이다. 그 스칼라 신호가 바로 means2d의 grad다.

---

## 3. 왜 3D `means`가 아니라 2D 화면 grad인가

`means`(3D 위치)에도 grad가 흐른다. 그런데도 굳이 means2d를 쓰는 이유:

1. **단위가 뷰-불변이다.** 3D grad의 크기는 씬의 물리 스케일(미터냐 임의 단위냐)과
   카메라까지의 거리에 정비례해 달라진다. "0.0002"라는 하나의 임계값을 씬마다 다시
   튜닝하지 않고 쓰려면, 스케일이 씬에 안 묶인 양이어야 한다. 화면 좌표는 이미지 해상도로
   정규화하면 `[-1,1]`이라는 절대 기준을 갖는다.
2. **오차가 정의된 곳이 화면이다.** 손실은 렌더된 픽셀에서 계산된다. "이 Gaussian을 화면에서
   어느 쪽으로 얼마나 밀면 손실이 줄어드는가"가 곧 "이 위치의 재구성이 얼마나 모자라는가"의
   직접적 척도다. 3D grad는 여기에 카메라 자코비안 J와 깊이 1/z가 한 번 더 곱해진 값이라
   같은 화면 오차라도 먼 물체는 작게, 가까운 물체는 크게 나온다.
3. **깊이 방향 성분이 섞이지 않는다.** 3D grad는 시선 방향(z) 성분을 포함하는데, 깊이 모호성
   때문에 이 성분은 노이즈가 크고 "디테일 부족" 신호와 무관하다. means2d grad는 화면 평면
   2성분만이라 신호가 깨끗하다.
4. **원근·컬링이 이미 반영돼 있다.** 화면 밖이거나 너무 작아 `radii=0`으로 컬링된 항목은
   애초에 통계에 안 들어간다(4절 참고).

요약: means2d는 "**손실이 정의된 좌표계에서의 위치**"이고, 그 grad는 씬 스케일과 무관한
정규화 가능한 단위를 갖는다. 그래서 하드코딩된 상수 하나로 전 씬에 통하는 휴리스틱이 된다.

---

## 4. `DefaultStrategy`는 grad를 어떻게 누적하나

`gsplat/strategy/default.py`. 학습 루프는 매 스텝 훅 두 개를 부른다.

```python
render, alpha, info = rasterization(...)
strategy.step_pre_backward(params, optimizers, state, step, info)   # ← retain_grad
loss = ...
loss.backward()
strategy.step_post_backward(params, optimizers, state, step, info)  # ← 누적 + refine
```

### (a) `step_pre_backward` — `retain_grad()`

```python
assert self.key_for_gradient in info, "The 2D means of the Gaussians is required but missing."
info[self.key_for_gradient].retain_grad()          # default.py:170
```

`key_for_gradient`는 기본 `"means2d"` (2DGS 백엔드에서는 `"gradient_2dgs"`).

### (b) `_update_state` — 정규화 후 두 버퍼에 누적

상태는 초기화 시 딱 두 개의 러닝 버퍼로 시작한다 (`initialize_state`).

```python
state = {"grad2d": None, "count": None, "scene_scale": scene_scale}
# - grad2d: 각 GS의 화면 grad 노름의 러닝 합
# - count : 각 GS가 몇 번 보였는지의 러닝 합
```

`step_post_backward` 안에서:

```python
grads = info["means2d"].grad.clone()               # (absgrad면 .absgrad)
# normalize grads to [-1, 1] screen space
grads[..., 0] *= info["width"]  / 2.0 * info["n_cameras"]
grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
```

- `× W/2`, `× H/2`: gsplat의 means2d는 **픽셀** 단위인데 논문 임계값은 NDC `[-1,1]` 기준이다.
  연쇄법칙으로 `∂L/∂ndc_x = ∂L/∂px_x · (W/2)` 이므로 곱해 주면 논문과 같은 축척이 된다.
  (해상도가 달라도 임계값이 그대로 통하는 이유이기도 하다.)
- `× n_cameras`: 한 번에 C개 뷰를 렌더하면 손실이 카메라 수로 평균되어 grad가 1/C로 줄어든다.
  이를 되돌려 "단일 뷰 기준"으로 맞춘다.

그다음 **가시 항목만 골라** 인덱스별로 더한다.

```python
if packed:                                   # grads: [nnz, 2]
    gs_ids = info["gaussian_ids"]
else:                                        # grads: [C, N, 2]
    sel    = (info["radii"] > 0.0).all(dim=-1)   # 컬링 안 된 (카메라, GS) 쌍
    gs_ids = torch.where(sel)[1]
    grads  = grads[sel]

state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))     # ‖∇‖를 더한다
state["count"] .index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))
```

여기서 놓치기 쉬운 두 가지:

- **노름을 먼저 취하고 더한다.** 평균 벡터의 노름이 아니라 **노름의 평균**이다.
  왼쪽 뷰는 +x, 오른쪽 뷰는 −x로 끌어당기는 상황에서 벡터 평균을 냈다면 0이 되어
  "잘 맞춰졌다"고 오판했을 것이다. 논문의 "average magnitude of view-space position
  gradients"가 이 뜻이다.
- **분모가 `count`다.** 스텝 수가 아니라 **그 Gaussian이 실제로 보인 (카메라, 스텝) 횟수**다.
  가끔만 보이는 Gaussian이 불리해지지 않는다.

### (c) `_grow_gs` — split과 duplicate를 가르는 지점

```python
count = state["count"]
grads = state["grad2d"] / count.clamp_min(1)          # ← 평균 화면 grad

is_grad_high = grads > self.grow_grad2d               # τ_pos = 0.0002
is_small = torch.exp(params["scales"]).max(dim=-1).values <= self.grow_scale3d * state["scene_scale"]

is_dupli = is_grad_high & is_small                    # 작다 → 복제
is_large = ~is_small
is_split = is_grad_high & is_large                    # 크다 → 분할

if step < self.refine_scale2d_stop_iter:              # 선택 기능(기본 off)
    is_split |= state["radii"] > self.grow_scale2d    # 화면에서 너무 큰 것도 분할
```

즉 **grad 임계값이 "손볼 대상인가"를 정하고, 스케일이 "어떻게 손볼 것인가"를 정한다.**
`grow_scale3d = 0.01` → 씬 스케일의 1%가 기준선.

정리하면:

| grad 평균 | 3D 스케일 | 결과 |
|---|---|---|
| > 0.0002 | ≤ 1%·scene_scale | **duplicate** (`ops.duplicate`) |
| > 0.0002 | > 1%·scene_scale | **split** (`ops.split`) |
| ≤ 0.0002 | — | 그대로 둠 |

`gsplat/strategy/ops.py`의 실제 동작:

- `duplicate` (ops.py:141) — 선택된 파라미터를 `torch.cat([p, p[sel]])`로 그냥 이어 붙인다.
  Adam의 exp_avg/exp_avg_sq는 0으로 채운다. `grad2d`/`count` 상태도 같이 복제된다.
- `split` (ops.py:175) — 원본 Gaussian의 3D 공분산을 PDF로 삼아 두 점을 샘플링하고
  (`rotmats @ diag(scales) @ randn`), `means`는 그 오프셋만큼 이동, `scales`는 **`log(scale/1.6)`**,
  나머지는 그대로 복사한 뒤 원본은 제거한다. `revised_opacity=True`면
  `α' = 1 − √(1−α)` (arXiv:2404.06109)로 불투명도까지 보정한다.
- duplicate로 새로 생긴 것은 같은 스텝에서 split 대상이 되지 않도록 마스크에 0을 이어 붙인다.

refine이 끝나면 두 버퍼를 리셋한다 — 즉 grad2d/count는 **refine 주기(기본 100스텝) 동안의
누적치**다.

```python
state["grad2d"].zero_()
state["count"].zero_()
```

---

## 5. `retain_grad()`가 왜 필요한가

PyTorch는 **leaf tensor**(사용자가 만든 `requires_grad=True` 파라미터)에만 `.grad`를 채운다.
중간(non-leaf) 텐서의 grad는 역전파 중에 다음 노드로 흘려보낸 뒤 **즉시 버린다**. 메모리
낭비를 막기 위한 기본 동작이다.

means2d는 `fully_fused_projection`의 출력, 즉 전형적인 non-leaf다. 그대로 두면
`loss.backward()` 후 `info["means2d"].grad`는 `None`이다. `retain_grad()`는
"이 노드의 grad는 버리지 말고 `.grad`에 남겨 달라"는 요청이다.

세 가지 부수 조건:

- **backward 전에** 불러야 한다. 그래서 `step_pre_backward`라는 별도 훅이 존재한다.
  backward 이후에 부르면 이미 버려진 뒤라 소용없다.
- **매 스텝** 불러야 한다. means2d는 스텝마다 새로 만들어지는 새 텐서다.
- 렌더 출력이 `requires_grad=False`인 상황(예: Gaussian을 얼려 두고 다른 모듈만 학습)에서는
  `retain_grad()`가 무효이자 불필요하다. `simple_trainer.py`가 이 경우 훅 호출 자체를 건너뛴다.

바꿔 말하면, means2d의 grad는 **렌더러가 반환값으로 주는 것이 아니라 전략이 스스로 붙잡아
두는 것**이다. `meta`에 means2d를 굳이 실어 보내는 이유의 절반이 여기에 있다.

---

## 6. `absgrad` 변형 (AbsGS)

`.grad`에는 여전히 상쇄 문제가 남아 있다. **한 뷰 안에서도** 큰 Gaussian이 여러 픽셀에
걸쳐 있을 때, 왼쪽 픽셀들은 "왼쪽으로 가라", 오른쪽 픽셀들은 "오른쪽으로 가라"고 말한다.
backward 커널이 이 기여들을 `atomicAdd`로 합치면 서로 지워져 **합이 거의 0**이 된다.
결과: 명백히 뭉개진 큰 Gaussian이 `grads ≤ τ_pos`로 판정되어 끝까지 분할되지 않는다.
이것이 3DGS의 대표적인 디테일 손실 원인이다.

**AbsGS** (arXiv:2404.10484)는 합치기 **전에** 절댓값을 취한다. CUDA backward 커널
(`RasterizeToPixels3DGSSerialBatchBwd.cu`)이 두 버퍼를 동시에 누적한다.

```cpp
v_xy_local     = { v_sigma * (conic.x*delta.x + conic.y*delta.y),
                   v_sigma * (conic.y*delta.x + conic.z*delta.y) };
if (v_means2d_abs != nullptr)
    v_xy_abs_local = { abs(v_xy_local.x), abs(v_xy_local.y) };   // ← 부호를 죽인다
...
atomicAdd_system(v_xy_ptr,     v_xy_local.x);          // → means2d.grad
atomicAdd_system(v_xy_abs_ptr, v_xy_abs_local.x);      // → means2d.absgrad
```

파이썬 쪽에서는 별도 텐서로 붙여 준다.

```python
if absgrad and not with_eval3d:
    means2d.absgrad = means2d_absgrad      # rendering.py:653
```

그래서 사용법이 **두 군데를 함께** 켜야 하는 형태가 된다.

```python
rasterization(..., absgrad=True)                       # 커널이 절댓값 합을 계산하게
DefaultStrategy(absgrad=True, grow_grad2d=0.0008)      # 전략이 .absgrad를 읽게
```

- `absgrad=True`인데 `rasterization(absgrad=True)`를 빼먹으면 `.absgrad` 속성이 없어 실패한다.
- 상쇄가 없어져 값 자체가 커지므로 **임계값도 같이 올려야 한다** — 권장 `0.0008` (기본의 4배).
  안 올리면 거의 모든 Gaussian이 임계값을 넘어 폭발적으로 분열한다.
- 대개 디테일이 살아나고 결과가 좋아진다. 대신 메모리·연산이 조금 늘고, `with_eval3d` 경로
  등 일부 렌더 모드에서는 지원되지 않는다.

`_update_state`에서 두 경로가 갈리는 지점은 단 두 줄이다.

```python
if self.absgrad:
    grads = info[self.key_for_gradient].absgrad.clone()
else:
    grads = info[self.key_for_gradient].grad.clone()
```

이후 정규화·누적·판정은 완전히 동일하다.

---

## 7. 한 장 요약

```
rasterization()
  └ ③ 투영 → means2d [C,N,2] (px)                       ← non-leaf 중간 텐서
        │
        ├─ forward: dx,dy = means2d − 픽셀중심 → σ → α → 색
        │
   step_pre_backward:  means2d.retain_grad()             ← 안 하면 .grad = None
        │
   loss.backward()
        │  CUDA bwd: 픽셀별 ∂L/∂means2d를 warpSum + atomicAdd
        │            (absgrad면 abs() 후 별도 버퍼에도 누적)
        ▼
   step_post_backward → _update_state
        grads = means2d.grad (또는 .absgrad)
        grads[...,0] *= W/2 * C ;  grads[...,1] *= H/2 * C    ← NDC [-1,1] 정규화
        grad2d[gid] += ‖grads‖  ;  count[gid] += 1            ← radii>0인 것만
        ▼
   (100스텝마다) _grow_gs
        g = grad2d / count
        g > 0.0002 ?
           ├ scale ≤ 1%·scene_scale → duplicate
           └ scale >  1%·scene_scale → split (2개, scale/1.6)
        → grad2d, count 리셋
```

**자주 헷갈리는 점**

- `means2d.grad`는 파라미터 grad가 아니다. Adam이 업데이트하는 것은 3D `means`이고,
  means2d의 grad는 **오직 밀도화 판정에만** 쓰인다. 두 용도가 완전히 분리돼 있다.
- 값(`means2d` 자체)은 forward에서, grad는 densification에서 쓰인다. 이 텐서만 양쪽에 걸쳐 있다.
- `radii`도 `meta`에 있어야 한다 — 어떤 (카메라, Gaussian) 쌍이 실제로 보였는지 골라
  `count`를 세는 데 필요하다. `grad2d` 하나만으로는 평균을 낼 수 없다.
- MCMC 전략(`gsplat/strategy/mcmc.py`)은 이 신호를 전혀 쓰지 않는다. means2d grad 기반
  판정은 `DefaultStrategy`(그리고 2DGS의 `gradient_2dgs`) 계열의 특징이다.

---

## 참고

- Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*,
  SIGGRAPH 2023 — arXiv:2308.04079 (4.1절 Adaptive Density Control, τ_pos = 0.0002, φ = 1.6)
- Ye et al., *AbsGS: Recovering Fine Details for 3D Gaussian Splatting* — arXiv:2404.10484
- Revised opacity for split — arXiv:2404.06109
- 코드: `gsplat/strategy/default.py` (`_update_state`, `_grow_gs`),
  `gsplat/strategy/ops.py` (`duplicate`, `split`),
  `gsplat/rendering.py` (`means2d.absgrad`),
  `gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchBwd.cu` (grad 누적)
