# `sh_degree_interval` — SH 차수를 점진적으로 켜는 학습 스케줄

> **Q.** 학습 코드의 `sh_degree_interval`은 무엇을 하는가?
>
> **A.** `sh_degree_to_use = min(step // sh_degree_interval, sh_degree)`로 **그 스텝에서 사용할 SH 차수**를 정한다.
> 기본값 1000이면 처음 1000스텝은 DC(0차)만 쓰고, 이후 1000스텝마다 한 차수씩 더 활성화해 3000스텝부터는
> 최대 차수(기본 3)를 전부 사용한다. 모델 파라미터는 처음부터 16개 계수를 전부 갖고 있고, 바뀌는 것은
> **forward에서 몇 개를 읽어 쓰느냐**뿐이다.

---

## 1. gsplat `examples/simple_trainer.py`의 실제 코드

### 1.1 `Config` 필드

```python
# Degree of spherical harmonics
sh_degree: int = 3
...
# Turn on another SH degree every this steps
sh_degree_interval: int = 1000
```

`sh_degree`는 **최대** 차수(계수 개수 K = (3+1)² = 16), `sh_degree_interval`은 "몇 스텝마다 한 차수를 더 켤지"다.
`--steps_scaler`로 전체 스텝을 줄이면 이 값도 같은 비율로 줄어든다.

```python
def adjust_steps(self, factor: float):
    self.eval_steps = [int(i * factor) for i in self.eval_steps]
    ...
    self.max_steps = int(self.max_steps * factor)
    self.sh_degree_interval = int(self.sh_degree_interval * factor)
```

즉 `--steps_scaler 0.25`(7,500스텝 학습)면 250스텝마다 한 차수씩 켜져, 전체 학습 중 차수 스케줄이 차지하는
**비율**(처음 10%)은 그대로 유지된다.

### 1.2 학습 루프에서의 계산과 `rasterization(sh_degree=...)` 전달

```python
# sh schedule
sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

# forward
renders, alphas, info = self.stage.render(
    self.scene.id,
    camtoworlds=camtoworlds,
    Ks=Ks,
    width=width,
    height=height,
    sh_degree=sh_degree_to_use,      # ← 이 스텝에 쓸 차수
    ...
)
...
desc = f"loss={loss.item():.3f}| sh degree={sh_degree_to_use}| "   # 진행 바에도 표시
```

`self.stage.render(...)`는 내부에서 `gsplat.rasterization(...)`을 호출하며 `sh_degree`를 그대로 넘긴다.
`rasterization`의 docstring도 이 용도를 명시한다.

```python
sh_degree: The SH degree to use, which can be smaller than the total
    number of bands. If set, the `colors` should be [N, K, D] SH coefficients ...
```

`rasterization` 안에서는 `assert (sh_degree + 1) ** 2 <= colors.shape[-2]`로 "요청 차수 ≤ 보유 계수"만 확인하고,
`_maybe_evaluate_sh(...)` → `spherical_harmonics(sh_degree, means, viewmats, features, masks)`로 색을 평가한다.
참조 PyTorch 구현(`gsplat/cuda/_torch_impl.py::_spherical_harmonics`)을 보면 어떤 일이 일어나는지 분명하다.

```python
num_bases = (degrees_to_use + 1) ** 2
bases = coeffs.new_zeros(dirs.shape[:-1] + (K,))
bases[..., :num_bases] = _eval_sh_bases_fast(num_bases, dirs)   # 활성 차수까지만 기저 계산
return (bases[..., None] * coeffs).sum(dim=-2)                   # 나머지 기저는 0 → 계수 무시
```

활성 차수 밖의 기저 값은 0이므로, 그 계수들은 forward 결과에 **아무 기여도 하지 않는다**(CUDA 커널
`sh_coeffs_to_color_fast(degrees_to_use, ...)`도 같은 규칙).

같은 `sh_degree_to_use`가 두 곳에 더 쓰인다.
- `app_opt`(외형 임베딩) 모드에서는 `self.app_module(..., sh_degree=sh_degree_to_use)`로 MLP 색 디코더에도 전달.
- 중간 PLY 저장 시 `app_module`을 원점 방향에서 평가해 색을 굽는(bake) 코드에도 현재 차수를 사용.

---

## 2. 스텝별 활성 차수 (기본 `sh_degree=3`, `sh_degree_interval=1000`)

| 스텝 범위 | `step // 1000` | `sh_degree_to_use` | 사용 계수 수 (ℓ+1)² | 사용되는 계수 |
|---|---|---|---|---|
| 0 – 999 | 0 | **0** | 1 | DC(`sh0`) 1개 — 시점 무관 기본색 |
| 1000 – 1999 | 1 | **1** | 4 | + `shN[0:3]` (1차 3개) — 방향에 따른 1차 기울기 |
| 2000 – 2999 | 2 | **2** | 9 | + `shN[3:8]` (2차 5개) |
| 3000 – 29999 | 3 이상 | **3** (`min`으로 고정) | 16 | + `shN[8:15]` (3차 7개) — 전부 활성 |

`min(..., cfg.sh_degree)` 덕분에 3000스텝 이후 `step // 1000`이 3을 넘어도 차수는 3에 고정된다.
기본 `max_steps=30_000`이므로 **전체 학습의 10%(첫 3000스텝)만 차수가 제한**되고, 나머지 90%는 완전한 16계수로 학습한다.
따라서 이 옵션의 실질적 영향은 학습 초반에 국한된다.

---

## 3. 원본 3DGS와 동일한 관행 — `oneupSHdegree()`

Inria 원본 구현(`train.py`)에도 같은 스케줄이 있다.

```python
# Every 1000 its we increase the levels of SH up to a maximum degree
if iteration % 1000 == 0:
    gaussians.oneupSHdegree()
```

```python
# scene/gaussian_model.py
def oneupSHdegree(self):
    if self.active_sh_degree < self.max_sh_degree:
        self.active_sh_degree += 1
```

원본은 `active_sh_degree`라는 **상태 변수**를 1000 iter마다 +1 하고 렌더러가 그것을 읽는 방식, gsplat은 상태 없이
매 스텝 `step // interval`로 **직접 계산**하는 방식이다. 결과는 같다(원본은 iteration이 1부터 시작해 999 iter 동안 0차,
gsplat은 1000 스텝 동안 0차 — 사실상 무시할 차이). gsplat은 이 상수 1000을 `sh_degree_interval`로 노출해 조절 가능하게
만든 것뿐이다.

---

## 4. 비활성 차수의 계수에는 무슨 일이 일어나는가

- 파라미터 텐서 자체는 처음부터 전부 존재한다: `sh0` `[N,1,3]`, `shN` `[N,15,3]` (`shN`은 0으로 초기화).
- 비활성 계수는 forward에서 곱해지는 기저가 0(또는 커널이 읽지 않음)이므로 **손실에 기여하지 않고, 기울기는 정확히 0**이다.
- Adam은 `grad`가 `None`이 아니라 **0 텐서**를 받으므로 파라미터를 건너뛰지는 않는다. 그러나
  `m ← β₁m + (1-β₁)·0`, `v ← β₂v + (1-β₂)·0` 이고 초기값이 0이므로 **1차·2차 모멘트가 0에 머물고**,
  갱신량 `lr·m̂/(√v̂+ε) = 0`. 즉 **계수 값은 초기값 0에 그대로 고정**된다(weight decay는 기본 0).
- 결과적으로 새 차수가 켜지는 순간 그 계수들은 전부 0이므로 **렌더 결과가 불연속적으로 튀지 않는다** —
  켜진 뒤 기울기를 받기 시작하면서 0에서부터 부드럽게 자란다.
- 미세한 부작용: `shN`은 하나의 텐서이므로 Adam의 `step` 카운트는 텐서 단위로 공유된다. 1000스텝 뒤 켜진 1차 계수는
  bias-correction 항(1-β^t)이 이미 ≈1인 상태에서 시작하지만, 모멘트가 0에서 축적되므로 실용적으로 문제 되지 않는다.

---

## 5. 왜 이런 커리큘럼이 필요한가

1. **기본색과 기하가 먼저 수렴해야 한다.** 학습 초반에는 Gaussian의 위치·크기·불투명도가 부정확하고
   (SfM 점 근처에 큰 Gaussian이 흩어져 있는 상태), 밀집화(densification)로 개체가 계속 분열·복제된다.
   이때 DC만 학습하면 "이 Gaussian은 대략 어떤 색인가"라는 가장 잘 조건화된 문제부터 풀게 된다.
2. **잘못된 방향 의존성이 새겨지는 것을 막는다.** 위치가 틀린 상태에서 고차 계수까지 자유롭게 풀어 두면,
   모델은 기하 오류를 고치는 대신 "이 카메라에서만 이런 색"이라는 식으로 시점 의존 색을 조작해 손실을 낮출 수 있다.
   이렇게 새겨진 방향 의존성은 기하가 나중에 맞아도 남아서 **새로운 시점에서 깜빡임·색 튐(floater 색 아티팩트)** 을 만든다.
3. **과적합 방지.** SH 계수 학습은 "관측한 카메라 방향에서의 색 → 구면 함수 복원"이라는 역문제다.
   asset 노트북 4.2절에서 보듯 관측 뷰가 적을수록 고차가 관측 사이에서 요동(과적합)한다. 초기엔 각 Gaussian이
   기여하는 유효 관측이 적으므로 표현력을 낮게 제한하는 것이 정규화 역할을 한다.
4. **저차→고차는 coarse-to-fine의 자연스러운 순서다.** SH는 ℓ이 커질수록 구면에서 더 빠르게 진동하는 성분이므로,
   차수를 하나씩 켜는 것은 색의 "주파수"를 낮은 것부터 맞추는 것과 같다(asset 4.1절의 그림이 바로 이것:
   같은 계수를 `sh_degree=0,1,2,3`으로 평가하면 세부가 점차 붙는다).

asset 노트북은 이 스케줄을 축소 재현한다(Gaussian 하나, 4000스텝, 500스텝 간격).

```python
for step in range(4000):
    degree_to_use = min(step // 500, 3)          # sh_degree_interval 흉내
    Kd = (degree_to_use + 1) ** 2
    coeffs = torch.cat([sh0, shN], dim=0)        # [16,3]
    pred = torch.clamp_min(A_obs[:, :Kd] @ coeffs[:Kd] + 0.5, 0.0)
```

---

## 6. 평가·렌더링 시에는 항상 최대 차수

학습 이외의 경로는 `sh_degree_to_use`를 쓰지 않고 `cfg.sh_degree`(최대 차수)를 고정으로 넘긴다.

```python
# eval(): 검증 이미지 렌더
colors, _, _ = self.stage.render(..., sh_degree=cfg.sh_degree, ...)

# render_traj(): 궤적 비디오 렌더
renders, _, _ = self.stage.render(..., sh_degree=cfg.sh_degree, ...)

# viewer 콜백: 슬라이더로 낮출 수는 있지만 상한은 cfg.sh_degree
render_colors, render_alphas, info = self.stage.render(
    ..., sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree), ...)
```

- `eval_steps`에 1000·2000이 들어 있어도 그 시점의 평가는 최대 차수로 렌더된다. 비활성 계수가 아직 0이므로
  결과는 학습 forward와 동일하고, 나중 스텝에서는 학습된 고차까지 전부 반영된다.
- 원본 3DGS도 체크포인트·PLY를 로드하면 `active_sh_degree = max_sh_degree`로 두므로 동일한 관행이다.
- 뷰어의 `max_sh_degree` 슬라이더는 디버깅용으로 차수를 **임시로 낮춰** 시점 의존 성분의 기여를 확인하는 용도다.

---

## 7. 한 줄 요약

`sh_degree_interval`은 파라미터를 바꾸는 옵션이 아니라 **forward가 읽는 계수 개수를 스텝에 따라 1→4→9→16으로 늘리는
스케줄**이다. 기하와 기본색이 먼저 자리 잡게 하고, 초반의 잘못된 시점 의존성·과적합을 막기 위한 원본 3DGS의
`oneupSHdegree()` 관행을 설정값으로 노출한 것으로, 기본 30k 스텝 학습에서는 첫 3000스텝에만 영향을 준다.
