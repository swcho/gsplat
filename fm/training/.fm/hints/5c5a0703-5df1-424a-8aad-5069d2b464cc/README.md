# 3DGS 최적화기 구성 — 파라미터마다 별도 Adam, `eps=1e-15`, `fused=True`

## 카드 요약

> **Q.** 3DGS 최적화기 구성의 특징은?
> **A.** 파라미터마다 별도의 Adam을 쓰고 `eps=1e-15`, `fused=True`로 설정한다. 학습률이 파라미터별로 크게 다르기 때문이다.

세 요소가 각각 다른 문제를 푼다.

| 설정 | 해결하는 문제 |
|---|---|
| 파라미터별로 **별도 Adam 인스턴스** | lr이 파라미터마다 다르고, densification이 파라미터 텐서를 매번 갈아치우며, 스케줄러/옵티마이저 종류를 파라미터별로 다르게 걸어야 한다 |
| `eps=1e-15` | 그래디언트가 극도로 작은 파라미터(고차 SH 등)에서 Adam의 정규화 성질이 무너지는 것을 막는다 |
| `fused=True` | 매 스텝 6개 옵티마이저를 돌리므로 elementwise 갱신을 단일 CUDA 커널로 묶고 host 동기화를 없앤다 |

---

## 1. 실제 코드

워크스루의 `init_splats_with_optimizers` (asset `training_walkthrough.py:157`) — 옵티마이저 생성부만 보면:

```python
lrs = {
    "means":     1.6e-4 * scene_scale,  # 위치는 씬 크기에 비례
    "scales":    5e-3,
    "quats":     1e-3,
    "opacities": 5e-2,
    "sh0":       2.5e-3,
    "shN":       2.5e-3 / 20,           # 고차 SH는 천천히
}
optimizers = {
    name: torch.optim.Adam([{"params": splats[name], "lr": lr, "name": name}],
                           eps=1e-15, fused=True)
    for name, lr in lrs.items()
}
```

핵심은 `optimizers`가 **단일 옵티마이저가 아니라 `{파라미터 이름: 옵티마이저}` 딕셔너리**라는 것이다. `Adam(splats.parameters(), ...)` 한 개도, param_group 6개를 가진 Adam 한 개도 아니다.

원본 `examples/simple_trainer.py:288 create_splats_with_optimizers`도 동일 구조이며, 배치 크기 보정과 옵티마이저 클래스 선택이 추가된다:

```python
BS = batch_size * world_size
optimizer_class = (torch.optim.SparseAdam if sparse_grad
                   else SelectiveAdam if visible_adam
                   else torch.optim.Adam)
optimizers = {
    name: optimizer_class(
        [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
        eps=1e-15 / math.sqrt(BS),
        betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        fused=True,
    )
    for name, _, lr in params
}
```

`lr`은 `sqrt(BS)` 배, `eps`는 `sqrt(BS)`로 나누고, `betas`의 "1과의 거리"는 `BS`배 — 배치를 키워도 SDE 관점에서 같은 학습 동역학을 유지하려는 스케일링 규칙이다.

---

## 2. 왜 "파라미터마다 별도" 인스턴스인가

lr이 다르다는 것만으로는 param_group 여러 개로도 충분하다. 별도 **인스턴스**여야 하는 이유는 네 가지다.

### (a) lr의 동적 범위가 300배

| 파라미터 | 저장 공간 | lr | 비고 |
|---|---|---|---|
| `means` | 그대로 | `1.6e-4 × scene_scale` | 씬 크기에 비례 → 씬 스케일 무관하게 만듦 |
| `scales` | log | `5e-3` | `exp`로 활성화되므로 log-space 스텝 |
| `quats` | 미정규화 | `1e-3` | 내부에서 normalize |
| `opacities` | logit | `5e-2` | 가장 큼. sigmoid 포화 구간을 빠르게 통과해야 함 |
| `sh0` | SH DC | `2.5e-3` | 기본 색 |
| `shN` | SH 고차 | `1.25e-4` (= `2.5e-3/20`) | 시점 의존 성분은 천천히 |

`opacities`(5e-2)와 `shN`(1.25e-4) 사이가 400배, `means`는 `scene_scale`에 따라 다시 달라진다. 각 파라미터가 서로 다른 재매개변수화(log / logit / SH 계수) 공간에 살고 있어서 "적당한 스텝 크기"의 스케일 자체가 다르기 때문이다.

### (b) 스케줄러를 `means`에만 걸 수 있다

```python
means_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizers["means"], gamma=0.01 ** (1.0 / MAX_STEPS)
)
```

`means`만 전체 학습 동안 초기값의 1%까지 지수 감쇠한다(원본은 `simple_trainer.py:809`). 위치는 초반에 크게 움직여 구조를 잡고 후반에는 거의 고정되어야 하지만, 색·불투명도는 끝까지 같은 lr로 미세 조정된다. 옵티마이저가 하나였다면 `ExponentialLR`이 6개 param_group 전부의 lr을 깎아버린다.

MCMC 전략은 이 스케줄된 lr을 노이즈 크기로 재사용한다 — `simple_trainer.py:1172`가 `lr=schedulers[0].get_last_lr()[0]`을 넘기고, `gsplat/strategy/mcmc.py:190`이 `noise_scale = lr * noise_lr`로 쓴다. 즉 `means`의 전용 옵티마이저/스케줄러가 SGLD 노이즈 스케줄까지 결정한다.

### (c) Densification이 옵티마이저 상태를 수술한다 — 이름으로 찾아야 한다

3DGS는 학습 중 Gaussian을 복제/분할/제거하므로 **파라미터 텐서의 첫 차원 N이 계속 바뀐다.** 새 `nn.Parameter`로 교체하면 Adam의 모멘텀(`exp_avg`, `exp_avg_sq`)도 같은 인덱스 구조로 다시 만들어줘야 한다. `gsplat/strategy/ops.py:96 _update_param_with_optimizer`가 그 일을 한다:

```python
optimizer = optimizers[name]          # ← 이름으로 옵티마이저를 직접 찾는다
for i in range(len(optimizer.param_groups)):
    param_state = optimizer.state[param]
    del optimizer.state[param]
    for key in param_state.keys():
        if key != "step":
            param_state[key] = optimizer_fn(key, param_state[key])
    optimizer.param_groups[i]["params"] = [new_param]
    optimizer.state[new_param] = param_state
```

연산별 `optimizer_fn`이 모멘텀을 파라미터와 같은 방식으로 재배열한다:

- **duplicate** (`ops.py:161`): 파라미터는 `cat([p, p[sel]])`, 모멘텀은 `cat([v, zeros])` — 복제된 Gaussian은 모멘텀을 물려받지 않고 0에서 다시 시작
- **split** (`ops.py:221`): `cat([v[rest], zeros(2*len(sel))])` — 쪼개진 자식들도 모멘텀 0
- **prune** (`ops.py:257`): `v[sel]` — 살아남은 인덱스만 남김

옵티마이저를 하나로 합치면 이 수술이 이름 기반 조회를 잃고, param_group 인덱스와 파라미터 정체성을 수동으로 맞춰야 한다. 그래서 베이스 클래스가 아예 규약으로 못 박아 둔다 (`gsplat/strategy/base.py:30 check_sanity`):

```python
assert trainable_params == set(optimizers.keys())      # 이름이 정확히 일치해야 함
for optimizer in optimizers.values():
    assert len(optimizer.param_groups) == 1            # 옵티마이저당 param_group 하나
```

학습 루프에서 `strategy.check_sanity(splats, optimizers)`가 실패하면 거의 항상 이 구조를 어긴 경우다.

### (d) 파라미터별로 옵티마이저 종류/호출 방식이 달라질 수 있다

`visible_adam` 모드는 `optimizer.step(visibility_mask)`라는 **비표준 시그니처**를 쓴다 (`gsplat/optimizers/selective_adam.py:63`, Taming-3DGS의 SelectiveAdam — 보이는 Gaussian만 갱신). `sparse_grad` 모드는 각 파라미터의 `.grad`를 `sparse_coo_tensor`로 바꿔 `SparseAdam`에 넘긴다. 딕셔너리 구조라 루프가 단순해진다 (`simple_trainer.py:1131`):

```python
for optimizer in self.optimizers.values():
    optimizer.step(visibility_mask) if cfg.visible_adam else optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

---

## 3. `eps=1e-15` — 왜 기본값 `1e-8`이 아닌가

Adam 갱신식은

$$\Delta\theta = -\text{lr}\cdot\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$$

이고, Adam의 매력은 `sqrt(v)`가 그래디언트 크기를 나눠주어 **스텝이 그래디언트 스케일에 거의 무관해진다**는 점이다. 그런데 이 성질은 `sqrt(v) ≫ eps`일 때만 성립한다. 그래디언트가 `eps`보다 작아지면 분모가 `eps`에 지배되어 갱신량이 `lr·m/eps`로 줄고, 실질 lr이 조용히 죽는다.

첫 스텝의 실제 갱신량을 측정하면(`lr=1.0`, bias correction 포함):

```
eps=1e-08  grad=1e-03 -> step=9.99990e-01    (lr의 100.0%)
eps=1e-08  grad=1e-06 -> step=9.90099e-01    (lr의  99.0%)
eps=1e-08  grad=1e-09 -> step=9.09091e-02    (lr의   9.1%)  ← 11배 느려짐
eps=1e-15  grad=1e-03 -> step=1.00000e+00    (lr의 100.0%)
eps=1e-15  grad=1e-06 -> step=9.99999e-01    (lr의 100.0%)
eps=1e-15  grad=1e-09 -> step=9.99999e-01    (lr의 100.0%)
```

3DGS에서 이게 문제가 되는 지점:

- **`shN` (고차 SH)** — 15×3 계수가 0으로 초기화되고, `sh_degree` 워밍업 때문에 초반에는 아예 사용되지 않는다. 그래디언트가 `1e-9` 수준으로 극히 작다.
- **가려진/작은 Gaussian** — 픽셀 기여가 거의 없어 그래디언트가 언더플로 직전이다. `eps=1e-8`이면 이들은 사실상 학습이 멈춘 채 densification 판정만 받는다.
- **`means`** — lr 자체가 `1.6e-4·scene_scale`로 작은 데다 후반엔 그 1%까지 감쇠하므로, 분모까지 `eps`에 먹히면 위치 미세 조정이 사라진다.

`eps`의 원래 목적은 0 나눗셈 방지이므로 `1e-15`로 내려도 float32에서 안전하다(`sqrt(v)`가 정확히 0인 경우만 막으면 된다). 3DGS 계열 구현이 관례적으로 `1e-15`를 쓰는 이유다.

---

## 4. `fused=True` — 무엇이 묶이고, 무엇을 요구하는가

`fused=True`는 PyTorch가 Adam의 elementwise 체인(모멘텀 갱신 → bias correction → 나눗셈 → 파라미터 갱신)을 **하나의 CUDA 커널**로 실행하게 한다. 부가 효과로 `state["step"]`이 GPU 텐서로 유지되어 스텝마다 GPU→CPU 동기화가 사라진다(CUDA graph capture 가능).

### 제약 — 파라미터가 contiguous여야 한다

워크스루가 `.contiguous()`를 붙이는 이유가 여기 있다 (`training_walkthrough.py:178`):

```python
"sh0": torch.nn.Parameter(colors[:, :1, :].contiguous()),
"shN": torch.nn.Parameter(colors[:, 1:, :].contiguous()),
```

`colors`는 `[N, K, 3]` 하나의 텐서이므로 `colors[:, 1:, :]`는 stride가 어긋난 **뷰**다. 그대로 fused Adam에 넘기면:

```
is_contiguous: False
RuntimeError: params, grads, exp_avgs, and exp_avg_sqs must have same dtype, device, and layout
```

`exp_avg`/`exp_avg_sq`는 `zeros_like`로 만들어져 contiguous인데 파라미터는 non-contiguous라 layout이 어긋나 커널이 거부한다. `.contiguous()` 하나 빼먹으면 학습 시작 직후 이 에러를 만나며, 메시지가 dtype/device를 먼저 언급해서 원인을 놓치기 쉽다. `fused=False`로 두면 통과하지만 그건 우회일 뿐이다.

그 밖의 요구사항: 모든 파라미터가 CUDA의 부동소수 텐서여야 하고, sparse gradient는 지원하지 않는다(그래서 `sparse_grad=True`는 `SparseAdam`으로 분기한다).

### 속도는 얼마나 이득인가 — 정직한 측정

`torch 2.9.1+cu128`에서 6개 파라미터 옵티마이저의 `step()`만 반복 측정(래스터라이저 제외):

```
N=10,000    fused=False: 0.762 ms/step    fused=True: 1.324 ms/step
N=100,000   fused=False: 0.792 ms/step    fused=True: 2.264 ms/step
N=1,000,000 fused=False: 5.046 ms/step    fused=True: 5.257 ms/step
```

즉 **이 마이크로벤치마크에서는 fused가 더 빠르지 않다.** 이유는 두 가지다.

1. `fused=False`의 기본 경로는 이미 `foreach`(multi-tensor) 커널이라 순차 elementwise 연산이 아니다.
2. 옵티마이저당 텐서가 **1개**뿐이라 fused가 묶을 것이 별로 없고, 대신 옵티마이저 인스턴스 6개 각각의 per-step 파이썬/디바이스 텐서 오버헤드를 낸다.

N이 커지면 둘 다 메모리 대역폭에 묶여 차이가 사라진다(5.0 vs 5.3 ms). 실제 학습에서는 스텝 시간의 대부분이 래스터라이저와 backward이므로 이 차이는 노이즈에 가깝다. `fused=True`의 실질적 가치는 **속도보다 GPU→CPU 동기화 제거와 `.contiguous()` 규율 강제**에 있다고 보는 편이 정확하다. 카드의 "`fused=True`로 설정한다"는 gsplat의 사실적 기본값이며, 성능 주장으로 과대해석할 필요는 없다.

---

## 5. 학습 루프에서의 위치

```python
# (5) 파라미터별 Adam step + zero_grad        training_walkthrough.py:380
for opt in optimizers.values():
    opt.step()
    opt.zero_grad(set_to_none=True)

# (6) means lr 지수 감쇠
means_lr_scheduler.step()

# (7) densification — 여기서 optimizers 딕셔너리를 넘긴다
strategy.step_post_backward(splats, optimizers, strategy_state, step, info, packed=False)
```

`strategy`에 `optimizers`를 통째로 넘기는 것이 (c)의 옵티마이저 상태 수술을 위한 통로다. 파라미터만 넘기면 모멘텀이 옛 N에 맞춰진 채 남아 다음 스텝에서 shape mismatch가 난다.

---

## 6. 함정 체크리스트

| 증상 | 원인 |
|---|---|
| `params, grads, exp_avgs, ... must have same dtype, device, and layout` | 슬라이스 뷰를 `nn.Parameter`로 씀 → `.contiguous()` 누락 |
| `trainable parameters and optimizers must have the same keys` | 파라미터를 추가했는데 옵티마이저를 안 만듦, 또는 옵티마이저 하나로 합침 |
| `Each optimizer must have exactly one param_group` | param_group 6개를 가진 Adam 한 개를 넘김 |
| densification 후 shape mismatch | `strategy.step_post_backward`에 `optimizers`를 안 넘겨 모멘텀이 갱신되지 않음 |
| 색/디테일이 안 올라옴 | `eps`를 기본값 `1e-8`로 되돌렸음 (특히 `shN`) |
| 큰 씬에서 위치가 안 움직임 / 폭주 | `means` lr에 `scene_scale`을 안 곱했음 |
| 배치를 키웠는데 결과가 나빠짐 | `lr·sqrt(BS)`, `eps/sqrt(BS)`, `betas` 보정 누락 |

---

## 7. 직접 확인해 보기

```bash
# eps가 작은 그래디언트에서 실질 lr을 어떻게 살리는지
python3 -c "
import torch
for eps in (1e-8, 1e-15):
    for g in (1e-3, 1e-6, 1e-9):
        p = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.Adam([p], lr=1.0, eps=eps)
        p.grad = torch.tensor([g]); opt.step()
        print(f'eps={eps:.0e} grad={g:.0e} -> step={-p.item():.5e}')
"

# fused Adam이 non-contiguous 파라미터를 거부하는 것
python3 -c "
import torch
c = torch.zeros(10, 16, 3, device='cuda')
p = torch.nn.Parameter(c[:, 1:, :])          # 뷰 → non-contiguous
opt = torch.optim.Adam([p], lr=1e-3, eps=1e-15, fused=True)
p.grad = torch.ones_like(p)
try: opt.step(); print('OK')
except RuntimeError as e: print('ERROR:', e)
"
```

## 참고 위치

- asset: `training_walkthrough.py:142-197` (설명 + `init_splats_with_optimizers`), `:332` (스케줄러), `:380-389` (루프)
- `examples/simple_trainer.py:288` `create_splats_with_optimizers`, `:809` 스케줄러, `:1131` step 루프
- `gsplat/strategy/base.py:30` `check_sanity`
- `gsplat/strategy/ops.py:96` `_update_param_with_optimizer`, `:161` duplicate, `:221` split, `:257` prune
- `gsplat/optimizers/selective_adam.py:21` `SelectiveAdam`
- `gsplat/strategy/mcmc.py:190` `noise_scale = lr * noise_lr`
