# 손실 곡선에서 읽는 "밀도화의 흔적"

> **Q.** 손실 곡선에서 밀도화의 흔적은 어떻게 나타나는가?
>
> **A.** Gaussian 수가 refine 구간(스텝 500~)에서 100스텝마다 계단식으로 늘어난다. 30k 스텝 완주 시에는 3000스텝마다 opacity reset 직후 loss가 튀었다 회복하는 패턴과 15000스텝 이후 개수 고정도 보인다.

---

## 1. 어디를 보고 하는 말인가

워크스루(`training_walkthrough.py`)는 학습 루프에서 매 스텝 세 가지를 기록한다.

```python
history = {"step": [], "loss": [], "num_gs": []}
...
history["step"].append(step)
history["loss"].append(loss.item())
history["num_gs"].append(len(splats["means"]))
```

그리고 학습이 끝나면 이 둘을 **나란히** 그린다.

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
axes[0].plot(history["step"], history["loss"], lw=0.5)   # loss (log scale)
axes[1].plot(history["step"], history["num_gs"])         # #Gaussians
```

핵심은 "손실 곡선"이 loss 하나만 보는 게 아니라 **loss와 Gaussian 개수를 같은 x축(step) 위에 놓고 본다**는 점이다. 밀도화는 파라미터 개수 자체를 바꾸는 이벤트라서, 그 흔적이 두 곡선 모두에 서로 다른 모양으로 찍힌다.

## 2. 흔적 ①: Gaussian 개수의 계단(staircase)

`DefaultStrategy.step_post_backward()`(`gsplat/strategy/default.py`)가 refine을 실행하는 조건:

```python
if (
    step > self.refine_start_iter          # 기본 500
    and step % self.refine_every == 0      # 기본 100
    and step % self.reset_every >= self.pause_refine_after_reset   # 기본 0 → 항상 참
):
    n_dupli, n_split = self._grow_gs(...)  # duplicate + split
    n_prune        = self._prune_gs(...)   # prune
```

즉 **연속적으로 늘지 않고, 100스텝에 한 번만 개수가 바뀐다.** 그 사이 99스텝 동안 개수는 완전히 평평하다. 그래서 `#Gaussians` 곡선이 매끈한 곡선이 아니라 **계단 모양**이 된다.

- 계단의 **가로 폭** = `refine_every = 100`
- 계단의 **세로 높이** = 그 refine에서 `duplicate + split − prune` 한 순증분
- 계단이 **시작되는 지점** = 스텝 500 직후. 조건이 `step > 500`(등호 없음)이라 실제 첫 계단은 **스텝 600**에서 올라간다.
- 계단이 **끝나는 지점** = `refine_stop_iter = 15000`

100스텝 사이에 무슨 일이 일어나는가: `_update_state()`가 매 스텝 화면공간 gradient를 `[-1,1]` NDC 기준으로 정규화해 `state["grad2d"]`에 누적하고 `state["count"]`를 센다. refine 때 그 평균이 `grow_grad2d = 2e-4`를 넘는 Gaussian을 골라 크기에 따라 duplicate(작은 것) / split(큰 것, 크기 /1.6)하고, opacity < `prune_opa = 0.005`이거나 과도하게 큰 것을 prune한 뒤 통계를 `zero_()`한다. **"통계를 100스텝 모아서 한 번에 실행"이 계단의 물리적 원인**이다.

`DefaultStrategy(verbose=True)`로 두면 계단 하나하나가 로그로도 찍힌다:

```
Step 600: 1234 GSs duplicated, 567 GSs split. Now having 178901 GSs.
Step 600: 89 GSs pruned. Now having 178812 GSs.
```

loss 곡선 쪽에서는 이 순간이 **작은 하향 꺾임**으로 나타난다. 표현력(파라미터 수)이 갑자기 늘어나 그 직후 몇 스텝 동안 loss가 평소보다 빠르게 떨어지기 때문이다. 반대로 split은 크기를 1.6으로 나누므로 직후 아주 작은 상향 잡음이 섞일 수도 있다.

## 3. 흔적 ②: 3000스텝마다의 loss 스파이크 (opacity reset)

같은 함수의 **refine 블록 바깥**에 별도 조건이 있다.

```python
if step % self.reset_every == 0 and step > 0:   # 기본 3000
    reset_opa(params, optimizers, state, value=self.prune_opa * 2.0)  # 0.01
```

`reset_opa()`(`gsplat/strategy/ops.py`)는 opacity를 0.01로 "설정"하는 게 아니라 **0.01을 상한으로 clamp**한다(`torch.clamp(p, max=logit(0.01))`), 그리고 해당 파라미터의 Adam 모멘텀 상태를 `zeros_like`로 초기화한다.

결과적으로 스텝 3000, 6000, 9000, 12000에서:

1. 거의 모든 Gaussian이 갑자기 투명해진다 → 렌더 이미지가 확 흐려짐 → **loss가 위로 튄다(스파이크)**
2. 이후 수십~수백 스텝에 걸쳐 opacity가 다시 올라오면서 **loss가 원래 궤적으로 회복**한다
3. 진짜 필요한 Gaussian만 opacity를 회복하고, 회복하지 못한 것은 다음 refine에서 `prune_opa = 0.005` 조건에 걸려 제거된다 → **floater 정리**

그래서 30k 학습의 loss 곡선(log scale)은 단조 감소가 아니라 **3000 간격으로 이빨(톱니)이 난 감소 곡선**이다. `#Gaussians` 곡선에서도 reset 직후 첫 refine에서 prune이 많이 발생해 **계단이 아래로 내려가는 구간**이 종종 보인다.

> 참고: `pause_refine_after_reset`를 기본 0이 아니라 학습 이미지 장수로 두면, reset 직후 opacity가 회복될 때까지 refine을 쉬게 해서 "회복도 못한 Gaussian이 잘못 판정되는" 문제를 줄인다.

## 4. 흔적 ③: 15000스텝 이후 개수 완전 고정

```python
def step_post_backward(...):
    if step >= self.refine_stop_iter:   # 기본 15000
        return
```

**early return이 함수 맨 앞에 있다.** 이 한 줄이 두 가지를 동시에 끈다.

- 더 이상 duplicate/split/prune 없음 → `#Gaussians` 곡선이 **완전한 수평선**
- opacity reset 블록도 이 return 아래에 있으므로 **15000 이후에는 reset도 없음** → loss 곡선의 톱니도 사라지고 매끄러운 수렴 구간만 남는다

즉 30k 학습의 그래프는 성격이 확연히 다른 두 구간으로 갈린다.

```
#Gaussians                                   loss (log)
    │            ┌─┘‾‾‾‾‾‾‾‾‾‾‾‾‾  (고정)      │╲
    │        ┌─┘‾                              │ ╲  ↑spike   ↑spike
    │    ┌─┘‾                                  │  ╲╱╲    ╱╲ ╱╲
    │┌─┘‾ ← 100스텝 계단                        │      ╲╱   ╲  ╲___  (매끈)
    └──┬──────────────┬──────────►             └──┬────┬────┬───┬──►
      500          15000  30000                 500 3000 6000 15000
```

## 5. 워크스루 그대로 돌리면 무엇이 보이나

`MAX_STEPS = 2_000`(데모용, 논문 재현은 30_000)이므로 **실제로 눈에 보이는 건 흔적 ①뿐이다.**

| 흔적 | 조건 | 2k 데모 | 30k 완주 |
|---|---|---|---|
| 100스텝 계단 | step 600~ | 보임 (계단 약 14개) | 보임 (계단 약 144개) |
| 3000스텝 loss 스파이크 | step 3000, 6000, … | **안 보임** (2000 < 3000) | 보임 (4회) |
| 개수 고정 | step ≥ 15000 | **안 보임** | 보임 |

카드의 답이 "30k 스텝 완주 시에는 …도 보인다"라고 조건을 붙인 이유가 바로 이것이다. 짧게 돌리면 밀도화의 흔적 중 계단 하나만 관찰된다.

한편 `simple_trainer.py`의 `adjust_steps(factor)`는 `max_steps`뿐 아니라 `refine_start_iter / refine_stop_iter / reset_every / refine_every`를 **모두 같은 비율로** 스케일한다. 그래서 짧게 돌릴 때도 세 흔적의 상대적 위치를 보존하고 싶다면 워크스루처럼 `MAX_STEPS`만 줄이지 말고 `adjust_steps`를 쓰는 편이 맞다.

## 6. 헷갈리기 쉬운 지점

- **"스텝 500부터"는 근사 표현**이다. 코드는 `step > 500`이므로 첫 계단은 600.
- **loss 스파이크의 원인은 refine이 아니라 opacity reset**이다. duplicate/split은 표현력을 늘리므로 loss를 (조금) 낮추는 방향이지, 위로 튀게 하지 않는다.
- **reset은 "전부 0.01로 만들기"가 아니라 "0.01 이하로 깎기"**(clamp)다. 이미 0.005였던 것은 그대로다.
- **15000 이후 loss가 계속 내려가는 것은 정상**이다. 개수가 고정됐을 뿐 means/scales/quats/opacity/SH는 계속 최적화되고, `means` lr은 `gamma = 0.01^(1/max_steps)`로 초기값의 1%까지 지수 감쇠하며 미세 조정으로 넘어간다.
- **MCMCStrategy를 쓰면 이 흔적들이 다르게 나온다.** MCMC는 `cap_max`로 개수 상한을 두고 opacity 기반 확률적 재배치를 하므로, 계단이 상한까지 올라간 뒤 평평해지고 opacity reset 스파이크는 없다.

## 7. 한 줄 요약

`#Gaussians` 곡선의 **100스텝 계단**(refine_every) = 밀도화가 주기적 배치 작업이라는 증거, loss 곡선의 **3000스텝 톱니**(reset_every) = opacity reset이 강제로 렌더를 망가뜨렸다 회복시킨 흔적, **15000 이후 두 곡선의 평탄화**(refine_stop_iter의 early return) = 구조 탐색이 끝나고 순수 파라미터 미세조정으로 넘어간 경계.
