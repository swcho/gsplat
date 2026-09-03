# `means`의 학습률 감쇠 스케줄

> **정답 요약**
> `torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / MAX_STEPS))`
> — 매 스텝 학습률에 $\gamma$를 곱해서, 총 `MAX_STEPS` 스텝이 지나면 초기 학습률의 **1%**가 되도록 지수 감쇠시킨다.

---

## 1. 학습률이란 무엇인가 (미분에서 출발)

3DGS 학습은 결국 "손실 함수 $L$을 가장 작게 만드는 파라미터 찾기"다. 고교 미적분에서 배운 대로, 어떤 지점에서 함수의 **도함수(기울기)** 는 그 점에서 함수가 얼마나 가파르게 증가하는지를 알려준다. 그러니 값을 줄이고 싶으면 기울기의 **반대 방향**으로 조금 움직이면 된다.

$$
\theta_{t+1} \;=\; \theta_t \;-\; \eta \,\frac{\partial L}{\partial \theta}\bigg|_{\theta_t}
$$

여기서 $\eta$(에타)가 **학습률(learning rate)**, 즉 "한 걸음의 보폭"이다. 파라미터가 벡터일 때는 $\partial L/\partial\theta$ 자리에 각 성분별 편미분을 모은 벡터(기울기 벡터, gradient)가 들어가지만, 아이디어는 똑같다.

- $\eta$가 너무 크면 → 최솟값을 **건너뛰고** 왔다 갔다 진동한다.
- $\eta$가 너무 작으면 → 제대로 된 답까지 가는 데 시간이 너무 오래 걸린다.

그래서 실전에서는 **처음엔 크게, 나중엔 작게** 라는 전략을 쓴다. 이것이 학습률 스케줄(schedule)이다.

### 왜 하필 `means`인가

`means`는 각 Gaussian의 **3D 위치** $\mu \in \mathbb{R}^3$다. 학습 초기에는 SfM 포인트가 대충 놓여 있으므로 Gaussian들이 씬 전체를 크게 돌아다녀야 한다. 반면 학습 후반에는 위치가 **픽셀 이하(sub-pixel)** 수준으로 미세하게 맞아야 렌더링이 선명해진다. 이때도 초기 보폭을 유지하면 파라미터가 정답 주변에서 계속 떨리며(진동) 이미지가 뭉개진다.

`scales`, `quats`, `opacities`, `sh0`, `shN`은 고정 학습률을 쓰고, gsplat에서 **감쇠 스케줄이 붙는 것은 `means`(그리고 옵션인 카메라 포즈·후처리 모듈)뿐**이다. `means`의 초기 학습률 자체도 씬 크기에 비례한다.

```python
lrs = {
    "means": 1.6e-4 * scene_scale,   # 위치는 씬 크기에 비례
    "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
    "sh0": 2.5e-3, "shN": 2.5e-3 / 20,
}
```

---

## 2. 지수 감쇠와 `gamma`

`ExponentialLR`은 `scheduler.step()`이 호출될 때마다 학습률에 상수 $\gamma$를 **곱한다**. 고교에서 배운 등비수열이다.

$$
\eta_t \;=\; \eta_0\,\gamma^{\,t},\qquad t = 0,1,2,\dots
$$

우리가 원하는 조건은 "총 $T$ 스텝 뒤에 초기값의 1%가 되어라"이다. 즉

$$
\eta_T = 0.01\,\eta_0
\quad\Longleftrightarrow\quad
\gamma^{\,T} = 0.01 .
$$

양변을 $T$제곱근 하면 (또는 로그를 취하면)

$$
\boxed{\;\gamma \;=\; 0.01^{\,1/T} \;=\; \exp\!\left(\frac{\ln 0.01}{T}\right)\;}
$$

이것이 코드의 `gamma = 0.01 ** (1.0 / MAX_STEPS)`다.

```python
means_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizers["means"], gamma=0.01 ** (1.0 / MAX_STEPS)
)
```

### 숫자로 확인

$\ln 0.01 = -4.60517\ldots$ 이므로

| `MAX_STEPS` $T$ | $\gamma = 0.01^{1/T}$ | 1스텝당 감소율 |
|---|---|---|
| 2,000 (워크스루 데모) | $0.997700$ | 약 $-0.23\%$ |
| 7,000 | $0.999342$ | 약 $-0.066\%$ |
| 30,000 (논문 기본값) | $0.9998465$ | 약 $-0.015\%$ |

한 스텝만 보면 거의 1에 가까워서 아무 일도 안 하는 것처럼 보이지만, 이것을 $T$번 곱하면 정확히 $1/100$이 된다. "티끌 모아 태산"의 곱셈 버전이다.

### 로그를 씌우면 직선

양변에 자연로그를 취하면

$$
\ln \eta_t \;=\; \ln \eta_0 \;+\; t\ln\gamma \;=\; \ln\eta_0 \;-\; \frac{t}{T}\,\ln 100 .
$$

즉 **세로축을 로그 스케일로 그리면 학습률 곡선은 완벽한 직선**이다. 이것이 "감쇠가 일정한 비율로 일어난다"는 말의 기하학적 의미다.

### 반감기 (물리의 방사성 붕괴와 같은 식)

$\eta$가 절반이 되는 데 걸리는 스텝 수 $t_{1/2}$는

$$
\gamma^{\,t_{1/2}} = \tfrac12 \;\Longrightarrow\;
t_{1/2} = \frac{\ln 2}{-\ln\gamma} = T\cdot\frac{\ln 2}{\ln 100} \approx 0.1505\,T .
$$

$T = 30{,}000$이면 약 4,515스텝마다 학습률이 반씩 줄어든다. 방사성 붕괴 $N(t)=N_0 e^{-\lambda t}$와 수학적으로 완전히 같은 형태이며, $\lambda = -\ln\gamma = \ln 100 / T$가 붕괴상수 역할을 한다.

---

## 3. 이 설계의 좋은 점

**(a) 총 스텝 수에 무관한 모양.** $t$를 진행률 $s = t/T$로 바꿔 쓰면

$$
\eta_t = \eta_0\,\gamma^{\,t} = \eta_0\,\left(0.01^{1/T}\right)^{sT} = \eta_0 \cdot 100^{-s}.
$$

$T$가 사라졌다! 2,000스텝짜리 데모든 30,000스텝짜리 본 학습이든, **진행률 기준으로는 똑같은 곡선**을 따른다. 스텝 수를 바꿔도 스케줄을 다시 튜닝할 필요가 없다는 뜻이다.

**(b) 0으로 죽지 않는다.** 지수함수는 아무리 곱해도 양수이므로 학습률이 0이 되는 일이 없다. 선형 감쇠($\eta_0(1-t/T)$)처럼 끝에서 정확히 0이 되어 학습이 완전히 멈추는 대신, 마지막까지 아주 작은 미세 조정 여지를 남긴다.

**(c) 1%라는 값.** 너무 크면 후반 진동이 남고, 너무 작으면 후반이 사실상 학습 정지가 된다. 두 자릿수(100배) 축소는 "충분히 미세하지만 아직 살아 있는" 절충점이다.

---

## 4. 학습 루프에서의 위치

```python
for step in range(MAX_STEPS):
    ...
    loss.backward()
    for opt in optimizers.values():      # (5) 파라미터별 Adam
        opt.step()
        opt.zero_grad(set_to_none=True)
    means_lr_scheduler.step()            # (6) means lr 감쇠
    strategy.step_post_backward(...)     # (7) 밀도화
```

주의할 점 두 가지:

1. `scheduler.step()`은 **`optimizer.step()` 뒤에**, 그리고 **매 iteration마다** 호출한다. (PyTorch의 다른 예제에서 epoch마다 부르는 관습과 헷갈리기 쉽다. 여기서 $t$의 단위는 iteration이다.)
2. `ExponentialLR`은 옵티마이저의 `param_groups`에 있는 `lr`을 직접 곱해 갱신한다. `means` 옵티마이저 하나만 넘겼기 때문에 다른 파라미터의 학습률은 건드리지 않는다.

### 밀도화/MCMC와의 연결

- **DefaultStrategy(densification)**: split/duplicate로 새 Gaussian이 생겨도 스케줄은 스텝 수만 보고 진행하므로, 후반에 태어난 Gaussian은 처음부터 작은 학습률을 받는다. 즉 후반에는 구조가 크게 흔들리지 않는다.
- **MCMCStrategy**: `step_post_backward(..., lr=schedulers[0].get_last_lr()[0])`로 **현재 `means` 학습률을 그대로 받아** 위치에 더할 노이즈 크기를 `noise_scale = lr * noise_lr`(기본 `noise_lr=5e5`)로 정한다. 따라서 학습률이 100배 줄면 탐색 노이즈도 100배 줄어든다. 물리의 **담금질(annealing)**, 즉 온도를 서서히 낮춰 결정 구조를 안정화시키는 과정과 정확히 같은 아이디어다.

---

## 5. 한 줄 정리

$\gamma^{T} = 0.01$이 되도록 $T$제곱근을 취한 값 $\gamma = 0.01^{1/T}$를 `ExponentialLR`에 주면, `means`의 학습률이 학습 전체에 걸쳐 매끄럽게 100분의 1로 줄어들며 "거친 탐색 → 미세 정렬"이 자동으로 일어난다.
