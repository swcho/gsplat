# backward 커널이 투과율 $T$를 어떻게 복원하는가

## 0. 무엇이 문제인가

한 픽셀의 색은 그 픽셀을 덮는 Gaussian들을 **앞에서 뒤로** 하나씩 섞어서 만든다.

$$C = \sum_{i=0}^{n-1} c_i\,\alpha_i\,T_i, \qquad T_0 = 1,\quad T_{i+1} = T_i(1-\alpha_i)$$

- $c_i$: $i$번째 Gaussian의 색
- $\alpha_i \in [0,1)$: 그 Gaussian이 이 픽셀을 얼마나 가리는가
- $T_i$: **투과율(transmittance)**. 앞의 $i$개를 통과하고 남은 빛의 비율

학습을 하려면 $\partial C/\partial \alpha_i$ 같은 미분값이 필요한데, 뒤에서 보겠지만 이 미분에는 **중간값 $T_i$가 그대로 등장한다**. 그런데 $T_i$를 forward에서 전부 저장해 두면 메모리가 `픽셀 수 × 그 픽셀의 Gaussian 수`가 되어 터진다. 그래서 gsplat의 backward 커널은 **저장하지 않고 다시 만들어 낸다**. 그 방법이 나눗셈 되감기다.

---

## 1. 곱셈 점화식은 나눗셈으로 정확히 되돌아간다

$T_{i+1} = T_i(1-\alpha_i)$ 는 **공비가 매 항마다 바뀌는 등비수열**이다. 고교 등비수열 $a_{n+1} = r\,a_n$ 에서 $a_n = a_{n+1}/r$ 로 한 칸 뒤로 갈 수 있는 것과 똑같이,

$$T_i = \frac{T_{i+1}}{1-\alpha_i}$$

즉 $1-\alpha_i \neq 0$ 이기만 하면 **곱셈의 역연산인 나눗셈으로 한 칸씩 정확히 되감을 수 있다**. 덧셈 점화식($T_{i+1}=T_i - \alpha_i$)이었다면 뺄셈으로 되감을 때 큰 수에서 작은 수를 빼는 자리 손실이 생기지만, 곱셈/나눗셈은 부동소수점에서 상대오차만 남기므로 되감기에 훨씬 안전하다.

전개해 쓰면 더 분명하다.

$$T_n = \prod_{j=0}^{n-1}(1-\alpha_j) \quad\Longrightarrow\quad T_i = \frac{T_n}{\prod_{j=i}^{n-1}(1-\alpha_j)}$$

**뒤에서 앞으로** 훑으면서 $(1-\alpha_j)$ 를 하나씩 나눠 주면 곱의 뒷부분이 하나씩 벗겨진다.

### 구체 예: $\alpha = (0.5,\ 0.2,\ 0.8)$

앞으로(forward):

| $i$ | $\alpha_i$ | $T_i$ | $T_{i+1}=T_i(1-\alpha_i)$ |
|---|---|---|---|
| 0 | 0.5 | $1$ | $1 \times 0.5 = 0.5$ |
| 1 | 0.2 | $0.5$ | $0.5 \times 0.8 = 0.4$ |
| 2 | 0.8 | $0.4$ | $0.4 \times 0.2 = 0.08$ |

forward는 **마지막 $T_3 = 0.08$ 하나만** 저장한다(실제 커널은 `render_alphas`로 $1-T_\text{final}$ 을 저장하고, backward에서 `T_final = 1 - render_alphas[pix_id]` 로 되살린다).

뒤로(backward), $T \leftarrow T/(1-\alpha)$:

| 처리 순서 | $\alpha_i$ | 시작 $T$ | $T/(1-\alpha_i)$ | 복원된 값 |
|---|---|---|---|---|
| $i=2$ | 0.8 | $0.08$ | $0.08/0.2$ | $T_2 = 0.4$ ✓ |
| $i=1$ | 0.2 | $0.4$ | $0.4/0.8$ | $T_1 = 0.5$ ✓ |
| $i=0$ | 0.5 | $0.5$ | $0.5/0.5$ | $T_0 = 1.0$ ✓ |

forward의 표와 정확히 같은 값이 되돌아왔다. 커널 코드가 하는 일이 이 한 줄이다.

```cpp
// RasterizeToPixels3DGSSerialBatchBwd.cu
float ra = 1.0f / fmaxf(MIN_ONE_MINUS_ALPHA, 1.0f - alpha);
T       *= ra;                    // T_{i+1} -> T_i
```

같은 타일 목록(`flatten_ids`)을, forward가 멈춘 지점(`last_ids`, 코드의 `bin_final`)부터 **역순으로** 훑기 때문에 $\alpha_i$ 는 그때그때 다시 계산해서 얻는다(위치·conic·opacity로부터 $\alpha$ 를 재평가). 즉 **$\alpha$도 저장하지 않고 재계산**한다.

---

## 2. 왜 $T_i$가 필요한가 — 연쇄법칙으로 확인

손실 $L$ 에 대해 우리가 원하는 것은 $\partial L/\partial \alpha_i$ 인데, 연쇄법칙으로

$$\frac{\partial L}{\partial \alpha_i} = \frac{\partial L}{\partial C}\cdot\frac{\partial C}{\partial \alpha_i}$$

이므로 결국 $\partial C/\partial \alpha_i$ 를 구하면 된다. $C$ 는 $\alpha_i$ 에 두 가지 경로로 의존한다.

**(a) 자기 항** $c_i \alpha_i T_i$ — 여기서 $T_i$ 는 $\alpha_0,\dots,\alpha_{i-1}$ 만 쓰므로 $\alpha_i$ 와 무관한 상수다:

$$\frac{\partial}{\partial \alpha_i}\big(c_i \alpha_i T_i\big) = c_i T_i$$

**(b) 뒤쪽 항 전부** $\sum_{j>i} c_j \alpha_j T_j$ — 여기서 $T_j\ (j>i)$ 는 모두 인수 $(1-\alpha_i)$ 를 품고 있다. $j>i$ 에 대해 $T_j = T_i(1-\alpha_i)\prod_{k=i+1}^{j-1}(1-\alpha_k)$ 이므로 $T_j$ 는 $\alpha_i$ 의 일차식이고,

$$\frac{\partial T_j}{\partial \alpha_i} = -\frac{T_j}{1-\alpha_i}$$

(합의 미분은 항별 미분의 합이라는 고교 규칙을 그대로 쓴 것이다.) 따라서

$$\frac{\partial C}{\partial \alpha_i} = c_i T_i - \frac{1}{1-\alpha_i}\sum_{j>i} c_j\,\alpha_j\,T_j$$

여기서 $T_i$ 가 **명시적으로 필요**하다는 것이 핵심이다. $T_i$ 없이는 이 식을 못 쓴다.

### 뒤쪽 누적 색 $S$ 도 같이 되감는다

뒤쪽 합을 "뒤쪽만 따로 렌더링한 색"으로 정규화해 두면 식이 깔끔해진다.

$$S_{i+1} \;\equiv\; \frac{1}{T_{i+1}}\sum_{j>i} c_j\,\alpha_j\,T_j$$

$T_{i+1} = T_i(1-\alpha_i)$ 를 대입하면 $\dfrac{1}{1-\alpha_i}\sum_{j>i} c_j\alpha_j T_j = \dfrac{T_{i+1}S_{i+1}}{1-\alpha_i} = T_i S_{i+1}$ 이므로

$$\boxed{\;\frac{\partial C}{\partial \alpha_i} = T_i\big(c_i - S_{i+1}\big)\;}$$

의미가 좋다: **"내 색이 내 뒤에 보이던 색보다 밝으면 $\alpha$ 를 키워라"**. 그리고 $T_i$ 가 곱해져 있으니 이미 가려진 뒤쪽 Gaussian은 gradient도 작다.

커널은 $\sum_{j>i} c_j\alpha_j T_j$ 를 `buffer[k]` 에 뒤→앞 순서로 누적하고, 위 유도 그대로 `buffer * ra` 로 $T_i S_{i+1}$ 을 만든다.

```cpp
v_alpha += (rgbs_batch[t*CDIM+k] * T          // c_i * T_i
          - buffer[k] * ra) * v_render_c[k];  // - (뒤쪽 누적)/(1-α_i) = T_i S_{i+1}
...
buffer[k] += rgbs_batch[t*CDIM+k] * fac;      // fac = α_i T_i, 다음(=앞쪽) 스텝을 위해 누적
```

즉 되감는 상태 변수는 **$T$ 와 $S$(=`buffer`) 두 개**이고, 둘 다 픽셀당 스칼라/색 하나뿐이다. 픽셀당 Gaussian 수와 무관하게 메모리가 $O(1)$.

### 나머지 gradient는 여기서 파생된다

$v_\alpha \equiv \partial L/\partial\alpha_i$ 하나가 나오면 나머지는 순전히 국소적인 체인 룰이다($\alpha = o\cdot v$, $v=e^{-\sigma}$, $\sigma = \tfrac12(a\,\delta_x^2 + c\,\delta_y^2) + b\,\delta_x\delta_y$):

$$v_{c_i} = \alpha_i T_i\,\bar C,\qquad v_{o} = v\,v_\alpha,\qquad v_\sigma = -o\,v\,v_\alpha$$
$$v_{a} = \tfrac12 v_\sigma \delta_x^2,\quad v_b = v_\sigma \delta_x\delta_y,\quad v_c = \tfrac12 v_\sigma \delta_y^2$$
$$v_{x} = v_\sigma(a\delta_x + b\delta_y),\qquad v_{y} = v_\sigma(b\delta_x + c\delta_y)$$

코드의 `v_rgb_local = fac * v_render_c`, `v_sigma = -opac*vis*v_alpha`, `v_conic_local`, `v_xy_local`, `v_opacity_local = vis * v_alpha` 가 그대로 이 식들이다. **모두 $T_i$ 하나에서 흘러나온다.**

---

## 3. 저장 대신 재계산 — 어떤 거래인가

| | 전부 저장 | 되감기(gsplat) |
|---|---|---|
| 메모리 | $O(\text{픽셀} \times \text{픽셀당 Gaussian})$ | $O(\text{픽셀})$ — 최종 $T$, `last_ids` 만 |
| 추가 연산 | 없음 | $\alpha$ 재계산 + 나눗셈 1회/Gaussian |
| 오차 | 없음 | 부동소수점 되감기 오차(아래) |

1080p 이미지에서 픽셀당 평균 수백 개 Gaussian이 섞인다고 하면 $2\times10^6 \times 300 \times 4\text{B} \approx 2.4\,$GB — 그것도 한 카메라, 한 채널 기준. 실제로는 GPU 메모리에 올릴 수 없다. 반면 나눗셈 한 번은 GPU에서 몇 사이클이다. **GPU는 연산이 싸고 메모리 대역폭이 비싸므로**, 저장을 재계산으로 바꾸는 거래는 거의 언제나 이득이다(이 아이디어 자체는 딥러닝의 gradient checkpointing과 같은 계열이다).

---

## 4. 왜 $\alpha < 1$ 이어야만 하는가 — `MAX_ALPHA = 0.99`

되감기의 유일한 전제는 **$1-\alpha_i$ 로 나눌 수 있어야 한다**는 것이다.

- $\alpha_i = 1$ 이면 $1-\alpha_i = 0$: forward에서 $T$ 가 $0$ 이 되고, $0/0$ 이므로 **원래 $T_i$ 를 복원할 정보 자체가 사라진다**(수학적으로 정보 손실이지 구현 실수가 아니다). 실제로는 $T/0 = \infty$ 또는 `NaN` 이 나와 그 픽셀의 gradient 전체가 오염되고, `atomicAdd` 로 Gaussian 파라미터에 퍼진다.
- $\alpha_i$ 가 1에 매우 가까우면($0.9999$ 등) 나눗셈 계수 $1/(1-\alpha_i)$ 가 $10^4$ 로 커져서, forward에서 이미 잃은 유효숫자를 **증폭**한다.

그래서 gsplat은 forward에서 아예 $\alpha \leftarrow \min(\alpha, 0.99)$ 로 자른다(`MAX_ALPHA = 0.99`). 이러면 되감기 계수는 항상 $1/(1-\alpha) \le 100$ 으로 묶인다. 게다가 `TRANSMITTANCE_THRESHOLD = 1e-4 = (1-\text{MAX\_ALPHA})^2` 로 잡혀 있어서, "가장 불투명한 Gaussian도 두 번은 겹쳐야 조기 종료된다"는 성질과 $T$ 하한이 동시에 보장된다. backward 쪽에도 2차 방어선이 있다.

```cpp
#define MIN_ONE_MINUS_ALPHA 1e-6f          // include/Common.h
float ra = 1.0f / fmaxf(MIN_ONE_MINUS_ALPHA, 1.0f - alpha);
```

한 Gaussian이 픽셀을 완전히 가리지 못하게 막는 이 상한은, 렌더링 품질을 거의 해치지 않으면서(0.99와 1.0의 시각적 차이는 없다) **backward를 수학적으로 가능하게 만드는** 장치다.

---

## 5. 부동소수점 오차 감각

float32는 유효숫자가 약 7자리, 즉 연산 하나의 상대오차가 $\varepsilon \approx 6\times10^{-8}$ 수준이다(machine epsilon $2^{-24}$).

곱셈/나눗셈만 반복하면 오차는 **상대오차로 누적**된다. $n$ 번 연산 후 최악의 경우 상대오차는 대략

$$(1+\varepsilon)^n - 1 \approx n\varepsilon$$

- $n = 100$: $\approx 6\times10^{-6}$
- $n = 500$: $\approx 3\times10^{-5}$

게다가 실전에서는 오차가 같은 방향으로만 쌓이지 않고 랜덤 워크에 가까워서 $\sqrt{n}\,\varepsilon$ 정도에 머무는 경우가 많다. gradient 스케일에서 $10^{-5}$ 상대오차는 SGD 노이즈에 묻히는 수준이므로 실용상 문제가 없다. (float64면 $\varepsilon\approx10^{-16}$ 이라 $n=500$ 이어도 $10^{-14}$.)

주의할 점은 이 결론이 **$\alpha$ 가 잘려 있을 때만** 성립한다는 것이다. $\alpha \to 1$ 이면 위의 $n\varepsilon$ 추정이 깨지고 오차가 폭발한다. `expy.py` 에서 $\alpha=0.999$ 와 $\alpha=1.0$ 케이스로 이걸 직접 확인한다.

---

## 6. 한 줄 요약

> forward는 최종 $T$ 와 마지막 인덱스만 남긴다. backward는 같은 타일 목록을 **뒤→앞**으로 걸으며 `T /= (1-α)` 로 $T_i$ 를, `buffer` 로 뒤쪽 누적 색 $S_{i+1}$ 을 동시에 되감아 $\partial C/\partial\alpha_i = T_i(c_i - S_{i+1})$ 을 만든다. 곱셈 점화식이라 나눗셈으로 정확히 되돌아가고, 그것이 가능하도록 $\alpha$ 는 `MAX_ALPHA = 0.99` 로 잘려 있다.
