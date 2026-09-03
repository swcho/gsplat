# SH 계수는 왜 "내적 한 번"으로 나오는가

> **질문**: SH 기저가 정규직교하기 때문에 계수 $c_\ell^m$은 어떻게 구할 수 있는가?
> **답**: 함수와 기저의 내적(사영) 한 번으로 구한다. $c_\ell^m=\int_{S^2} f(\mathbf d)\,Y_\ell^m(\mathbf d)\,d\Omega$

이 카드의 핵심은 "정규직교 기저에서는 좌표를 **내적**으로 뽑아낼 수 있다"는 원리다.
고교 기하의 **벡터 정사영**에서 출발해 구면 위 함수까지 세 단계로 쌓아 올려 보자.

---

## 1단계 — 고교 벡터: 좌표는 내적으로 뽑는다

평면 벡터 $\mathbf v = (3, 4)$는 표준 기저 $\mathbf e_1=(1,0)$, $\mathbf e_2=(0,1)$로

$$
\mathbf v = 3\,\mathbf e_1 + 4\,\mathbf e_2
$$

라고 쓸 수 있다. 여기서 "3"이라는 좌표를 어떻게 얻는가? 내적을 취하면 된다.

$$
\mathbf v\cdot\mathbf e_1 = (3\mathbf e_1 + 4\mathbf e_2)\cdot\mathbf e_1
= 3\,(\mathbf e_1\cdot\mathbf e_1) + 4\,(\mathbf e_2\cdot\mathbf e_1)
= 3\cdot 1 + 4\cdot 0 = 3
$$

여기서 쓰인 성질이 딱 두 가지다.

| 성질 | 식 | 이름 |
|---|---|---|
| 자기 자신과의 내적은 1 | $\mathbf e_i\cdot\mathbf e_i = 1$ | **정규**(normal, 길이 1) |
| 서로 다른 기저끼리 내적은 0 | $\mathbf e_i\cdot\mathbf e_j = 0\ (i\neq j)$ | **직교**(orthogonal) |

두 성질을 합쳐 **정규직교**라 하고, 기호로는 크로네커 델타 $\delta_{ij}$로 쓴다.

$$
\mathbf e_i\cdot\mathbf e_j=\delta_{ij}=
\begin{cases}1 & i=j\\ 0 & i\neq j\end{cases}
$$

정규직교 기저라면 **어떤** 벡터든 $i$번째 좌표는 "$\mathbf e_i$와 내적 한 번"이다.
고교 기하에서 배운 말로는, $\mathbf v$를 $\mathbf e_i$ 방향으로 **정사영**한 길이가 바로 좌표다.
($|\mathbf e_i| = 1$이므로 정사영 길이 $|\mathbf v|\cos\theta = \mathbf v\cdot\mathbf e_i$.)

> 만약 기저가 직교하지 **않았다면** — 예를 들어 $\mathbf e_2$ 대신 $(1,1)$을 썼다면 — 내적 $\mathbf v\cdot\mathbf e_1$에 다른 항의 기여가 섞여 들어와 좌표 하나를 알기 위해 연립방정식을 풀어야 한다. 직교성이 "한 번에 한 좌표씩 떼어낼 수 있게" 해 주는 것이다.

---

## 2단계 — 함수도 벡터다: 내적을 "적분"으로 바꾼다

벡터 $(v_1, v_2, \dots, v_n)$의 내적은 **성분별로 곱해서 더한 것**이다.

$$
\mathbf v\cdot\mathbf w = \sum_{i=1}^{n} v_i\,w_i
$$

함수 $f(x)$를 "각 점 $x$마다 값 $f(x)$를 성분으로 갖는, 성분이 무한히 많은 벡터"로 생각해 보자.
성분이 연속적으로 무한히 많으니 "더한다"는 자연스럽게 **적분**이 된다.

$$
\langle f, g\rangle = \int f(x)\,g(x)\,dx
$$

이것이 함수의 내적이다. 고교 미적분에서 배운 정적분 그대로이며, 새로 배울 것은 "이걸 내적으로 부른다"는 관점만이다.

이 관점에서 1단계의 표를 그대로 옮겨 쓸 수 있다.

- 함수 $g_1, g_2, \dots$가 **정규직교**: $\displaystyle\int g_i(x)\,g_j(x)\,dx = \delta_{ij}$
- 어떤 함수 $f = \sum_i c_i\,g_i$의 **계수**: $\displaystyle c_i = \langle f, g_i\rangle = \int f(x)\,g_i(x)\,dx$

푸리에 급수가 정확히 이 구조다. $\sin(nx), \cos(nx)$들이 $[0, 2\pi]$에서 서로 직교하기 때문에, 푸리에 계수는 "$f$에 $\cos(nx)$를 곱해 적분"하는 한 번으로 나온다.

---

## 3단계 — 구면 위 함수와 SH

3DGS에서 다루는 것은 **방향 $\mathbf d$에 따라 색이 달라지는 함수** $f(\mathbf d)$다.
$\mathbf d$는 길이 1인 3차원 방향 벡터이므로, 정의역은 단위구면 $S^2$이다.

구면 위 함수의 내적은 "구면 전체에 대해 곱해서 적분"이다.

$$
\langle f, g\rangle = \int_{S^2} f(\mathbf d)\,g(\mathbf d)\,d\Omega
$$

$d\Omega$는 구면의 아주 작은 조각(입체각)의 넓이다. 구면좌표 $(\theta, \varphi)$로는 $d\Omega = \sin\theta\,d\theta\,d\varphi$이고, 구면 전체를 적분하면 $\int_{S^2} d\Omega = 4\pi$(반지름 1인 구의 표면적)가 된다.

Spherical Harmonics $Y_\ell^m$은 이 내적에 대해 **정규직교**하도록 만들어진 함수들이다.

$$
\int_{S^2} Y_\ell^m(\mathbf d)\,Y_{\ell'}^{m'}(\mathbf d)\,d\Omega = \delta_{\ell\ell'}\,\delta_{mm'}
$$

즉 $(\ell, m) = (\ell', m')$이면 1, 하나라도 다르면 0이다. 노트북에서는 이 성질을 격자 구적으로 직접 확인한다
(16개 기저의 내적을 모두 계산한 $16\times16$ Gram 행렬이 단위행렬과 거의 같음을 출력).

---

## 4단계 — 급수 양변에 $Y_\ell^m$을 곱해 적분하면 왜 다른 항이 전부 사라지는가

이제 카드의 질문에 정면으로 답한다. 함수 $f$가 SH 급수로 쓰인다고 하자.

$$
f(\mathbf d) = \sum_{\ell'}\sum_{m'} c_{\ell'}^{m'}\,Y_{\ell'}^{m'}(\mathbf d)
$$

**계수 하나, 예컨대 $c_\ell^m$을 알고 싶다.** 1단계에서 $\mathbf v\cdot\mathbf e_1$을 계산했듯, 양변에 $Y_\ell^m(\mathbf d)$를 곱하고 구면 전체에서 적분한다.

$$
\int_{S^2} f(\mathbf d)\,Y_\ell^m(\mathbf d)\,d\Omega
= \int_{S^2}\Bigl(\sum_{\ell', m'} c_{\ell'}^{m'}\,Y_{\ell'}^{m'}(\mathbf d)\Bigr)\,Y_\ell^m(\mathbf d)\,d\Omega
$$

적분과 합의 순서를 바꾸고(유한합이면 항상 가능), 상수 $c_{\ell'}^{m'}$을 적분 밖으로 꺼내면

$$
= \sum_{\ell', m'} c_{\ell'}^{m'}\underbrace{\int_{S^2} Y_{\ell'}^{m'}(\mathbf d)\,Y_\ell^m(\mathbf d)\,d\Omega}_{=\ \delta_{\ell\ell'}\delta_{mm'}}
$$

밑줄 친 적분은 정규직교성에 의해

- $(\ell', m') \neq (\ell, m)$인 **모든** 항: $0$ → 그 항은 통째로 사라진다.
- $(\ell', m') = (\ell, m)$인 **단 하나의** 항: $1$ → $c_\ell^m \cdot 1$만 남는다.

따라서

$$
\boxed{\;c_\ell^m = \int_{S^2} f(\mathbf d)\,Y_\ell^m(\mathbf d)\,d\Omega\;}
$$

1단계의 $\mathbf v\cdot\mathbf e_1 = 3\cdot1 + 4\cdot 0$과 완전히 같은 계산이다. 16개(혹은 무한히 많은) 항이 있어도 연립방정식을 풀 필요가 없고, **원하는 계수 하나를 내적 한 번으로 따로 떼어낼 수 있다**. 이것이 "사영(projection)"이라 부르는 이유다 — $f$라는 벡터를 $Y_\ell^m$ 방향으로 정사영한 길이가 $c_\ell^m$이다.

---

## 5단계 — 가장 쉬운 예: DC 계수 $c_0^0$

$Y_0^0 = \dfrac{1}{2\sqrt\pi}$는 방향에 무관한 **상수**다. 공식에 넣으면

$$
c_0^0 = \int_{S^2} f(\mathbf d)\,\frac{1}{2\sqrt\pi}\,d\Omega
= \frac{1}{2\sqrt\pi}\int_{S^2} f\,d\Omega
$$

이 계수로 복원한 0차 항은

$$
c_0^0\,Y_0^0 = \frac{1}{4\pi}\int_{S^2} f\,d\Omega = \overline f\quad(\text{구면 전체에서의 } f\text{의 평균})
$$

즉 "함수에 상수 기저를 곱해 적분한 것"은 곧 **평균**이다. 평균이란 게 "모든 값을 더해 개수(여기선 넓이 $4\pi$)로 나눈 것"임을 떠올리면 자연스럽다.
그리고 $\ell \ge 1$인 모든 $Y_\ell^m$은 $Y_0^0$(상수)과 직교하므로 구면 적분이 0 — 고차 항은 평균이 0인 "변동"만 담는다는 뜻이다. 3DGS가 DC 계수를 "시점과 무관한 기본색"으로 취급하는 이유가 여기 있다.

---

## 6단계 — 실제 코드와 3DGS에서는

노트북의 `project_to_sh` 함수는 위 적분을 그대로 격자 구적으로 계산한다.

$$
c_k \approx \sum_{\text{grid}} f(\mathbf d)\,Y_k(\mathbf d)\,w(\mathbf d)
$$

여기서 $w(\mathbf d)$는 격자 한 칸의 넓이(작은 $d\Omega$)이며, 전부 더하면 $4\pi$가 된다. "곱해서 더한다"는 내적의 모습이 코드에 그대로 드러난다 (`einsum("abk,abd,ab->kd", bases, f, w)`).

단, 한 가지 주의할 점이 있다. **3DGS 학습에서는 이 적분을 직접 계산하지 않는다.** 실제 장면의 참 함수 $f(\mathbf d)$(어떤 방향에서 봐도 그 Gaussian이 어떤 색인지)를 모르기 때문이다.
대신 여러 카메라에서 관측한 색과 렌더 결과의 차이를 역전파해 계수를 조정한다. 그래도 이 카드의 공식은 여전히 중요하다.

- $c_\ell^m$이 "**$f$의 $Y_\ell^m$ 방향 성분**"이라는 의미를 알려 준다 → 학습이 수렴하면 결국 이 값에 가까워진다.
- DC 계수 초기화 `sh0 = (rgb − 0.5) / C0`가 "평균색을 상수 기저에 사영한 값"이라는 해석을 가능하게 한다.
- 각 계수가 **서로 독립적**(직교)이므로 하나를 바꿔도 다른 방향 성분이 흔들리지 않는다 → 학습이 안정적이다.

---

## 한 줄 정리

정규직교 기저에서는 **좌표 = 그 기저와의 내적**이다 ($\mathbf v\cdot\mathbf e_i = v_i$).
함수의 내적은 적분이므로, 구면 위 함수 $f$의 SH 계수는
$c_\ell^m = \int_{S^2} f\,Y_\ell^m\,d\Omega$ — 급수에 $Y_\ell^m$을 곱해 적분하면 $\delta_{\ell\ell'}\delta_{mm'}$ 덕분에 다른 항은 모두 0이 되고 $c_\ell^m$만 남기 때문이다.
