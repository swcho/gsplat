# DC 항 $c_0^0Y_0^0$은 왜 "함수의 구면 평균"인가 — 고교 수학에서 출발하는 설명

## 0. 결론 먼저

구면 위에 정의된 함수 $f$를 Spherical Harmonics(SH)로 전개했을 때, 가장 낮은 차수($\ell=0$, "DC") 항만으로 복원한 값은

$$
c_0^0\,Y_0^0=\frac{1}{4\pi}\int_{S^2} f\,d\Omega=\overline f
$$

즉 **$f$를 구면 전체에서 평균한 값**이다. 이 문서는 이 한 줄이 왜 성립하는지, 고등학교 미적분에서 배운 "평균 = 적분 ÷ 길이"에서 출발해 한 단계씩 쌓아 올린다.

---

## 1. 구간 평균: 적분 ÷ 길이

고교 미적분에서 배운 함수의 평균값부터 시작한다. 구간 $[a,b]$에서 $f(x)$의 평균은

$$
\overline f=\frac{1}{b-a}\int_a^b f(x)\,dx
$$

이다. 직관은 이렇다. 구간을 아주 작은 조각 $dx$들로 나누고, 각 조각의 값 $f(x)$에 조각의 길이 $dx$를 곱해서 다 더한 것($\int f\,dx$)을 전체 길이($b-a$)로 나눈다. "가중치가 균등한 평균"이다.

$f$가 상수 $A$라면 $\int_a^b A\,dx=A(b-a)$이므로 평균은 당연히 $A$가 된다. 이 "상수를 적분하면 상수 × 길이"라는 사실을 아래에서 계속 쓴다.

## 2. 구면 평균: 적분 ÷ 표면적 $4\pi$

이제 정의역을 구간에서 **단위구의 표면** $S^2$(반지름 1인 구의 껍질)로 바꾼다. $f$는 "방향 $\mathbf d$마다 값 하나"를 주는 함수다. 예: 하늘 방향은 밝고 땅 방향은 어두운 환경 조명.

구간의 "작은 조각 길이 $dx$" 역할을 하는 것이 **작은 조각 넓이 $d\Omega$** 다. 구면 좌표(극각 $\theta$, 방위각 $\varphi$)로 쓰면 $d\Omega=\sin\theta\,d\theta\,d\varphi$인데, 지금 중요한 것은 공식보다 성질 하나다:

$$
\int_{S^2} 1\,d\Omega=(\text{단위구의 표면적})=4\pi
$$

(고교 기하: 반지름 $r$ 구의 표면적은 $4\pi r^2$, $r=1$이면 $4\pi$.)

따라서 1절과 완전히 같은 논리로

$$
\overline f=\frac{1}{4\pi}\int_{S^2} f\,d\Omega
$$

가 **구면 평균**의 정의다. 상수 $A$의 구면 평균은 $\frac{1}{4\pi}\cdot A\cdot 4\pi=A$로, 역시 자기 자신이다.

## 3. $Y_0^0$은 상수다 — 그리고 그 값이 $\frac{1}{2\sqrt\pi}$인 이유

SH 기저 $Y_\ell^m$은 구면 위 함수들이고, 서로 **정규직교**한다:

$$
\int_{S^2} Y_\ell^m\,Y_{\ell'}^{m'}\,d\Omega=
\begin{cases}1 & (\ell,m)=(\ell',m')\\ 0 & \text{그 외}\end{cases}
$$

"정규(normal)"는 자기 자신과의 적분이 1, "직교(orthogonal)"는 서로 다른 기저끼리의 적분이 0이라는 뜻이다. 고교 벡터의 내적으로 비유하면, 두 함수의 "내적"을 $\langle g,h\rangle=\int gh\,d\Omega$로 정의했을 때 기저들이 길이 1이고 서로 수직인 단위벡터 묶음이라는 말이다.

가장 낮은 차수 $\ell=0$의 기저 $Y_0^0$은 **방향에 무관한 상수**다. 그 값을 $k$라 두고 "정규" 조건을 적용하자:

$$
\int_{S^2}(Y_0^0)^2\,d\Omega=1
\;\Longrightarrow\;
\int_{S^2}k^2\,d\Omega=k^2\cdot 4\pi=1
\;\Longrightarrow\;
k=\frac{1}{\sqrt{4\pi}}=\frac{1}{2\sqrt\pi}\approx 0.2821
$$

2절의 "상수 적분 = 상수 × $4\pi$"를 그대로 썼다. 3DGS/gsplat 코드의 `C0 = 0.28209479...`가 바로 이 숫자다.

## 4. 계수 $c_0^0$을 계산해 보기 — 한 줄씩

SH 계수는 "$f$를 기저에 사영(내적)"해서 얻는다:

$$
c_\ell^m=\int_{S^2} f\,Y_\ell^m\,d\Omega
$$

$\ell=0$일 때 $Y_0^0$은 상수 $\frac{1}{2\sqrt\pi}$이므로 적분 밖으로 나온다:

$$
\begin{aligned}
c_0^0 &=\int_{S^2} f\,Y_0^0\,d\Omega \\
      &=\int_{S^2} f\cdot\frac{1}{2\sqrt\pi}\,d\Omega \\
      &=\frac{1}{2\sqrt\pi}\int_{S^2} f\,d\Omega
\end{aligned}
$$

이제 "DC 항만으로 복원"한다는 것은 급수 $\sum c_\ell^mY_\ell^m$에서 첫 항 $c_0^0Y_0^0$만 남기는 것이다:

$$
\begin{aligned}
c_0^0\,Y_0^0 &=\left(\frac{1}{2\sqrt\pi}\int_{S^2} f\,d\Omega\right)\cdot\frac{1}{2\sqrt\pi} \\
             &=\frac{1}{(2\sqrt\pi)^2}\int_{S^2} f\,d\Omega \\
             &=\frac{1}{4\pi}\int_{S^2} f\,d\Omega \\
             &=\overline f
\end{aligned}
$$

마지막 줄이 2절의 구면 평균 정의와 정확히 같다. 정리하면:

> $Y_0^0$이 상수이므로, "$Y_0^0$에 사영한다"는 계산이 "$f$를 적분한다"와 상수배만 다르고, 그 상수 두 번의 곱 $(Y_0^0)^2=\frac{1}{4\pi}$이 정확히 표면적의 역수가 되어 평균이 튀어나온다.

## 5. 고차 항은 평균에 아무 기여도 하지 않는다

그러면 나머지 항 $c_\ell^mY_\ell^m$ ($\ell\ge1$)의 평균은? 각 $Y_\ell^m$은 $Y_0^0$과 직교하므로

$$
\int_{S^2} Y_\ell^m\,d\Omega
=\frac{1}{Y_0^0}\int_{S^2} Y_\ell^m\,Y_0^0\,d\Omega
=\frac{1}{Y_0^0}\cdot 0=0\qquad(\ell\ge1)
$$

즉 $\ell\ge1$ 기저는 모두 **구면 평균이 0**이다 (양의 영역과 음의 영역이 정확히 상쇄). 따라서 $f$는

$$
f=\underbrace{c_0^0Y_0^0}_{\text{평균 } \overline f}+\underbrace{\sum_{\ell\ge1}c_\ell^mY_\ell^m}_{\text{평균이 0인 변동}}
$$

으로 깨끗하게 나뉜다. DC 항이 "기준 밝기", 고차 항이 "방향에 따라 얼마나 더 밝거나 어두운가"를 맡는다.

## 6. 1차원 비유: 푸리에 급수의 $a_0/2$

같은 현상이 푸리에 급수에도 있다. 주기 $2\pi$인 함수를

$$
f(x)=\frac{a_0}{2}+\sum_{n\ge1}\big(a_n\cos nx+b_n\sin nx\big),\qquad a_0=\frac{1}{\pi}\int_0^{2\pi}f(x)\,dx
$$

로 쓰면 첫 항은

$$
\frac{a_0}{2}=\frac{1}{2\pi}\int_0^{2\pi}f(x)\,dx=\text{(구간 } [0,2\pi]\text{에서의 평균)}
$$

이고, $\cos nx,\ \sin nx$ ($n\ge1$)는 한 주기 적분이 0이라 평균에 기여하지 않는다. 신호처리에서 이 상수항을 **DC(direct current, 직류) 성분**이라 부르는데, SH의 $\ell=0$ 항이 "DC 계수"라 불리는 이유가 여기 있다. 표면적 $4\pi$가 주기 길이 $2\pi$ 자리를 대신하는 것만 다르다.

## 7. 3DGS에서의 의미

3DGS는 Gaussian마다 색을 SH 계수 16개 × RGB로 저장한다. 그중 DC 계수 `sh0`에 $C_0$를 곱한 값은 **그 Gaussian을 모든 방향에서 봤을 때의 평균 색**이다. 그래서

- 초기화 `sh0 = (rgb − 0.5) / C0`: SfM 점의 색 `rgb`를 "평균 색"으로 삼겠다는 뜻 (`+0.5` 오프셋을 뺀 뒤 $C_0$로 나눠 계수로 환산).
- 고차 계수 `shN`은 평균을 바꾸지 않고 시점에 따른 변동(하이라이트 등)만 담으므로, 학습률을 낮게 두고 천천히 배워도 기본색이 흔들리지 않는다.

## 요약

| 단계 | 식 | 쓰인 사실 |
|---|---|---|
| 구간 평균 | $\overline f=\frac{1}{b-a}\int_a^b f\,dx$ | 고교 미적분 |
| 구면 평균 | $\overline f=\frac{1}{4\pi}\int_{S^2}f\,d\Omega$ | 단위구 표면적 $=4\pi$ |
| $Y_0^0$ 값 | $k^2\cdot4\pi=1\Rightarrow Y_0^0=\frac{1}{2\sqrt\pi}$ | 정규 조건 $\int(Y_0^0)^2=1$ |
| DC 계수 | $c_0^0=\frac{1}{2\sqrt\pi}\int f\,d\Omega$ | 상수는 적분 밖으로 |
| DC 복원 | $c_0^0Y_0^0=\frac{1}{4\pi}\int f\,d\Omega=\overline f$ | $(Y_0^0)^2=\frac{1}{4\pi}$ |
| 고차 항 | $\int Y_\ell^m\,d\Omega=0\ (\ell\ge1)$ | $Y_0^0$과의 직교성 |
