# SH 기저의 정규직교성 — 고교 수학에서 출발하는 설명

> **한 줄 답**: 구면 전체에 대해 $d\Omega=\sin\theta\,d\theta\,d\varphi$로 적분했을 때
> $$\int_{S^2} Y_\ell^m\,Y_{\ell'}^{m'}\,d\Omega=\delta_{\ell\ell'}\,\delta_{mm'}$$
> 이다. 다른 기저끼리의 "내적"은 0, 자기 자신과의 "내적"은 1이다.

이 식에는 고교에서 배우지 않은 개념이 세 가지 숨어 있다.
**(1) 함수끼리의 내적**, **(2) 구면 위에서의 적분과 $\sin\theta$ 가중치**, **(3) $\delta$ 기호**.
각각을 고교 벡터·적분에서 한 단계씩 쌓아 올려 보자.

---

## 1. 벡터 내적 → 함수 내적

### 1.1 고교에서 배운 내적

두 벡터 $\mathbf a=(a_1,a_2,a_3)$, $\mathbf b=(b_1,b_2,b_3)$의 내적은

$$\mathbf a\cdot\mathbf b = a_1b_1+a_2b_2+a_3b_3 = \sum_{i=1}^{3} a_i b_i .$$

- $\mathbf a\cdot\mathbf b=0$ 이면 두 벡터는 **직교**(수직)한다.
- $\mathbf a\cdot\mathbf a=\|\mathbf a\|^2$ 이므로 $\mathbf a\cdot\mathbf a=1$ 이면 **크기가 1**(정규화됨)이다.
- 표준 기저 $\mathbf e_1=(1,0,0),\ \mathbf e_2=(0,1,0),\ \mathbf e_3=(0,0,1)$은 서로 직교하고 크기가 1이다.
  이런 기저를 **정규직교 기저**라고 부른다.

정규직교 기저가 좋은 이유는 **좌표를 내적 한 번으로 뽑아낼 수 있다**는 점이다.

$$\mathbf v = v_1\mathbf e_1+v_2\mathbf e_2+v_3\mathbf e_3 \quad\Longrightarrow\quad v_k=\mathbf v\cdot\mathbf e_k .$$

양변에 $\mathbf e_k$를 내적하면 $\mathbf e_j\cdot\mathbf e_k$가 $j\ne k$일 때 0으로 사라지고 $j=k$ 항만 $1\cdot v_k$로 남기 때문이다. 이 사실을 꼭 기억해 두자 — SH 계수 공식이 정확히 같은 논리다.

### 1.2 성분이 아주 많은 벡터

이제 벡터의 성분 개수를 3개에서 $n$개로 늘려 보자. $\sum_{i=1}^{n}a_ib_i$ 로 내적 정의는 그대로다.

여기서 발상을 하나 바꾸자. 구간 $[0,1]$에서 정의된 함수 $f(x)$를 $n$등분한 점 $x_1,\dots,x_n$에서 값을 읽으면
$(f(x_1),\dots,f(x_n))$이라는 $n$차원 벡터가 된다. 즉 **함수는 "성분이 무한히 많은 벡터"** 로 볼 수 있다.

두 함수 $f,g$를 이렇게 벡터로 보고 내적을 만들면

$$\sum_{i=1}^{n} f(x_i)\,g(x_i)\,\Delta x \;\xrightarrow{\;n\to\infty\;}\; \int_0^1 f(x)\,g(x)\,dx .$$

($\Delta x=1/n$을 곱한 것은 $n$이 커질 때 값이 발산하지 않게 하는 정규화이고, 이는 정확히 고교 미적분의 **구분구적법(리만 합)** 이다.)

그래서 **함수의 내적은 "곱해서 적분"** 으로 정의한다.

$$\langle f,g\rangle=\int f(x)\,g(x)\,dx .$$

- $\langle f,g\rangle=0$ → 두 함수가 **직교**한다.
- $\langle f,f\rangle=1$ → 함수의 "크기"가 1이다(**정규화**).

> **예시(고교 적분으로 확인 가능)**: 구간 $[-\pi,\pi]$에서
> $$\int_{-\pi}^{\pi}\sin x\cos x\,dx=\tfrac12\int_{-\pi}^{\pi}\sin 2x\,dx=0$$
> 이므로 $\sin x$와 $\cos x$는 직교한다. 반면 $\int_{-\pi}^{\pi}\sin^2x\,dx=\pi$ 이므로 $\sin x/\sqrt\pi$ 로 나누면 크기 1이 된다.
> 푸리에 급수가 사인·코사인을 기저로 쓸 수 있는 근거가 바로 이것이고, SH는 같은 아이디어를 **구면 위 함수**로 옮긴 것이다.

---

## 2. 구면 위에서 적분하기 — 왜 $\sin\theta$가 붙는가

SH는 직선 구간이 아니라 **단위 구면 $S^2$** (반지름 1인 구의 표면) 위에 정의된 함수다.
입력은 방향 벡터 $\mathbf d$, $\|\mathbf d\|=1$이다. 그러니 "곱해서 적분"의 적분 영역도 구면 전체가 되어야 한다.

### 2.1 구면 좌표 (고교 기하의 확장)

구면 위 한 점은 두 각도로 지정한다.

- $\theta$(극각): $z$축(북극)에서 얼마나 내려왔는가. $0\le\theta\le\pi$.
- $\varphi$(방위각): $xy$평면에서 $x$축을 기준으로 얼마나 돌았는가. $0\le\varphi<2\pi$ (또는 $-\pi\le\varphi<\pi$).

$$\mathbf d=(\sin\theta\cos\varphi,\ \sin\theta\sin\varphi,\ \cos\theta)$$

고교에서 배운 극좌표 $(r\cos\varphi, r\sin\varphi)$에서, 반지름 $r$ 자리에 "그 높이에서의 위도 원의 반지름" $\sin\theta$가 들어간 것이라고 보면 된다.

### 2.2 넓이 조각의 크기 — 핵심

구간 적분에서는 $\Delta x$ 조각이 모두 같은 길이였다. 그런데 구면을 $(\theta,\varphi)$ 격자로 나누면 **조각의 넓이가 위치마다 다르다.**

점 $(\theta,\varphi)$ 근처에서 $\theta$를 $d\theta$만큼, $\varphi$를 $d\varphi$만큼 움직여 만든 작은 사각형을 생각하자.

- **세로 변**($\theta$ 방향): 반지름 1인 대원 위를 $d\theta$만큼 이동 → 호의 길이 $=1\cdot d\theta=d\theta$.
  (고교 공식: 호의 길이 $=r\theta$)
- **가로 변**($\varphi$ 방향): 이 점은 $z=\cos\theta$ 높이의 **위도 원** 위에 있고, 그 원의 반지름은 $\sin\theta$이다.
  그 위를 $d\varphi$만큼 돌면 호의 길이 $=\sin\theta\,d\varphi$.

따라서 작은 조각의 넓이는

$$d\Omega=(d\theta)\times(\sin\theta\,d\varphi)=\sin\theta\,d\theta\,d\varphi .$$

직관적으로: **지구본의 경선(세로줄)은 적도에서는 넓게 벌어져 있고 극지방에 가면 한 점으로 모인다.**
같은 $\Delta\varphi$라도 적도($\theta=\pi/2$, $\sin\theta=1$)에서는 가장 넓고, 극($\theta=0,\pi$, $\sin\theta=0$)에서는 넓이가 0에 가깝다.
$\sin\theta$ 가중치가 없으면 극 근처의 조각들을 실제보다 훨씬 크게 세는 셈이 된다.

> **검산**: 구 전체 넓이가 $4\pi$(고교 공식 $4\pi r^2$, $r=1$)가 나와야 한다.
> $$\int_0^{2\pi}\!\!\int_0^{\pi}\sin\theta\,d\theta\,d\varphi=\int_0^{2\pi}\Big[-\cos\theta\Big]_0^{\pi}d\varphi=\int_0^{2\pi}2\,d\varphi=4\pi .\ \checkmark$$
> 자료의 `sphere_grid` 함수가 `w = sinθ · (π/nθ) · (2π/nφ)`를 만들고 `Σw ≈ 4π`를 출력하는 것이 정확히 이 검산이다.

### 2.3 구면 위 함수의 내적

이제 1절의 정의를 구면에 옮기면

$$\langle f,g\rangle=\int_{S^2} f(\mathbf d)\,g(\mathbf d)\,d\Omega
=\int_0^{2\pi}\!\!\int_0^{\pi} f(\theta,\varphi)\,g(\theta,\varphi)\,\sin\theta\,d\theta\,d\varphi .$$

기호 $\int_{S^2}$은 "구면 $S^2$ 전체에 대해"라는 뜻이고, $d\Omega$ 안에 이미 $\sin\theta$가 들어 있다.
(한 변수 적분을 두 번 겹쳐 하는 **이중적분**은 고교 범위 밖이지만, "안쪽 $\theta$ 적분을 먼저 하고 결과를 $\varphi$로 다시 적분한다"는 순서로 이해하면 충분하다.)

---

## 3. $\delta$ 기호 — 정규직교를 한 식에 적기

**크로네커 델타** $\delta_{ij}$는 아주 단순한 약속이다.

$$\delta_{ij}=\begin{cases}1 & i=j\\ 0 & i\ne j\end{cases}$$

즉 3×3 단위행렬의 $(i,j)$ 성분이다. 1.1의 정규직교 기저 조건 "$\mathbf e_i\cdot\mathbf e_j$는 같으면 1, 다르면 0"을 한 식으로 쓰면 $\mathbf e_i\cdot\mathbf e_j=\delta_{ij}$ 이다.

SH 기저는 인덱스가 $(\ell,m)$ 두 개이므로 델타도 두 개를 곱한다.

$$\delta_{\ell\ell'}\,\delta_{mm'}=\begin{cases}1 & \ell=\ell' \text{ 이고 } m=m'\\ 0 & \text{둘 중 하나라도 다르면}\end{cases}$$

---

## 4. 세 조각을 합치기

1~3절을 합치면 카드의 답이 그대로 나온다.

$$\boxed{\;\int_{S^2} Y_\ell^m(\mathbf d)\,Y_{\ell'}^{m'}(\mathbf d)\,d\Omega=\delta_{\ell\ell'}\,\delta_{mm'}\;}$$

읽는 법:

| 상황 | 값 | 의미 |
|---|---|---|
| $(\ell,m)\ne(\ell',m')$ | $0$ | 서로 다른 SH 기저는 **직교**한다 (벡터의 수직에 해당) |
| $(\ell,m)=(\ell',m')$ | $1$ | 각 SH 기저의 "크기"는 **1**이다 (정규화) |

정의식의 앞에 붙은 상수 $K_\ell^m=\sqrt{\frac{2\ell+1}{4\pi}\frac{(\ell-|m|)!}{(\ell+|m|)!}}$ 가 하는 일이 바로 두 번째 줄, 즉 $\langle Y_\ell^m,Y_\ell^m\rangle=1$로 맞춰 주는 것이다.

### 4.1 가장 쉬운 예: $Y_0^0$

$Y_0^0=\frac{1}{2\sqrt\pi}$ 는 상수다. 자기 자신과의 내적은

$$\int_{S^2}\Big(\frac{1}{2\sqrt\pi}\Big)^2 d\Omega=\frac{1}{4\pi}\cdot(\text{구 넓이 }4\pi)=1 .\ \checkmark$$

"왜 $\frac{1}{2\sqrt\pi}\approx0.2821$이라는 어색한 상수를 쓰는가"의 답이 여기 있다 — 구 넓이 $4\pi$의 제곱근의 역수로 잡아야 크기가 1이 된다.

### 4.2 직교의 예: $Y_0^0$과 $Y_1^0$

$Y_1^0=\sqrt{\tfrac{3}{4\pi}}\,z=\sqrt{\tfrac{3}{4\pi}}\cos\theta$ 이다.

$$\int_{S^2}Y_0^0\,Y_1^0\,d\Omega
=\frac{1}{2\sqrt\pi}\sqrt{\frac{3}{4\pi}}\int_0^{2\pi}\!\!\int_0^{\pi}\cos\theta\,\sin\theta\,d\theta\,d\varphi .$$

안쪽 적분은 고교 치환적분($u=\sin\theta$)으로 $\int_0^\pi\cos\theta\sin\theta\,d\theta=\big[\tfrac12\sin^2\theta\big]_0^\pi=0$ 이다. 따라서 전체가 0 → 직교. $\checkmark$

직관: $z$는 북반구에서 양, 남반구에서 음이고 상하 대칭이라 구 전체에서 더하면 상쇄된다.

---

## 5. 이 성질이 왜 중요한가 — 계수를 내적 한 번으로

1.1에서 본 "정규직교 기저면 좌표 $=$ 내적"이 그대로 함수에도 성립한다.

$$f(\mathbf d)=\sum_{\ell,m}c_\ell^m\,Y_\ell^m(\mathbf d)$$

양변에 $Y_{\ell'}^{m'}$을 곱해 구면 적분하면, 우변에서 $(\ell,m)\ne(\ell',m')$인 항은 정규직교성 때문에 전부 0이 되고 하나만 살아남는다.

$$c_\ell^m=\int_{S^2} f(\mathbf d)\,Y_\ell^m(\mathbf d)\,d\Omega .$$

연립방정식을 풀 필요 없이 **기저마다 적분 한 번**으로 계수를 얻는다. 3DGS에서 Gaussian 하나의 색을 SH 계수 16개(RGB면 48개)로 표현할 수 있는 근거이며, 자료의 3절에서 $c_0^0\,Y_0^0$이 "함수의 구면 평균"이 되는 것도 이 공식에 $Y_0^0$이 상수임을 넣은 결과다.

---

## 6. 코드에서 어떻게 확인하는가

자료의 다음 두 줄이 위 식을 수치로 검증한다.

```python
w = th.sin() * (math.pi / n_theta) * (2 * math.pi / n_phi)     # dΩ = sinθ dθ dφ 를 격자 하나의 넓이로
gram = torch.einsum("abi,abj,ab->ij", B, B, w)                # Gᵢⱼ = Σ Yᵢ Yⱼ w  ≈ ∫ Yᵢ Yⱼ dΩ
print((gram - torch.eye(16)).abs().max())                      # ≈ 0 이면 정규직교
```

- `w`가 2.2절의 $d\Omega$에 해당한다. 격자의 $\theta$ 간격 $\pi/n_\theta$, $\varphi$ 간격 $2\pi/n_\varphi$에 $\sin\theta$를 곱했다.
- `einsum`은 1.2절의 리만 합 $\sum_i Y_a(x_i)Y_b(x_i)\,\Delta\Omega_i$를 16×16개 쌍에 대해 한 번에 계산한다.
- 결과 행렬(**Gram 행렬**)이 단위행렬 $I_{16}$(= $\delta_{ij}$를 행렬로 쓴 것)에 가까우면 정규직교가 확인된다.

`w`에서 `th.sin()`을 빼고 실행해 보면 Gram 행렬이 단위행렬에서 크게 벗어난다 — $\sin\theta$ 가중치가 장식이 아니라 필수라는 것을 직접 볼 수 있다.

---

## 요약

| 고교 개념 | 대응하는 SH 개념 |
|---|---|
| 벡터 내적 $\sum a_ib_i$ | 함수 내적 $\int f\,g\,d\Omega$ (구분구적법의 극한) |
| $\mathbf e_i\cdot\mathbf e_j=$ 같으면 1, 다르면 0 | $\int Y_\ell^m Y_{\ell'}^{m'}d\Omega=\delta_{\ell\ell'}\delta_{mm'}$ |
| 호의 길이 $r\theta$, 구 넓이 $4\pi$ | $d\Omega=\sin\theta\,d\theta\,d\varphi$, $\int_{S^2}d\Omega=4\pi$ |
| 좌표 $v_k=\mathbf v\cdot\mathbf e_k$ | 계수 $c_\ell^m=\int f\,Y_\ell^m\,d\Omega$ |
