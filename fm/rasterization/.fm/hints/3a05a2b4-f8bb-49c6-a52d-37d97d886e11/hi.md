# 원근 투영의 Jacobian $J$ — 고교 미적분에서 출발해 만들어 보기

## 0. 목표

우리가 만들려는 것은 이 행렬 하나다.

$$J = \begin{bmatrix} f_x/z & 0 & -f_x x/z^2 \\[2pt] 0 & f_y/z & -f_y y/z^2 \end{bmatrix}$$

낯설게 보이지만, 실제로 필요한 지식은 **닮은 삼각형**과 **몫의 미분법** 두 개뿐이다. 순서대로 쌓아 보자.

---

## 1. 핀홀 카메라: 닮은 삼각형으로 $u = f_x x/z + c_x$ 유도

### 1-1. 바늘구멍 사진기의 그림

카메라를 아주 단순하게 모형화하면 **바늘구멍(pinhole)** 하나다. 빛은 그 구멍 하나를 지나 직진해서 필름(센서)에 닿는다.

좌표를 이렇게 잡자.

- 원점 $O$ = 바늘구멍(카메라 중심)
- $z$축 = 카메라가 바라보는 정면 방향 (**깊이**)
- $x$축 = 오른쪽, $y$축 = 아래쪽 (영상 좌표계 관례)
- 센서(이미지 평면)는 $z = f$ 위치에 있다.

3차원 점 $P = (x, y, z)$가 있고, $P$에서 원점을 향해 직진한 빛이 이미지 평면과 만나는 점을 $P' = (x', y', f)$라 하자.

```
        이미지 평면 (z = f)
              |
   P'=(x',f)  ●            ● P = (x, z)
              |           /
        ------O----------+---------→ z
              |         /
        (카메라중심)   깊이 z
              ↑
         초점거리 f
```

### 1-2. 닮음비 한 줄

$O$, $P'$, $P$가 **한 직선 위**에 있으므로, $z$축을 밑변으로 하는 두 직각삼각형

- 작은 삼각형: 밑변 $f$, 높이 $x'$
- 큰 삼각형: 밑변 $z$, 높이 $x$

는 **닮음(AA 닮음: 직각 공유 + 대응각 공유)** 이다. 따라서

$$\frac{x'}{f} = \frac{x}{z} \quad\Longrightarrow\quad x' = f\,\frac{x}{z}, \qquad\text{같은 방식으로}\quad y' = f\,\frac{y}{z}.$$

이것이 **원근(perspective)** 의 전부다. 분모에 $z$가 있으므로 **멀리 있을수록 화면에서 작게 보인다.**

### 1-3. 밀리미터를 픽셀로: $f_x, f_y, c_x, c_y$

위 식의 $x'$는 아직 "센서 위의 물리적 길이(mm)"다. 우리가 원하는 건 **픽셀 좌표**다. 두 가지를 더 보정한다.

1. **단위 환산.** 센서의 픽셀이 가로 방향으로 1mm당 $m_x$개, 세로로 $m_y$개 들어 있다고 하면, 픽셀 단위 초점거리는
   $$f_x = f\,m_x,\qquad f_y = f\,m_y.$$
   픽셀이 정확한 정사각형이 아닐 수 있어서 $f_x \neq f_y$인 경우가 있고, 그래서 두 개를 따로 둔다.
2. **원점 이동.** 이미지의 좌표 원점은 보통 **왼쪽 위 모서리**인데, 위 유도의 원점은 **광축이 지나는 점**이다. 그 차이(주점, principal point)를 $(c_x, c_y)$로 두고 더해 준다. 보통 이미지 중앙 근처, 예를 들어 $640\times480$ 영상이면 $(320, 240)$쯤이다.

합치면 최종 투영 함수 $\pi$가 나온다.

$$\boxed{\;\pi(x,y,z) = \big(u,\ v\big) = \left(f_x\frac{x}{z} + c_x,\ \ f_y\frac{y}{z} + c_y\right)\;}$$

입력 3개 $(x,y,z)$, 출력 2개 $(u,v)$인 함수다. 이 "3 in / 2 out"이 나중에 행렬의 **모양(2×3)** 을 결정한다.

---

## 2. 편미분: "여러 변수 중 하나만 움직일 때의 기울기"

고교 미적분에서 도함수는 이렇게 배웠다. 함수 $g(t)$에 대해

$$g'(t) = \lim_{h\to 0}\frac{g(t+h) - g(t)}{h}$$

이고, 의미는 **"$t$를 아주 조금 움직였을 때 $g$가 그 몇 배로 변하는가"** 이다. 즉 $\Delta g \approx g'(t)\,\Delta t$.

그런데 $u = f_x x/z + c_x$는 변수가 **세 개**($x, y, z$)다. 이럴 때는 이렇게 한다.

> **편미분(partial derivative)**: 관심 있는 변수 하나만 변수로 두고, **나머지는 전부 상수로 취급**해서 평범하게 미분한다. 기호는 $d$ 대신 $\partial$("라운드 디")를 쓴다.

새로운 계산법이 아니다. 예를 들어 $u = f_x x/z$에서 $x$로 편미분할 때는 $z$를 그냥 숫자 $5$처럼 보고

$$u = \frac{f_x}{5}x \quad\Rightarrow\quad \frac{\partial u}{\partial x} = \frac{f_x}{5}$$

처럼 **고교에서 배운 그 미분을 그대로** 하면 된다. 그 $5$를 다시 $z$라고 쓰면 끝이다.

**왜 이걸 하는가?** 3D 점이 $(\Delta x, \Delta y, \Delta z)$만큼 아주 조금 움직였을 때 화면 위 점이 얼마나 움직이는지를 알고 싶기 때문이다. 각 방향의 기여를 따로 재서 더하면 되고(1차 근사),

$$\Delta u \;\approx\; \frac{\partial u}{\partial x}\Delta x + \frac{\partial u}{\partial y}\Delta y + \frac{\partial u}{\partial z}\Delta z.$$

이걸 두 출력 $u, v$에 대해 나란히 쓴 것이 바로 **Jacobian 행렬**이다.

---

## 3. 여섯 개의 편미분을 하나씩

$$u = f_x\frac{x}{z} + c_x, \qquad v = f_y\frac{y}{z} + c_y.$$

### 3-1. $\partial u/\partial x = f_x/z$

$y, z$를 상수로 본다. 그러면 $u$는 $x$에 대한 **일차함수**다.

$$u = \underbrace{\frac{f_x}{z}}_{\text{상수(기울기)}}\cdot x + \underbrace{c_x}_{\text{상수}} \quad\Longrightarrow\quad \frac{\partial u}{\partial x} = \frac{f_x}{z}.$$

### 3-2. $\partial u/\partial y = 0$

$u$의 식 어디에도 $y$가 **등장하지 않는다.** $y$에 대해서는 $u$가 상수함수이므로

$$\frac{\partial u}{\partial y} = 0.$$

물리적으로도 당연하다. 점을 순수하게 아래로 내리면 화면에서도 순수하게 아래로 내려갈 뿐, **가로 위치는 변하지 않는다.** 이 "$x$는 $u$만, $y$는 $v$만 건드린다"는 성질이 나중에 $J$의 두 개의 $0$이 된다.

### 3-3. $\partial u/\partial z = -f_x x/z^2$ (몫의 미분)

이번엔 $x$가 상수, $z$가 변수다. 고교에서 배운 **몫의 미분법**

$$\left(\frac{A}{B}\right)' = \frac{A'B - AB'}{B^2}$$

을 $A = f_x x$ (상수, $A' = 0$), $B = z$ ($B' = 1$)에 적용하면

$$\frac{\partial u}{\partial z} = \frac{0\cdot z - (f_x x)\cdot 1}{z^2} = -\frac{f_x x}{z^2}.$$

몫의 미분을 안 써도 된다. $u = (f_x x)\cdot z^{-1}$로 보고 지수법칙 $\left(z^{-1}\right)' = -z^{-2}$를 쓰면

$$\frac{\partial u}{\partial z} = (f_x x)\cdot(-z^{-2}) = -\frac{f_x x}{z^2}$$

로 같은 답이다.

**부호가 음수인 이유**: $z$가 커진다 = 점이 **멀어진다** = 화면에서 **중심 쪽으로 당겨진다**. $x > 0$(중심 오른쪽)인 점이라면 멀어질수록 $u$가 줄어드니 도함수가 음수인 게 맞다.

### 3-4. $v$에 대해서도 똑같이

$v = f_y y/z + c_y$는 $u$와 완전히 대칭이다.

$$\frac{\partial v}{\partial x} = 0,\qquad \frac{\partial v}{\partial y} = \frac{f_y}{z},\qquad \frac{\partial v}{\partial z} = -\frac{f_y\,y}{z^2}.$$

---

## 4. 여섯 개를 표로 묶으면 2×3 행렬

편미분 여섯 개를 **행 = 출력, 열 = 입력** 규칙으로 표에 채운다.

| | $\partial/\partial x$ | $\partial/\partial y$ | $\partial/\partial z$ |
|---|---|---|---|
| $u$ | $f_x/z$ | $0$ | $-f_x x/z^2$ |
| $v$ | $0$ | $f_y/z$ | $-f_y y/z^2$ |

표의 숫자 부분만 떼어내면 그것이 **Jacobian 행렬**이다.

$$J = \begin{bmatrix} \dfrac{\partial u}{\partial x} & \dfrac{\partial u}{\partial y} & \dfrac{\partial u}{\partial z} \\[10pt] \dfrac{\partial v}{\partial x} & \dfrac{\partial v}{\partial y} & \dfrac{\partial v}{\partial z} \end{bmatrix} = \begin{bmatrix} f_x/z & 0 & -f_x x/z^2 \\[4pt] 0 & f_y/z & -f_y y/z^2 \end{bmatrix}$$

### 왜 하필 $2 \times 3$인가

- **행 2개** = 출력의 개수. 화면 좌표는 $(u, v)$ 두 개다.
- **열 3개** = 입력의 개수. 3D 점은 $(x, y, z)$ 세 개다.

일반적으로 입력 $n$개, 출력 $m$개인 함수의 Jacobian은 $m \times n$이다. 정사각형일 필요가 전혀 없다. 여기서는 3차원을 2차원으로 **납작하게 눌러 담는** 함수이므로 가로로 넓은 직사각 행렬이 된다.

### 행렬로 쓰면 뭐가 좋은가

§2에서 본 근사식

$$\Delta u \approx \frac{\partial u}{\partial x}\Delta x + \frac{\partial u}{\partial y}\Delta y + \frac{\partial u}{\partial z}\Delta z, \qquad \Delta v \approx \frac{\partial v}{\partial x}\Delta x + \cdots$$

두 줄이, 행렬-벡터 곱 **한 줄**로 압축된다.

$$\begin{bmatrix}\Delta u\\ \Delta v\end{bmatrix} \;\approx\; J\begin{bmatrix}\Delta x\\ \Delta y\\ \Delta z\end{bmatrix}$$

즉 $J$는 **"3D에서의 작은 움직임을 화면에서의 작은 움직임으로 바꿔 주는 환율표"** 다. 원래 함수 $\pi$는 $z$가 분모에 있는 비선형 함수지만, **한 점 근방만 보면** 이렇게 곱셈 하나로 근사된다. 고교에서 배운 "미분계수 = 접선의 기울기"의 다변수 버전이며, 접선이 **접평면**으로 바뀐 것뿐이다.

---

## 5. 각 항이 말해 주는 것 (직관)

### 5-1. $f_x/z$ — 멀어질수록 $1/z$로 작아진다

$J$의 첫 열은 $(f_x/z,\ 0)^\top$이다. "$x$ 방향으로 1cm 옮기면 화면에서 $f_x/z$ 픽셀 움직인다"는 뜻이다.

$f_x = 300$이라 하고 두 경우를 비교하자.

| 깊이 $z$ | $x$로 1 이동 시 화면 이동량 |
|---|---|
| $z = 2$ | $300/2 = 150$ px |
| $z = 8$ | $300/8 = 37.5$ px |

**4배 멀어지면 화면 이동량은 1/4로 준다.** 달리는 차 안에서 창밖을 볼 때, 가까운 가드레일은 휙휙 지나가는데 멀리 있는 산은 거의 안 움직이는 그 현상이다. **모션 패럴랙스**라고 부르며, 그 수학적 정체가 바로 $1/z$ 인자다.

$3\times 3$ 카메라 행렬 $K$로 투영할 때의 스케일이 $f_x/z$라는 것과 완전히 같은 이야기다. 다만 여기서는 **점의 위치**가 아니라 **위치의 변화량**에 적용된다는 점이 다르다.

### 5-2. $-f_x x/z^2$ — $z$ 이동은 화면 중심에서 먼 점을 더 많이 움직인다

셋째 열은 $(-f_x x/z^2,\ -f_y y/z^2)^\top$이고, **$x$와 $y$에 비례**한다. 여기서 두 가지가 읽힌다.

**(1) 광축 위의 점($x = y = 0$)은 앞뒤로 움직여도 화면에서 안 움직인다.** $x = 0$이면 셋째 열의 첫 성분이 $0$이다. 정면 정중앙에 있는 물체를 향해 곧장 걸어가면, 그 물체는 **커지기만 할 뿐 화면에서 제자리**다.

**(2) 화면 가장자리의 점일수록 앞뒤 이동에 크게 반응한다.** $|x|$가 클수록 $|-f_x x/z^2|$도 커진다. 터널을 통과할 때 화면 중앙 소실점은 가만히 있는데 **가장자리 벽면이 바깥으로 확 흘러나가는** 그 느낌이다. 컴퓨터 비전에서는 이 방사형 패턴을 **확장 초점(focus of expansion)** 이라고 한다.

**(3) $1/z^2$이라 깊이에 훨씬 예민하다.** 첫 열은 $1/z$인데 셋째 열은 $1/z^2$이다. 가까운 물체($z$ 작음)의 앞뒤 움직임은 화면을 격렬하게 흔들고, 먼 물체는 거의 반응하지 않는다.

**(4) $J$의 랭크는 2다.** 3차원 정보가 2차원으로 눌리므로, **화면 위 점을 전혀 움직이지 않는 3D 방향**이 반드시 하나 존재한다($J\mathbf{d} = \mathbf{0}$인 $\mathbf{d}$). 그 방향은 바로 원점에서 $P$를 향하는 **시선 방향** $(x, y, z)$ 자신이다. 실제로 대입해 보면

$$\frac{f_x}{z}x + 0\cdot y + \left(-\frac{f_x x}{z^2}\right)z = \frac{f_x x}{z} - \frac{f_x x}{z} = 0.$$

당연하다 — 시선을 따라 앞뒤로 미끄러지는 점은 언제나 화면의 **같은 픽셀**에 찍힌다.

---

## 6. $c_x, c_y$는 왜 $J$에 나타나지 않는가

$$u = f_x\frac{x}{z} + c_x$$

에서 $c_x$는 **입력 $(x,y,z)$와 아무 상관 없는 고정된 숫자**(카메라를 만들 때 정해지는 값)다. 고교에서 배운 그대로,

$$\frac{d}{dt}\big[g(t) + C\big] = g'(t) + 0 = g'(t)$$

상수항은 미분하면 사라진다. 그래서 $c_x, c_y$는 여섯 개의 편미분 중 **어디에도** 나타나지 않는다.

**기하학적 의미**: $c_x, c_y$를 더하는 것은 이미지 전체를 통째로 **평행이동**하는 것이다. 원점을 어디에 두든(왼쪽 위든 중앙이든) **점들 사이의 상대적 위치 관계는 그대로**다. Jacobian은 "얼마나 움직이는가(변화량)"만 재는 도구이므로, 전체를 똑같이 옮기는 상수는 관심 밖이다.

같은 이유로, 카메라 좌표계로 옮기는 단계 $\mu_c = R\mu + t$의 평행이동 $t$도 (그 단계의) Jacobian에는 나타나지 않는다.

---

## 7. 이 $J$가 실제로 어디에 쓰이나 — 3D Gaussian Splatting

3DGS에서 이 행렬이 필요한 이유는 한 문장이다.

> **투영 $\pi$는 비선형($z$가 분모)이라, 3D Gaussian을 통과시키면 그 결과가 더 이상 Gaussian이 아니다.** 그래서 중심점 $\mu_c$ 근방에서 $\pi$를 $J$로 **1차 근사(선형화)** 한 뒤, 선형변환이 Gaussian을 Gaussian으로 보낸다는 성질을 이용한다.

고교 확률과 통계에서 배운 성질 — 확률변수 $X$에 상수 $a$를 곱하면 분산은 $a^2$배가 된다($\mathrm{Var}(aX) = a^2\mathrm{Var}(X)$) — 의 다차원 버전이 이것이다.

$$\Sigma_{2D} = J\,\Sigma_c\,J^\top + \varepsilon I$$

$a^2 \to J(\cdot)J^\top$으로 바뀌었을 뿐 완전히 같은 발상이다. $\Sigma_c$가 $3\times 3$이고 $J$가 $2\times 3$이므로, 크기를 따라가 보면

$$\underbrace{(2\times 3)}_{J}\cdot\underbrace{(3\times 3)}_{\Sigma_c}\cdot\underbrace{(3\times 2)}_{J^\top} = (2\times 2)$$

로 화면 위 2D 타원의 공분산이 나온다. 이 근사법을 **EWA splatting**이라 부른다.

에셋 노트북(`rasterization_walkthrough.py` §3)이 이 식을 그대로 코드로 옮긴다.

```python
means2d = torch.stack([fx * x / z + cx, fy * y / z + cy], dim=-1)   # π(μ_c)
O = torch.zeros_like(z)
J = torch.stack([fx / z, O, -fx * x / z**2,
                 O, fy / z, -fy * y / z**2], -1).reshape(-1, 2, 3)   # ← 우리가 유도한 J
cov2d = J @ covars_c @ J.transpose(-1, -2)
cov2d = cov2d + eps2d * torch.eye(2, device=means.device)            # 최소 0.3px² 블러
```

`torch.stack([...]).reshape(-1, 2, 3)`은 여섯 개 편미분을 표(§4)의 **읽는 순서 그대로** 늘어놓고 $2\times3$으로 접는 것이다. 참조 구현 `gsplat/cuda/_torch_impl.py`의 `_persp_proj`도 변수 이름만 다를 뿐 똑같다.

```python
J = torch.stack([fx / tz, O, -fx * tx / tz2,
                 O, fy / tz, -fy * ty / tz2], dim=-1).reshape(..., 2, 3)
cov2d = torch.einsum("...ij,...jk,...kl->...il", J, covars, J.transpose(-1, -2))
```

한 가지 실전 장치가 더 있다. `_persp_proj`는 $J$를 만들기 **전에** $x/z$, $y/z$를 시야각의 1.3배 안으로 잘라낸다(clamp).

```python
tx = tz * torch.clamp(tx / tz, min=-lim_x_neg, max=lim_x_pos)
```

이유는 §5-2에서 이미 봤다 — $-f_x x/z^2$ 항이 $|x|$에 비례하므로, **화면 밖 아주 멀리 있는 점**에서는 이 값이 폭주해 1차 근사가 완전히 무너진다. 어차피 안 보일 점이니 값을 시야 경계로 눌러 두는 것이다. 3DGS 원본(Inria) 구현부터 있던 트릭이다.

---

## 8. 요약

1. **닮은 삼각형** 한 번으로 $x' /f = x/z$, 여기에 픽셀 환산 $f_x, f_y$와 주점 $c_x, c_y$를 붙이면
   $$\pi(x,y,z) = (f_x x/z + c_x,\; f_y y/z + c_y).$$
2. **편미분**은 "나머지 변수를 상수로 놓고 하는 평범한 미분"이다. 새 계산법이 아니다.
3. $u$를 $x$로 미분하면 일차함수의 기울기 $f_x/z$, $y$로 미분하면 $y$가 식에 없으니 $0$, $z$로 미분하면 몫의 미분법으로 $-f_x x/z^2$. $v$도 대칭적으로 같다.
4. **출력 2개 × 입력 3개** 이므로 결과는 $2\times3$ 행렬:
   $$J = \begin{bmatrix} f_x/z & 0 & -f_x x/z^2 \\ 0 & f_y/z & -f_y y/z^2 \end{bmatrix}.$$
5. **직관**: $f_x/z$ 는 "멀수록 화면 이동이 $1/z$로 줄어든다"(모션 패럴랙스). $-f_x x/z^2$ 는 "앞뒤 이동은 중심에서 먼 점일수록 크게 흔들고, 정중앙 점은 안 흔든다"(확장 초점). 시선 방향은 $J$의 영공간이라 화면이 전혀 안 움직인다.
6. **$c_x, c_y$는 상수라 미분하면 사라진다.** 화면 전체의 평행이동은 변화량에 영향을 주지 않는다.
7. 3DGS는 이 $J$로 $\Sigma_{2D} = J\Sigma_c J^\top + \varepsilon I$를 계산한다 — $\mathrm{Var}(aX) = a^2\mathrm{Var}(X)$의 행렬 버전이다.
