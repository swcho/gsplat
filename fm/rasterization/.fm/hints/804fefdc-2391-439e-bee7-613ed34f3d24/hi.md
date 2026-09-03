# σ는 "타원 좌표로 잰 거리의 제곱, 그 절반"

목표 공식:

$$\sigma_i = \tfrac12\left(a\,dx^2 + c\,dy^2\right) + b\,dx\,dy$$

여기서 $(a,b,c)$는 conic(2D 공분산의 역행렬 성분), $(dx,dy)$는 픽셀 중심과 Gaussian 중심의 차이다.
왜 하필 이런 모양인지, 왜 $dx^2,\,dy^2$ 앞에만 $\tfrac12$이 붙고 $dx\,dy$ 앞에는 안 붙는지를 1차원 정규분포에서부터 쌓아 올려 보자.

---

## 1. 1차원 정규분포: "표준편차를 자로 삼아 잰 거리"

고등학교에서 배운 정규분포 $N(\mu,\sigma_{\text{sd}}^2)$의 확률밀도는

$$f(x) = \frac{1}{\sqrt{2\pi}\,\sigma_{\text{sd}}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma_{\text{sd}}^2}\right)$$

> ⚠️ 이름 충돌 주의: 여기 나오는 $\sigma_{\text{sd}}$는 **표준편차**이고, 문제에서 묻는 $\sigma$는 **지수 안에 들어가는 숫자**다.
> 이 문서에서는 표준편차를 $\sigma_{\text{sd}}$로, 렌더링 코드의 값을 $\sigma$로 구분해 쓴다.

지수부만 떼어 보자. $z = \dfrac{x-\mu}{\sigma_{\text{sd}}}$ 라고 두면

$$-\frac{(x-\mu)^2}{2\sigma_{\text{sd}}^2} = -\frac{1}{2}z^2$$

$z$는 "중심에서 표준편차 몇 개만큼 떨어졌나"다. 즉 **자(단위)를 cm가 아니라 그 분포의 표준편차로 바꿔서 잰 거리**다.
이 $z^2$이 바로 **마할라노비스 거리의 제곱**의 1차원 버전이다.

- $z=1$ → 1시그마 지점, 밀도는 $e^{-0.5}\approx 0.61$배
- $z=3$ → 3시그마 지점, 밀도는 $e^{-4.5}\approx 0.011$배

핵심: **정규분포의 "높이"는 오직 $z^2$ 하나로 결정된다.** 그래서 $z^2$의 절반을 $\sigma$라고 이름 붙이면 밀도는 그냥 $e^{-\sigma}$가 된다.

---

## 2. 2차원: 축이 기울면 x, y를 따로 재면 안 된다

이제 화면 위의 2D Gaussian이다. 중심에서의 변위를 벡터로 쓰자.

$$\mathbf{d} = \begin{bmatrix} dx \\ dy \end{bmatrix}$$

### (a) 축이 정렬된 경우 — 쉬운 경우

타원의 장축/단축이 x축, y축과 나란하다면 x방향 표준편차 $\sigma_x$, y방향 $\sigma_y$로 각각 재서 더하면 된다.

$$r^2 = \frac{dx^2}{\sigma_x^2} + \frac{dy^2}{\sigma_y^2}$$

$r^2 = 1$인 점들의 자취는 $\dfrac{dx^2}{\sigma_x^2}+\dfrac{dy^2}{\sigma_y^2}=1$, 즉 기하에서 배운 **타원의 표준형**이다.

### (b) 축이 기울어진 경우 — 교차항이 필요하다

Gaussian splat은 3D 타원체를 화면에 투영한 것이므로, 화면 위 타원이 x/y축과 나란할 이유가 전혀 없다. 45° 기운 길쭉한 타원을 생각해 보자.

이때 (a)의 공식을 그대로 쓰면 틀린다. 왜냐하면 x가 크면 y도 큰 쪽으로 "정보가 새기" 때문이다 — 즉 x와 y가 **상관되어 있다**. 상관을 무시하고 각 축을 독립으로 재면, 실제로는 타원 **안**에 있는 점을 바깥으로 판정하게 된다.

기울어진 타원의 방정식은 고등학교 기하에서도 나오듯 $Adx^2 + Bdx\,dy + Cdy^2 = 1$ 꼴, 즉 **$dx\,dy$ 교차항이 반드시 붙는다**. 이 교차항이 곧 "x와 y의 상관"을 표현한다.

---

## 3. 2×2 역공분산 행렬로 한 번에 쓰기

x, y 사이의 상관까지 담는 그릇이 **공분산 행렬** $\Sigma$ (2×2, 대칭)이다. 마할라노비스 거리의 제곱은 이것의 **역행렬**로 정의된다.

$$r^2 = \mathbf{d}^\top \Sigma^{-1} \mathbf{d}$$

여기서 $\Sigma^{-1}$을 성분으로 쓰자. $\Sigma$가 대칭이면 역행렬도 대칭이므로 자유도는 3개뿐이다.

$$\Sigma^{-1} = \begin{bmatrix} a & b \\ b & c \end{bmatrix}$$

**이 $(a,b,c)$가 코드의 `conics`다.** (공분산 3개가 아니라 그 역행렬 3개를 미리 계산해 둔다. 2차형식 $=$ 상수가 원뿔곡선(conic)의 방정식이라서 붙은 이름이다.)

### 행렬 곱을 손으로 풀어 보기

$$\mathbf{d}^\top \Sigma^{-1} \mathbf{d}
= \begin{bmatrix} dx & dy \end{bmatrix}
\begin{bmatrix} a & b \\ b & c \end{bmatrix}
\begin{bmatrix} dx \\ dy \end{bmatrix}$$

먼저 오른쪽 두 개부터:

$$\begin{bmatrix} a & b \\ b & c \end{bmatrix}\begin{bmatrix} dx \\ dy \end{bmatrix}
= \begin{bmatrix} a\,dx + b\,dy \\ b\,dx + c\,dy \end{bmatrix}$$

이제 왼쪽 행벡터와 내적:

$$\begin{aligned}
\mathbf{d}^\top \Sigma^{-1}\mathbf{d}
&= dx\,(a\,dx + b\,dy) + dy\,(b\,dx + c\,dy)\\
&= a\,dx^2 + b\,dx\,dy + b\,dx\,dy + c\,dy^2\\
&= a\,dx^2 + 2b\,dx\,dy + c\,dy^2
\end{aligned}$$

**$b$가 두 번 등장해서 $2b$가 되는 것**(오른쪽 위 성분에서 한 번, 왼쪽 아래 성분에서 한 번)이 이 문제의 전부다.

---

## 4. 절반을 취하면 답변 공식이 정확히 나온다

1차원에서 지수부가 $-\tfrac12 z^2$이었듯, 다변수 정규분포의 지수부도

$$\exp\!\left(-\tfrac{1}{2}\,\mathbf{d}^\top\Sigma^{-1}\mathbf{d}\right)$$

이다. 그래서 그 $\tfrac12$까지 미리 먹여서 $\sigma$라는 하나의 값으로 정의한다.

$$\begin{aligned}
\sigma &= \tfrac12\,\mathbf{d}^\top\Sigma^{-1}\mathbf{d}\\
&= \tfrac12\left(a\,dx^2 + 2b\,dx\,dy + c\,dy^2\right)\\
&= \tfrac12\left(a\,dx^2 + c\,dy^2\right) + \underbrace{\tfrac12 \cdot 2}_{=1}\,b\,dx\,dy\\
&= \boxed{\ \tfrac12\left(a\,dx^2 + c\,dy^2\right) + b\,dx\,dy\ }
\end{aligned}$$

즉 **대각항 $dx^2, dy^2$에는 $\tfrac12$이 남고, 교차항 $dx\,dy$에는 $2\times\tfrac12=1$이라 $\tfrac12$이 사라진다.**
코드가 비대칭적으로 보이는 이유가 이것이다. 실수해서 빠뜨린 $\tfrac12$이 아니라, $2b$의 2와 약분된 것이다.

```python
a, b, c = conics[g]
sigma = 0.5 * (a * dx * dx + c * dy * dy) + b * dx * dy
```

$(dx,dy)$는 **픽셀 중심**과 Gaussian 중심 `means2d`의 차다. 픽셀 $(px, py)$의 중심은 정수 좌표가 아니라 $(px+0.5,\ py+0.5)$이므로 코드에도 `+ 0.5`가 들어간다.

> 부호는 걱정하지 않아도 된다. `rasterize_naive`는 `mean - pixel`, `_torch_impl._rasterize_to_pixels`는 `pixel - mean`으로 계산하지만, $\sigma$는 $dx,dy$에 대해 **2차식**이라 둘 다 동시에 부호가 바뀌면 $dx^2, dy^2, dx\,dy$ 모두 그대로다. 결과는 같다.

---

## 5. 그래서 알파 블렌딩이 이렇게 간단해진다

$\sigma$ 하나만 있으면 Gaussian의 값은 그냥 $e^{-\sigma}$이고, 불투명도 $o_i$를 곱하면 알파다.

$$\alpha_i = \min\!\left(0.99,\ o_i\,e^{-\sigma_i}\right)$$

$$C_p = \sum_i c_i\,\alpha_i\,T_i,\qquad T_{i+1}=T_i(1-\alpha_i)$$

$\sigma$는 지수·나눗셈·제곱근 없이 **곱셈 5번, 덧셈 2번**으로 끝난다. 픽셀 × Gaussian 개수만큼 반복되는 가장 안쪽 루프이므로, 이 값싼 형태가 성능상 결정적이다.

### σ는 등고선 하나에 대응한다

$\sigma$가 같은 점들의 자취 $\tfrac12(a\,dx^2+c\,dy^2)+b\,dx\,dy = \text{const}$ 는 하나의 **타원**이다.

| $\sigma$ | $r=\sqrt{2\sigma}$ (몇 시그마) | $e^{-\sigma}$ |
|---|---|---|
| $0$ | 0 (정중앙) | 1 |
| $0.5$ | 1시그마 | 0.607 |
| $2$ | 2시그마 | 0.135 |
| $4.5$ | 3시그마 | 0.011 |
| $5.544$ | 3.33시그마 | **0.00391 ≈ 1/255** |

마지막 줄이 gsplat의 상수 `GAUSSIAN_EXTEND = 3.33`의 정체다.

$$\sigma = \tfrac12 \times 3.33^2 = 5.5445,\qquad e^{-5.5445} = 0.003909 \approx \frac{1}{255} = 0.003922$$

8비트 색에서 $1/255$보다 작은 기여는 어차피 반올림하면 0이다. 그래서 반경을 $3.33\sqrt{\lambda}$ 정도로 자르고(`radii`), 블렌딩 루프에서도 `alpha < ALPHA_THRESHOLD(=1/255)`이면 건너뛴다. **"3.33 시그마"는 임의로 고른 숫자가 아니라 $\ln 255 \approx 5.541$에서 역산된 값이다.**

---

## 6. 왜 `sigma < 0`이면 건너뛰는가

CUDA 커널(`eval_gaussian_weight`)과 참조 구현 모두 이 조건이 있다.

```cpp
const float sigma = 0.5f * (conic.x*dx*dx + conic.z*dy*dy) + conic.y*dx*dy;
out.valid = !(sigma < 0.f || alpha < ALPHA_THRESHOLD);
```

```python
valid = (sigma >= 0) & (alpha >= ALPHA_THRESHOLD) & ~done
```

**수학적으로는 절대 일어나지 않아야 하는 일이다.** $\Sigma$는 진짜 공분산 행렬이므로 양의 정부호(positive definite)이고, 그러면 $\Sigma^{-1}$도 그렇다. 양의 정부호란 정확히 "0이 아닌 모든 $\mathbf{d}$에 대해 $\mathbf{d}^\top\Sigma^{-1}\mathbf{d} > 0$"이라는 뜻이다. 2×2에서 이 조건은 판별하기 쉽다.

$$a > 0 \quad\text{그리고}\quad ac - b^2 > 0$$

(고등학교식으로 보면, $y$를 고정하고 $dx$에 대한 이차식으로 볼 때 판별식 $ (2b\,dy)^2 - 4a(c\,dy^2) = 4dy^2(b^2-ac) < 0 $이라 항상 양수라는 뜻이다.)

**그런데 실제로는 음수가 나올 수 있다.**
- $\Sigma_{2D}$가 거의 납작한(det $\approx 0$) 타원이면 역행렬 성분이 폭발적으로 커진다.
- float32로 역행렬을 구하는 과정에서 반올림 오차가 생겨 $ac-b^2$가 아주 살짝 음수가 되는 등, PD가 깨진 conic이 나올 수 있다.
- 그런 conic으로 $\sigma$를 계산하면 음수가 되고, $e^{-\sigma}$는 **1보다 큰 값으로 폭발**한다. 그러면 $\alpha$가 곧장 0.99로 포화되어 화면에 거대한 불투명 얼룩이 찍힌다.

즉 `sigma < 0`은 **"정상 입력이면 절대 걸리지 않지만, 걸렸다면 수치가 깨졌다는 신호이므로 그 Gaussian을 조용히 버린다"**는 안전장치다. 알파를 clamp하는 것보다 아예 기여를 0으로 만드는 편이 안전하다.

---

## 7. 수치로 검산해 보기

### (a) 축 정렬 예 ($b=0$)

$\sigma_x = 2,\ \sigma_y = 1$, 상관 없음.

$$\Sigma = \begin{bmatrix} 4 & 0 \\ 0 & 1\end{bmatrix},\qquad
\Sigma^{-1} = \begin{bmatrix} 0.25 & 0 \\ 0 & 1\end{bmatrix}
\;\Rightarrow\; (a,b,c) = (0.25,\ 0,\ 1)$$

픽셀이 중심에서 $\mathbf{d}=(2,1)$만큼 떨어져 있다면 (x로 1시그마, y로 1시그마):

$$\sigma = \tfrac12\left(0.25\cdot 4 + 1\cdot 1\right) + 0\cdot 2\cdot 1 = \tfrac12(1+1) = 1$$

$r^2 = 2\sigma = 2$, 즉 각 축으로 1시그마씩. §2(a)의 $\dfrac{4}{4}+\dfrac{1}{1}=2$와 정확히 일치한다. $b=0$일 때 공식은 축 정렬 공식으로 자연스럽게 퇴화한다.

### (b) 45° 기울어진 예 ($b \ne 0$)

같은 타원(장축 $\sigma=2$, 단축 $\sigma=1$)을 45° 돌리면

$$\Sigma = \begin{bmatrix} 2.5 & 1.5 \\ 1.5 & 2.5\end{bmatrix},\qquad
\Sigma^{-1} = \frac{1}{2.5^2-1.5^2}\begin{bmatrix} 2.5 & -1.5 \\ -1.5 & 2.5\end{bmatrix}
= \begin{bmatrix} 0.625 & -0.375 \\ -0.375 & 0.625\end{bmatrix}$$

$$(a,b,c) = (0.625,\ -0.375,\ 0.625)$$

이제 **장축 방향으로 정확히 1시그마** 떨어진 점을 잡자. 장축 방향 단위벡터는 $\left(\tfrac{1}{\sqrt2},\tfrac{1}{\sqrt2}\right)$, 그 방향 표준편차는 2이므로 $\mathbf{d} = (\sqrt2,\ \sqrt2) \approx (1.414,\ 1.414)$.

$$\begin{aligned}
\tfrac12\left(a\,dx^2 + c\,dy^2\right) &= \tfrac12(0.625\cdot 2 + 0.625\cdot 2) = \tfrac12(1.25+1.25) = 1.25\\
b\,dx\,dy &= -0.375 \cdot \sqrt2 \cdot \sqrt2 = -0.75\\
\sigma &= 1.25 - 0.75 = \mathbf{0.5}
\end{aligned}$$

$\sigma = 0.5$ → $r = \sqrt{2\sigma} = 1$ → 정확히 **1시그마**. 기대한 값이다. ✅

**교차항을 빠뜨리면 어떻게 되나?** $\Sigma$의 대각만 보고 $\sigma_x^2=\sigma_y^2=2.5$라고 착각해서 §2(a) 공식을 쓰면

$$r^2_{\text{틀림}} = \frac{2}{2.5} + \frac{2}{2.5} = 1.6 \quad(\text{참값 } 1.0)$$

$\sigma$로는 $0.8$ vs $0.5$, 밀도로는 $e^{-0.8}=0.449$ vs $e^{-0.5}=0.607$ — **35% 어두워진다.** 게다가 이 오차는 방향에 따라 반대로도 나타난다. 단축 방향 $\mathbf{d}=(\tfrac{1}{\sqrt2},-\tfrac{1}{\sqrt2})$에서는 $b\,dx\,dy$가 $+$가 되어 참값이 커진다. 그 결과 교차항을 버리면 기울어진 타원이 **축 정렬된 둥근 얼룩**으로 뭉개진다. $b$는 있으면 좋은 보정이 아니라, 타원의 방향 정보 그 자체다.

---

## 8. 한 줄 요약

$\sigma$는 **"이 픽셀이 Gaussian 중심에서 몇 시그마 떨어졌나"를 제곱해서 반으로 나눈 값**이다.
기울어진 타원에서는 x, y를 따로 재면 안 되므로 역공분산 $\begin{bmatrix} a & b \\ b & c\end{bmatrix}$의 2차형식으로 재고, 그 전개 $a\,dx^2+2b\,dx\,dy+c\,dy^2$의 절반이 곧
$\tfrac12(a\,dx^2+c\,dy^2)+b\,dx\,dy$다. 이 하나의 스칼라로 Gaussian 값이 $e^{-\sigma}$가 되고, 알파 블렌딩 전체가 그 위에 올라간다.
