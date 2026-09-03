# `meta["conics"]`가 뭔지, 고등학교 수학에서 출발해 쌓아 보기

## 0. 결론 먼저

`conics`는 **`[C,N,3]` 모양의 텐서**로, 각 카메라 $C$개 × 각 Gaussian $N$개마다
2D 공분산 행렬 $\Sigma_2$의 **역행렬** $\Sigma_2^{-1}$을 구해서, 그 안의 서로 다른 값 **3개 $(a,b,c)$**만 뽑아 저장한 것이다.

래스터화(픽셀 칠하기) 단계에서 실제로 GPU가 쓰는 것은 $\Sigma_2$가 아니라 이 $(a,b,c)$다.

왜 하필 "역행렬"이고, 왜 하필 "3개"고, 왜 이름이 "conic(원뿔곡선)"인지를 아래에서 차근차근 만들어 보자.

---

## 1. 1차원 정규분포에서 출발

고등학교 확률과 통계에서 본 정규분포의 모양은 이렇게 생겼다.

$$f(x) \;\propto\; e^{-\frac{x^2}{2\sigma^2}}$$

($\propto$는 "앞에 붙는 상수는 지금 신경 안 쓴다"는 뜻이다. 그림의 모양만 보자.)

지수 안의 $\dfrac{x^2}{2\sigma^2}$를 잘 보면, **분산 $\sigma^2$이 분모에 있다**. 즉 실제로 곱해지는 건 $\sigma^2$이 아니라 $\dfrac{1}{\sigma^2}$, 다시 말해 **분산의 역수**다.

$$e^{-\frac{1}{2}\cdot \frac{1}{\sigma^2}\cdot x^2}$$

이 한 줄이 이 카드의 핵심을 이미 담고 있다. **정규분포의 지수 안에 들어가는 건 분산이 아니라 분산의 역수다.**

- $\sigma^2$이 크면 $1/\sigma^2$이 작다 → 지수가 천천히 줄어든다 → 넓고 퍼진 종 모양.
- $\sigma^2$이 작으면 $1/\sigma^2$이 크다 → 지수가 급히 줄어든다 → 좁고 뾰족한 종 모양.

---

## 2. 2차원으로 확장: 분산의 역수 → 행렬의 역행렬

화면 위의 Gaussian은 점 하나가 아니라 **타원 얼룩**이다. 좌표가 두 개($x$, $y$) 필요하니 위치를 벡터로 쓴다.

$$\mathbf{d} = \begin{bmatrix} dx \\ dy \end{bmatrix} = \begin{bmatrix} x - \mu_x \\ y - \mu_y \end{bmatrix}$$

($\mathbf{d}$는 Gaussian 중심 $\mu$에서 지금 칠하려는 픽셀까지의 변위 벡터. 코드의 `dx, dy`가 정확히 이것이다.)

가장 단순한 경우, $x$방향과 $y$방향이 서로 독립이면 두 정규분포를 그냥 곱하면 된다.

$$e^{-\frac{dx^2}{2\sigma_x^2}} \cdot e^{-\frac{dy^2}{2\sigma_y^2}} = \exp\!\left(-\frac{1}{2}\left(\frac{dx^2}{\sigma_x^2} + \frac{dy^2}{\sigma_y^2}\right)\right)$$

그런데 3D Gaussian이 비스듬히 기울어져 화면에 투영되면 타원도 **기울어진다**. 기울어진 타원은 $dx\,dy$ 교차항이 있어야 표현된다. 그래서 지수 안에 들어가는 일반적인 꼴은

$$-\frac{1}{2}\Big( a\,dx^2 + 2b\,dx\,dy + c\,dy^2 \Big)$$

가 되고, 이걸 행렬로 깔끔하게 묶으면

$$a\,dx^2 + 2b\,dx\,dy + c\,dy^2 = \begin{bmatrix} dx & dy\end{bmatrix}\begin{bmatrix} a & b \\ b & c \end{bmatrix}\begin{bmatrix} dx \\ dy \end{bmatrix} = \mathbf{d}^\top Q\, \mathbf{d}$$

이렇게 **벡터 × 행렬 × 벡터** 형태로 2차식을 쓴 것을 **이차형식(quadratic form)**이라고 한다. 직접 전개해서 확인해 보면 좋다: 첫 곱을 하면 $[\,a\,dx + b\,dy,\; b\,dx + c\,dy\,]$, 여기에 $\mathbf{d}$를 다시 곱하면 $a\,dx^2 + b\,dx\,dy + b\,dx\,dy + c\,dy^2$, 즉 위 식이다. 교차항 계수가 $2b$로 나오는 이유가 여기 있다 — $b$가 대칭 위치에 **두 번** 들어가기 때문이다.

이제 1차원과 짝을 맞춰 보자.

| | 1차원 | 2차원 |
|---|---|---|
| 퍼짐 정도 | 분산 $\sigma^2$ | 공분산 행렬 $\Sigma_2 = \begin{bmatrix} \sigma_{xx} & \sigma_{xy} \\ \sigma_{xy} & \sigma_{yy}\end{bmatrix}$ |
| 지수에 들어가는 것 | $\dfrac{1}{\sigma^2}$ (역수) | $\Sigma_2^{-1}$ (**역행렬**) |
| 지수 전체 | $-\frac{1}{2}\cdot\frac{1}{\sigma^2}x^2$ | $-\frac{1}{2}\,\mathbf{d}^\top \Sigma_2^{-1}\mathbf{d}$ |

즉 위에서 쓴 그 행렬 $Q = \begin{bmatrix} a & b \\ b & c\end{bmatrix}$의 정체가 바로 $\Sigma_2^{-1}$이다.

$$G(\mathbf{d}) = \exp\!\left(-\tfrac{1}{2}\,\mathbf{d}^\top \Sigma_2^{-1} \mathbf{d}\right)$$

**"1차원에서 분모에 $\sigma^2$이 있던 자리 = 2차원에서 역행렬이 있는 자리"** — 역행렬이 갑자기 튀어나온 게 아니라, 나눗셈의 2차원 판이 역행렬일 뿐이다.

---

## 3. 2×2 대칭 행렬의 역행렬 공식

역행렬은 대학 선형대수의 물건 같지만, $2\times2$짜리는 공식이 딱 하나라 외울 것도 별로 없다.

$$M = \begin{bmatrix} p & q \\ r & s \end{bmatrix} \quad\Longrightarrow\quad M^{-1} = \frac{1}{ps-qr}\begin{bmatrix} s & -q \\ -r & p \end{bmatrix}$$

(분모 $ps-qr$가 행렬식 $\det M$. 이게 0이면 역행렬이 없다.)

공분산 행렬은 **대칭**($q = r$)이므로 더 간단해진다. $\Sigma_2 = \begin{bmatrix} A & B \\ B & D\end{bmatrix}$라 두면

$$\Sigma_2^{-1} = \frac{1}{AD - B^2}\begin{bmatrix} D & -B \\ -B & A \end{bmatrix}$$

**대칭 행렬의 역행렬도 다시 대칭**이라는 점을 눈으로 확인하자. 오른쪽 위와 왼쪽 아래가 둘 다 $-B/(AD-B^2)$로 똑같다.

그래서

$$a = \frac{D}{AD-B^2},\qquad b = \frac{-B}{AD-B^2},\qquad c = \frac{A}{AD-B^2}$$

이 $(a,b,c)$가 바로 `conics`에 저장되는 세 숫자다. 코드로는 이렇게 생겼다.

```python
inv = torch.linalg.inv(cov2d)                                   # Σ₂⁻¹  (2×2)
conics = torch.stack([inv[:,0,0], inv[:,0,1], inv[:,1,1]], -1)  # (a, b, c)
```

행렬의 4칸 중 **[0,0], [0,1], [1,1]** 세 칸만 뽑는다. 남은 [1,0]은 [0,1]과 같으니 저장할 이유가 없다. 이렇게 **대각선 위쪽(과 대각선)만 뽑는 것**을 상삼각(upper-triangular) 성분이라 부른다. 4개가 아니라 3개 → 메모리 25% 절약이고, 수백만 개의 Gaussian × 카메라 수만큼 곱해지면 결코 작지 않은 차이다.

---

## 4. 왜 이름이 "conic"인가 — 기하 시간의 원뿔곡선

기하 시간에 배운 이차곡선(원뿔곡선, conic section)의 일반형을 떠올려 보자.

$$Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0$$

$B^2 - 4AC$의 부호에 따라 타원 / 포물선 / 쌍곡선이 갈렸다. 그 중 중심이 원점인 타원은

$$a x^2 + 2b xy + c y^2 = k \quad (k>0)$$

꼴이다. 그런데 이건 §2에서 만든 이차형식과 **글자 그대로 같은 식**이다.

$$\mathbf{d}^\top \Sigma_2^{-1}\mathbf{d} = k$$

즉 Gaussian의 **등고선**(밝기가 같은 점들의 자취)을 $G(\mathbf{d}) = \text{const}$로 놓으면, 지수 안이 상수라는 뜻이고, 그것이 바로 위 타원 방정식이다.

- $\Sigma_2^{-1}$이 양정치(모든 방향에서 $\mathbf{d}^\top\Sigma_2^{-1}\mathbf{d} > 0$)면 이 곡선은 항상 **타원**이다.
- 그 타원의 기울기와 납작한 정도를 결정하는 계수가 정확히 $(a, b, c)$다.

**"conic"이라는 이름은 여기서 왔다.** $\Sigma_2^{-1}$은 "2D 공분산의 역행렬"이라는 통계적 이름보다, "화면 위에 그려질 타원(원뿔곡선)의 계수 3개"라는 기하적 이름이 렌더링 코드에서는 더 직관적이라, 3D Gaussian Splatting 구현들은 이걸 관습적으로 `conic`이라 부른다.

한 가지 더: $b^2 - ac$의 부호를 보면 §4 맨 위의 판별식 $B^2-4AC$와 대응된다($B = 2b$이므로 $B^2-4AC = 4(b^2-ac)$). 공분산의 역행렬은 항상 양정치라 $ac - b^2 > 0$, 즉 판별식이 음수 → **언제나 타원**임이 보장된다.

---

## 5. 왜 미리 역행렬을 구해 저장하는가 — 계산 효율

이게 `conics`를 **투영 단계에서 만들어 `meta`에 담아 두는** 진짜 이유다.

래스터화 커널이 하는 일은 픽셀 하나마다 다음을 계산하는 것이다 (실제 gsplat 코드).

```python
dx, dy = means2d[g,0] - px, means2d[g,1] - py   # 중심에서 픽셀까지 변위
a, b, c = conics[g]                             # 미리 구해 둔 Σ₂⁻¹
sigma = 0.5 * (a*dx*dx + c*dy*dy) + b*dx*dy     # = ½ dᵀ Σ₂⁻¹ d
alpha = opacities[g] * torch.exp(-sigma)        # 이 픽셀에서의 알파
```

세 번째 줄이 §2의 $\frac{1}{2}\mathbf{d}^\top\Sigma_2^{-1}\mathbf{d}$를 전개한 것이다 (교차항이 $2b$이므로 $\frac{1}{2}\cdot 2b\,dx\,dy = b\,dx\,dy$가 되어, $b$ 항만 $\frac12$이 붙지 않는다).

이 줄에는 **곱셈 5번, 덧셈 2번**뿐이다. 나눗셈도 역행렬 계산도 없다.

만약 $\Sigma_2$만 저장했다면, 픽셀마다 $\det = AD - B^2$를 구하고 그걸로 세 번 나누는 일을 반복해야 한다. 나눗셈은 GPU에서 곱셈보다 훨씬 비싼 연산이고, 무엇보다 **완전히 낭비**다. 같은 Gaussian이 덮는 픽셀 수백 개가 전부 똑같은 $\Sigma_2^{-1}$을 다시 구하게 되기 때문이다.

규모를 감으로 잡아 보면:

| | 횟수 |
|---|---|
| 역행렬 계산 (투영 단계, Gaussian × 카메라당 1번) | 약 $N \times C$ |
| $\sigma$ 계산 (래스터화 단계, Gaussian이 덮는 픽셀마다) | 그보다 **수백~수천 배** |

즉 "**한 번 비싸게 계산해서, 수백 번 싸게 재사용한다**"는 전형적인 사전계산(precompute) 최적화다. 상수 시간의 준비 작업을 앞으로 빼서 안쪽 루프를 가볍게 만드는 것 — 알고리즘에서 늘 하는 그 일이다.

---

## 6. 한 장 요약

$$\underbrace{\Sigma_2}_{\text{퍼짐 정도}} \xrightarrow{\ \text{역행렬}\ } \underbrace{\Sigma_2^{-1} = \begin{bmatrix} a & b \\ b & c\end{bmatrix}}_{\text{conic}} \xrightarrow{\ \text{대칭이라 3개만}\ } \underbrace{(a,b,c)}_{\texttt{conics[C,N,3]}}$$

- **무엇**: $\Sigma_2^{-1}$의 상삼각 성분 3개, 모양은 `[C, N, 3]` (카메라 수 × Gaussian 수 × 3).
- **왜 역행렬**: 1차원 $e^{-x^2/(2\sigma^2)}$에서 분산이 분모에 있듯, 2차원 Gaussian의 지수에도 $\Sigma_2$가 아니라 $\Sigma_2^{-1}$이 들어간다.
- **왜 3개**: 공분산도 그 역행렬도 대칭이라 $b$가 두 번 중복된다.
- **왜 conic**: $\mathbf{d}^\top\Sigma_2^{-1}\mathbf{d} = k$가 곧 타원 방정식 $a x^2 + 2bxy + cy^2 = k$이기 때문.
- **왜 미리**: 픽셀마다 나눗셈으로 역행렬을 다시 구하지 않기 위해. 래스터화 안쪽 루프는 곱셈 몇 번으로 끝난다.

그래서 카드의 답 마지막 문장 — "래스터화에서 실제로 쓰이는 것은 $\Sigma_2$가 아니라 이 conic이다" — 는 단순한 사실 서술이 아니라, **$\Sigma_2$는 conic을 만들기 위한 중간 재료일 뿐이며 커널에는 전달조차 되지 않는다**는 설계상의 선언이다.
