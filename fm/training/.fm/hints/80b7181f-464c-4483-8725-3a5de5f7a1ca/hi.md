# 3D Gaussian의 공분산 행렬은 어떻게 합성되며, 왜 항상 양의 정부호인가?

**한 줄 답**: quaternion 회전 $R$과 스케일 대각행렬 $S = \mathrm{diag}(s_1,s_2,s_3)$로부터
$\Sigma = R\,S\,S^\top R^\top$로 합성한다. $M = RS$로 묶으면 $\Sigma = MM^\top$ 꼴이 되고,
이 꼴은 $x^\top \Sigma x = \lVert M^\top x\rVert^2 \ge 0$ 이므로 **파라미터 값이 무엇이든** 양의 준정부호이며,
$s_k > 0$ 이기만 하면(gsplat은 `scales`를 log 공간에 저장하므로 자동) 순 양의 정부호다.

아래에서는 이 한 줄이 왜 필요하고 각 기호가 무엇인지를, 고교 수학
(정규분포, 벡터의 내적, 이차곡선·이차곡면, 회전, 행렬의 곱)에서 출발해 단계적으로 쌓아 올린다.

---

## 1. 출발점: 1변수 정규분포를 3차원으로 늘리기

고교 확률과 통계에서 배운 정규분포 $N(\mu, \sigma^2)$의 확률밀도는

$$
f(x) \propto \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

였다. 3DGS의 한 Gaussian은 이걸 3차원 공간으로 늘린 것이다. 우선 **축에 정렬된** 경우,
즉 $x,y,z$ 세 방향이 서로 독립이고 각각 표준편차 $s_1, s_2, s_3$을 갖는 경우를 보자.
독립이면 확률밀도는 곱이므로

$$
f(\Delta) \propto
\exp\!\left(-\frac{\Delta_1^2}{2s_1^2}\right)
\exp\!\left(-\frac{\Delta_2^2}{2s_2^2}\right)
\exp\!\left(-\frac{\Delta_3^2}{2s_3^2}\right)
= \exp\!\left(-\frac{1}{2}\left[\frac{\Delta_1^2}{s_1^2}+\frac{\Delta_2^2}{s_2^2}+\frac{\Delta_3^2}{s_3^2}\right]\right)
$$

($\Delta = \mathbf{x} - \boldsymbol\mu$, 중심에서의 변위)

여기서 대괄호 안이 낯익어야 한다. 고교 기하의 **타원(체) 방정식**

$$
\frac{\Delta_1^2}{a^2}+\frac{\Delta_2^2}{b^2}+\frac{\Delta_3^2}{c^2} = 1
$$

과 똑같은 모양이다. 즉 **확률밀도가 일정한 등고면(level set)은 반축 길이가 $s_1,s_2,s_3$의 상수배인
타원체**다. 3DGS에서 "Gaussian 하나 = 흐릿한 타원체 물감 방울"이라는 그림이 여기서 나온다.

대괄호 안을 행렬로 묶어 보자. $\Sigma_0 = \mathrm{diag}(s_1^2, s_2^2, s_3^2)$ 로 두면
$\Sigma_0^{-1} = \mathrm{diag}(1/s_1^2, 1/s_2^2, 1/s_3^2)$ 이고, 행렬곱을 직접 전개해 보면

$$
\Delta^\top \Sigma_0^{-1}\Delta = \frac{\Delta_1^2}{s_1^2}+\frac{\Delta_2^2}{s_2^2}+\frac{\Delta_3^2}{s_3^2}
$$

이므로

$$
f(\Delta) \propto \exp\!\left(-\tfrac{1}{2}\,\Delta^\top \Sigma_0^{-1} \Delta\right)
$$

이 **이차형식(quadratic form)** 표기가 3DGS 문서 어디서나 등장하는 그 식이다.
워크스루 3단계의 알파 계산식

$$
\alpha_i = o_i \exp\!\left(-\tfrac12\,\Delta^\top \Sigma'^{-1}\Delta\right)
$$

도 정확히 이 형태다 (다만 $\Sigma'$는 화면 위 2×2 버전 — 5절에서 다룬다).

> 이 시점에서 이미 **결정적인 요구사항**이 하나 생긴다. 식에 들어가는 것은 $\Sigma$가 아니라
> $\Sigma^{-1}$ 이다. 역행렬이 존재해야 하고, 게다가 지수의 부호가 뒤집히지 않아야 한다
> (뒤집히면 중심에서 멀어질수록 밝아지는, 무한히 발산하는 괴물이 된다).
> 이 두 조건을 한꺼번에 요구하는 것이 바로 "양의 정부호"다.

## 2. 기울어진 타원체 = 회전시킨 축정렬 타원체

축에 정렬된 타원체만 쓸 수는 없다. 실제 장면에는 비스듬한 벽면, 기울어진 나뭇가지가 있다.
자유도를 세어 보면 축정렬은 $s_1,s_2,s_3$ 세 개뿐이라, 이 표현으로 만들 수 있는 3DGS는
축을 향해서만 늘어난 물감 방울들의 집합이다. 3DGS가 "**비등방(anisotropic)** Gaussian"을
자랑하는 지점이 바로 이 자세(orientation) 자유도다.

방법은 간단하다. **축정렬 타원체를 만들고 나서 회전시킨다.** $R$을 3×3 회전행렬이라 하고,
회전된 좌표 $\Delta$ 를 "고개를 돌려서" 타원체의 자기 축 좌표계 $\tilde\Delta$ 로 되돌리면
$\tilde\Delta = R^\top \Delta$ 다 (회전행렬의 역행렬은 전치, $R^{-1} = R^\top$).
그 좌표계에서는 1절의 축정렬 식이 그대로 성립하므로

$$
\Delta^\top \Sigma^{-1}\Delta \;=\; \tilde\Delta^\top \Sigma_0^{-1}\tilde\Delta
\;=\; (R^\top\Delta)^\top \Sigma_0^{-1}(R^\top \Delta)
\;=\; \Delta^\top \underbrace{R\,\Sigma_0^{-1}R^\top}_{=\;\Sigma^{-1}}\Delta
$$

모든 $\Delta$에 대해 성립해야 하니 $\Sigma^{-1} = R\Sigma_0^{-1}R^\top$, 양변의 역을 취하면

$$
\Sigma = R\,\Sigma_0\,R^\top = R\,\mathrm{diag}(s_1^2,s_2^2,s_3^2)\,R^\top
$$

마지막으로 $S = \mathrm{diag}(s_1,s_2,s_3)$ (제곱 안 한 것!) 로 두면 대각행렬이라
$SS^\top = S^2 = \mathrm{diag}(s_1^2,s_2^2,s_3^2)$ 이므로

$$
\boxed{\;\Sigma = R\,S\,S^\top R^\top\;}
$$

이것이 카드의 답이다. **$S$를 제곱 안 한 채 두 번 쓰는 이유**는 여기에 있다 —
`scales`는 "표준편차"(길이 단위)이고 공분산은 "분산"(길이²) 단위라서, 어딘가에서 한 번은
제곱해야 한다. 그 제곱을 $S \cdot S^\top$ 로 쪼개 놓은 것이 다음 절의 마법을 부른다.

### 자유도 확인: 손실 없이 전부 표현되는가

$\Sigma$는 3×3 대칭행렬이므로 독립 성분이 대각 3개 + 비대각 3개 = **6개**다.
반면 이 파라미터화는 $R$ (3차원 회전이므로 자유도 3) + $S$ (3) = **6**.
정확히 일치한다. 즉 이 형태는 표현력을 깎아서 안전성을 사는 것이 아니라,
**양의 정부호 대칭행렬 전체를 정확히 한 번씩 훑는다** (선형대수의 스펙트럼 정리가 보장하는 사실이지만,
고교 기하 언어로는 "어떤 타원체든 축 3개의 길이와 그 자세만 정하면 결정된다"로 읽으면 된다).

## 3. 핵심: 왜 항상 양의 정부호인가

"양의 정부호(positive definite)"의 정의는 이렇다.

> 모든 $x \ne 0$ 에 대해 $x^\top \Sigma x > 0$.

증명은 놀랄 만큼 짧다. 회전과 스케일을 **하나의 행렬로 묶는다.**

$$
M = R\,S \qquad\Longrightarrow\qquad \Sigma = R S S^\top R^\top = (RS)(RS)^\top = M M^\top
$$

(전치의 성질 $(AB)^\top = B^\top A^\top$ 을 썼다.) 이제 임의의 $x$에 대해

$$
x^\top \Sigma x = x^\top M M^\top x = (M^\top x)^\top (M^\top x) = \lVert M^\top x\rVert^2 \;\ge\; 0
$$

**고교 벡터에서 배운 "어떤 벡터든 자기 자신과의 내적은 길이의 제곱이라 음수가 될 수 없다"** —
이 한 줄이 전부다. $\Sigma$가 무엇으로 만들어졌는지는 전혀 보지 않았다.
$M$이 어떤 3×3 행렬이든 $MM^\top$은 음수 값을 낼 수 없다. 이런 형태를 **Gram 행렬**이라 부른다.

등호는 언제인가? $\lVert M^\top x\rVert^2 = 0 \iff M^\top x = \mathbf{0}$ 이다.
$M$이 가역이면(즉 $\det M \ne 0$) $x = \mathbf{0}$ 뿐이므로, $x \ne 0$ 에서는 언제나 **순부등호**가 된다.

$$
\det M = \det(RS) = \det R \cdot \det S = 1 \cdot s_1 s_2 s_3
$$

(회전행렬의 행렬식은 $+1$, 대각행렬의 행렬식은 성분의 곱)

즉 **$s_1 s_2 s_3 \neq 0$ 이면 양의 정부호**다. 그리고 gsplat은 `scales`를 log 공간에 저장하고
렌더 직전 `torch.exp`를 통과시키므로 ($s_k = e^{\tilde s_k}$), 저장값이 $-\infty$ 로 가더라도
$s_k$는 정의상 절대 0이나 음수가 되지 않는다. **두 장치(Gram 형태 + log 저장)가 맞물려
"어떤 학습 스텝을 밟아도 공분산이 깨지지 않는다"를 구조적으로 보장한다.**

고유값으로 다시 확인해 보자. $R$의 열벡터를 $\mathbf{r}_1,\mathbf{r}_2,\mathbf{r}_3$ 이라 하면
회전행렬이므로 이들은 정규직교($\mathbf{r}_i\cdot\mathbf{r}_j = \delta_{ij}$)다. 계산하면

$$
\Sigma \mathbf{r}_k = R S^2 R^\top \mathbf{r}_k = R S^2 \mathbf{e}_k = R (s_k^2 \mathbf{e}_k) = s_k^2\,\mathbf{r}_k
$$

$$
\therefore\quad \mathrm{eig}(\Sigma) = \{s_1^2,\, s_2^2,\, s_3^2\},\qquad
\text{고유벡터} = R\text{의 열} = \text{타원체의 세 축 방향}
$$

고유값이 전부 **실수의 제곱**이라 음수가 될 방법이 없다. 기하학적으로도 자명하다 —
$s_k$는 타원체의 반축 길이이고, 길이는 음수일 수 없다.

### 대조: 6원소를 직접 파라미터로 두면 어떻게 되는가

만약 $\Sigma$의 대칭 6원소 $(\Sigma_{11},\Sigma_{12},\Sigma_{13},\Sigma_{22},\Sigma_{23},\Sigma_{33})$을
그대로 학습 파라미터로 뒀다면, "대칭"은 유지되지만 "양의 정부호"는 **6차원 공간 속의 곡면으로 둘러싸인
좁은 영역(원뿔 모양)** 이고, Adam은 그 경계를 존중해 주지 않는다. 한 스텝이 경계를 넘으면
$\Sigma^{-1}$이 부정부호가 되어 $\exp(+\text{큰 값})$ 이 되고, 그 Gaussian은 화면 전체를 태워 버린다.

`expy.py`의 실험 결과: 랜덤 대칭 6원소는 **99.6%** 가 최소고유값 음수(즉 사용 불가)였고,
$RSS^\top R^\top$ 재매개화는 **0%** 였다. 게다가 목표 $\Sigma$에 경사하강으로 맞추는 실험에서
6원소 직접 파라미터화는 **1000스텝 중 869스텝**을 부정부호 상태로 지나갔고,
재매개화는 **0스텝**이었다.

## 4. 실제 코드에서: `quat_scale_to_covar_preci`

gsplat의 CUDA 커널 (`gsplat/cuda/include/Utils.cuh:285`)이 하는 일이 위 유도와 글자 그대로 같다.

```cpp
mat3 R = quat_to_rotmat(quat);
// C = R * S * S * Rt
mat3 S = mat3(scale[0], 0, 0,  0, scale[1], 0,  0, 0, scale[2]);
mat3 M = R * S;
*covar = M * glm::transpose(M);
```

PyTorch 참조 구현(`gsplat/cuda/_math.py:700`)도 동일하다.
대각행렬 곱은 열 단위 브로드캐스트로 처리해 실제 행렬곱을 아낀다.

```python
R = _quat_to_rotmat(quats)                       # [..., 3, 3]
M = R * scales[..., None, :]                     # R @ diag(s) — 열마다 s_k를 곱한 것
covars = torch.einsum("...ij,...kj -> ...ik", M, M)   # M @ M^T
```

### 곁가지 1: 역행렬을 아예 계산하지 않는다

같은 함수는 **precision matrix** $P = \Sigma^{-1}$ 도 돌려주는데, 역행렬 루틴을 부르지 않는다.

$$
\Sigma^{-1} = (R S S^\top R^\top)^{-1}
= (R^\top)^{-1}(SS^\top)^{-1}R^{-1}
= R\,S^{-1}S^{-\top}R^\top
$$

($R^{-1}=R^\top$ 이므로 $(R^\top)^{-1} = R$)

대각행렬의 역행렬은 성분의 역수뿐이므로, 코드는 그냥 `1.0f / scale[k]` 로 $S^{-1}$을 만들어
같은 Gram 곱을 한 번 더 한다.

```cpp
// P = R * S^-1 * S^-1 * Rt
mat3 S = mat3(1.0f/scale[0], 0,0, 0,1.0f/scale[1],0, 0,0,1.0f/scale[2]);
mat3 M = R * S;
*preci = M * glm::transpose(M);
```

$\Sigma^{-1}$ 역시 Gram 형태이므로 **역행렬도 자동으로 양의 정부호**다.
수치적으로도 안전하다 — 일반 3×3 역행렬은 $\det$ 이 0에 가까우면 폭발하지만,
여기서는 나눗셈이 $1/s_k$ 뿐이고 $s_k = e^{\tilde s_k} > 0$ 이 보장된다.
(그럼에도 커널에 `assert(scale[k] != 0)` 가 있고, 투영 단계가 $s < \varepsilon$ 인 퇴화 Gaussian을
컬링해 이 경로에 도달하지 못하게 한다.)

### 곁가지 2: $M$은 "$\Sigma$의 제곱근"이라서 샘플링에 그대로 쓰인다

밀도화의 **split** 연산은 큰 Gaussian 하나를 둘로 쪼개면서 자식의 위치를 부모 타원체 안에서
랜덤 샘플링한다 (`gsplat/strategy/ops.py:196`).

```python
scales  = torch.exp(params["scales"][sel])
quats   = F.normalize(params["quats"][sel], dim=-1)
rotmats = normalized_quat_to_rotmat(quats)          # [N,3,3]
samples = torch.einsum("nij,nj,bnj->bni", rotmats, scales, torch.randn(2, len(scales), 3))
```

이 einsum이 계산하는 것은 정확히 $R\,S\,\varepsilon = M\varepsilon$ ($\varepsilon \sim N(0, I)$)이다.
왜 이게 옳은가? 공분산의 변환 규칙을 쓰면

$$
\mathrm{Cov}(M\varepsilon) = M\,\mathrm{Cov}(\varepsilon)\,M^\top = M I M^\top = MM^\top = \Sigma
$$

즉 **$\Sigma$를 만드는 데 쓴 바로 그 $M$이, $\Sigma$를 갖는 표본을 뽑는 도구이기도 하다.**
1변수에서 $Z \sim N(0,1)$ 일 때 $\sigma Z \sim N(0,\sigma^2)$ 였던 것의 3차원 판이고,
$M$이 $\sigma$의 자리를 차지한다. $\Sigma$의 6원소만 갖고 있었다면 이 샘플링을 하기 위해
Cholesky 분해나 고유값 분해를 매번 돌려야 했을 것이다 — 재매개화는 그 분해를
**미리, 파라미터 형태로** 들고 있는 셈이다.

MCMC 전략의 노이즈 주입도 같은 함수를 호출해 $\Sigma$로 노이즈를 성형한다 (`ops.py:497`) —
Gaussian이 납작한 방향으로는 조금, 긴 방향으로는 많이 흔들리게 만들기 위함이다.

### 곁가지 3: 부동소수점 비대칭 지우기

`triu=True` 로 6원소만 뽑을 때 코드는 그냥 상삼각을 잘라내지 않고 전치와 평균을 낸다
(`_math.py:707`). float 곱셈 순서 때문에 $\Sigma_{12}$와 $\Sigma_{21}$이 마지막 비트에서
어긋날 수 있는데, $\tfrac12(\Sigma + \Sigma^\top)$ 는 수학적으로 $\Sigma$와 같으면서
대칭성을 비트 단위로 강제한다.

## 5. 화면으로 투영해도 양정부호가 살아남는가 (그리고 안 살아남는 곳)

렌더링에 실제로 쓰이는 것은 3D $\Sigma$가 아니라 화면 위 2×2 공분산 $\Sigma'$ 다.
EWA splatting의 유도는 world→camera 회전 $W$와 원근투영의 야코비안 $J$ (2×3 행렬,
$\partial(\text{화면좌표})/\partial(\text{카메라좌표})$)를 써서

$$
\Sigma' = J\,W\,\Sigma\,W^\top J^\top
$$

이고 코드도 그대로다 (`gsplat/cuda/_torch_impl.py:100`).

```python
cov2d = torch.einsum("...ij,...jk,...kl->...il", J, covars, J.transpose(-1, -2))
```

$A = JWM$ 으로 묶으면 $\Sigma' = AA^\top$ 이라 **또 Gram 형태**다. 양의 준정부호는 공짜로 따라온다.

하지만 여기서 **순부등호는 깨질 수 있다.** $A$는 2×3 행렬이므로 rank가 최대 2인데,
3D Gaussian이 시선 방향으로 극단적으로 납작하면(면을 표현하는 얇은 원반이 카메라를 향해 모로 선 경우)
$A$의 rank가 1로 떨어져 $\det \Sigma' = 0$ 이 된다. 그러면 $\Sigma'^{-1}$이 존재하지 않는다 —
"두 배 얇으면 두 배 밝은 선"이 되다가 폭이 픽셀 하나보다 작아지면 그냥 사라져 버린다(에일리어싱).

gsplat의 처방은 **대각선에 상수를 더하는 것**(dilation)이다 (`_torch_impl.py:309`).

```python
covars2d = covars2d + torch.eye(2) * eps2d      # eps2d = 0.3
```

$\Sigma' + \varepsilon I$ 의 고유값은 원래 고유값에 $\varepsilon$을 더한 값이므로
**최소고유값 $\ge 0.3$ 이 강제되어 순 양의 정부호가 된다.** 물리적으로는 "화면에서 최소
$\sqrt{0.3}\approx 0.55$ 픽셀 폭의 저역통과 필터를 씌운다"는 뜻이다.
그 덕분에 바로 다음 줄의 `conics`( = $\Sigma'^{-1}$ 의 3원소)를 $\det$ 으로 나눠 안전하게 얻는다.

당연히 대가가 있다 — 없던 흐림을 더했으니 총 밝기가 퍼져 어두워진다.
그래서 흐림 전후의 행렬식 비로 보정계수를 만든다.

$$
\text{compensation} = \sqrt{\frac{\det \Sigma'_{\text{orig}}}{\det(\Sigma' + \varepsilon I)}}
$$

(2D Gaussian의 적분값이 $\sqrt{\det\Sigma'}$ 에 비례하기 때문. 이것이 "antialiased" 모드의 핵심)

덧붙여, 타일 반경도 이 공분산의 대각에서 나온다 (`_torch_impl.py:337`).

```python
radius_x = torch.ceil(3.33 * torch.sqrt(covars2d[..., 0, 0]))
```

$\sqrt{\Sigma'_{11}}$ 이 $x$ 방향 표준편차이므로, 이건 고교 확률의 **$3\sigma$ 규칙**
("정규분포 값의 99.7%는 $\pm3\sigma$ 안") 을 조금 넉넉하게 쓴 컬링이다.

## 6. 왜 회전을 quaternion으로 저장하는가

$\Sigma = RSS^\top R^\top$ 에서 $S$ 쪽은 log 저장으로 해결됐다. $R$ 쪽은 어떤가?
회전을 파라미터로 두는 방법은 여러 가지인데, 학습 가능성 관점에서 비교하면 이렇다.

| 표현 | 저장 개수 | 제약 | 학습 시 문제 |
|---|---|---|---|
| 회전행렬 9원소 | 9 | $R^\top R = I$ (등식 6개) | Adam 한 스텝이면 직교성이 깨진다. 매 스텝 재직교화(QR/SVD) 필요 — 비싸다 |
| 오일러각 | 3 | 없음 | **gimbal lock** — 특정 자세에서 두 축이 겹쳐 자유도가 사라지고 gradient가 죽는다. 각도의 주기성 때문에 $\pm\pi$ 경계에서 불연속 |
| 축-각 (회전벡터) | 3 | 없음 | 노름이 $2\pi$ 근처에서 특이. $\theta \to 0$ 에서 야코비안이 불안정 |
| **단위 quaternion** | 4 | $\lVert q\rVert = 1$ (등식 **1개**) | 제약이 하나뿐이고, 그 하나가 **없어도 되는** 제약이다 (아래) |

quaternion $q = (w,x,y,z)$ 에서 회전행렬을 만드는 공식이 `gsplat/utils.py:123`
`normalized_quat_to_rotmat` 이다 (gsplat은 `wxyz` 순서 규약).

$$
R(q) = \begin{pmatrix}
1-2(y^2+z^2) & 2(xy - wz) & 2(xz + wy)\\
2(xy + wz) & 1-2(x^2+z^2) & 2(yz - wx)\\
2(xz - wy) & 2(yz + wx) & 1-2(x^2+y^2)
\end{pmatrix}
$$

성분이 전부 **2차 다항식**이라는 점이 중요하다. 삼각함수가 없으니 미분이 다항식이고,
특이점도 없고, GPU에서 곱셈 몇 번으로 끝난다.

### 노름 제약은 "없어도 되는 제약"이다

$\lVert q \rVert = 1$ 은 등식 제약이라 log/logit 같은 단조 사상으로는 제거할 수 없다.
gsplat의 처방은 **제약 위반을 그냥 허용하고, 정규화를 forward에 흡수시키는 것**이다.

- 저장: `torch.rand((N, 4))` — 정규화되지 않은 자유 4-벡터. 학습 중에도 노름을 강제하지 않는다
- 사용: 렌더 직전 `F.normalize(quats, dim=-1)` (`gsplat/rendering.py:1283`), CUDA 커널도
  `quats: Quaternions (No need to be normalized)` 라고 명시한다 (`_wrapper.py:667`)

이게 성립하는 이유는 **노름이 gauge 자유도**이기 때문이다. $q$와 $\lambda q$ ($\lambda>0$)를
정규화하면 같은 단위 quaternion이 되므로 **같은 회전**을 준다. 따라서 노름 방향의 gradient는
손실에 영향을 주지 않고(정규화 층이 걸러낸다), 옵티마이저는 자연히 구면에 접하는 성분만 쓴다.
Riemannian 최적화나 매 스텝 재정규화 같은 비싼 장치가 필요 없다.

> **주의할 함정**: 정규화를 건너뛰고 미정규화 $q$를 위 공식에 직접 넣으면 나오는 $R$은 **직교가 아니다**
> ($R^\top R = \lVert q\rVert^4 \cdot(\dots) \ne I$). 그래도 $\Sigma = R S S^\top R^\top$ 은
> Gram 형태니까 **양의 정부호는 그대로 유지된다** — 3절의 증명이 $R$의 직교성을 전혀 쓰지 않았다는
> 사실을 상기하라. 깨지는 것은 정부호성이 아니라 **기하**다: 고유값이 더 이상 $s_k^2$ 이 아니게 되어
> Gaussian의 실제 크기가 $\lVert q\rVert$ 에 따라 멋대로 커지거나 작아진다.
> `expy.py`에서 $\lVert q \rVert = 2$ 일 때 축 길이가 약 $4$배로 부풀었다.
> 즉 **양정부호는 Gram 형태가, 올바른 크기는 정규화가** 각각 담보한다.

## 7. 요약

1. 축정렬 Gaussian은 $\Sigma_0 = \mathrm{diag}(s_k^2)$, 등고면은 반축 $s_k$의 타원체.
   기울어진 것을 표현하려면 회전 $R$을 씌워 $\Sigma = R\,\mathrm{diag}(s_k^2)\,R^\top$.
   $S = \mathrm{diag}(s_k)$ 로 제곱을 쪼개면 $\Sigma = R\,S\,S^\top R^\top$. 자유도 3+3 = 대칭행렬의 6과 정확히 일치.
2. $M = RS$ 로 묶으면 $\Sigma = MM^\top$ (**Gram 형태**). 그러면
   $x^\top\Sigma x = \lVert M^\top x\rVert^2 \ge 0$ 이 파라미터 값과 무관하게 성립 — 양의 준정부호는 **구조적**이다.
3. 순부등호(순 양정부호)는 $\det M = s_1s_2s_3 \ne 0$ 에서 오고, gsplat은 `scales`를 log 공간에
   저장해 $s_k = e^{\tilde s_k} > 0$ 을 보장한다. 고유값은 정확히 $\{s_k^2\}$, 고유벡터는 $R$의 열(타원체 축).
4. 얻는 것: (a) $\Sigma^{-1}$ 이 항상 존재하고 그것도 $R S^{-1}S^{-1}R^\top$ **닫힌 형식**으로
   (역행렬 루틴 불필요, `Utils.cuh:285`), (b) $M$이 곧 $\Sigma$의 제곱근이라 split/MCMC의
   샘플링에 그대로 재사용 (`ops.py:196`), (c) 투영 $\Sigma' = (JWM)(JWM)^\top$ 도 Gram이라 PSD 상속.
5. 안 되는 곳: 투영은 2×3 야코비안이라 rank가 떨어지면 준정부호가 되어 $\Sigma'^{-1}$ 이 없다 →
   `+ 0.3·I` dilation으로 최소고유값을 강제하고, $\sqrt{\det_{orig}/\det}$ 로 밝기를 보정한다.
6. 회전을 quaternion으로 두는 이유는 제약이 $\lVert q\rVert=1$ 하나뿐이고, 그 하나가
   **gauge 자유도**여서 정규화를 forward에 흡수하면 사라지기 때문. 오일러각의 gimbal lock,
   회전행렬 9원소의 직교 제약 유지 비용을 모두 피한다.
