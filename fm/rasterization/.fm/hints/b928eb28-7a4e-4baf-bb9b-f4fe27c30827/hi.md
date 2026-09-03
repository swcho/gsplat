# 카메라 위치를 viewmat에서 되찾기: $\mathbf{c} = -R^\top \mathbf{t}$

## 0. 왜 카메라 위치가 필요한가

가우시안 스플래팅에서 색은 **보는 방향에 따라 달라진다**(구면조화, SH). 각 가우시안의 중심 $\boldsymbol{\mu}$를 어느 방향에서 보는지 알려면 시선 벡터

$$\mathbf{d} = \boldsymbol{\mu} - \mathbf{c}$$

가 필요하고, 여기서 $\mathbf{c}$가 **세계 좌표계에서의 카메라 위치**다. 그런데 코드가 갖고 있는 것은 카메라 위치가 아니라 `viewmats[C,4,4]`뿐이다. 그래서 거기서 $\mathbf{c}$를 "복원"해야 한다.

---

## 1. viewmat이 하는 일: 세계 좌표 → 카메라 좌표

고등학교 기하에서 배운 **평행이동**과 **회전**을 3차원에서 이어붙인 것이 좌표 변환이다. `viewmat`은 world→camera 변환이므로, 세계 좌표 $\mathbf{p}_w$인 점을 카메라가 보는 좌표 $\mathbf{p}_c$로 바꾼다:

$$\mathbf{p}_c = R\,\mathbf{p}_w + \mathbf{t}$$

- $R$은 $3\times 3$ **회전행렬** — `viewmat[:3,:3]`
- $\mathbf{t}$는 $3$차원 **이동 벡터** — `viewmat[:3,3]` (4열의 위쪽 3개)

`viewmat`이 $4\times4$인 이유는 이 "회전 + 이동"을 행렬 곱 **하나로** 쓰기 위해서다(동차좌표):

$$
\begin{pmatrix}\mathbf{p}_c\\ 1\end{pmatrix}
=
\begin{pmatrix} R & \mathbf{t}\\ \mathbf{0}^\top & 1\end{pmatrix}
\begin{pmatrix}\mathbf{p}_w\\ 1\end{pmatrix}
$$

실제로 곱해보면 위쪽 3줄이 정확히 $R\mathbf{p}_w + \mathbf{t}$가 된다. 코드의 주석도 그대로다:

```
viewmats  [C,4,4] world→camera 변환 (상단 3x3 = R, 4열 = t)
```

> 주의: $\mathbf{t}$는 **카메라 위치가 아니다.** $\mathbf{t}$는 "세계 원점이 카메라 좌표계에서 어디에 찍히는가"다($\mathbf{p}_w=\mathbf{0}$을 넣어보면 $\mathbf{p}_c=\mathbf{t}$). 이 둘을 혼동하는 것이 가장 흔한 실수다.

---

## 2. 핵심 아이디어: 카메라는 자기 좌표계의 원점이다

여기가 이 문제 전체의 급소다. 아주 당연한 사실 하나에서 출발한다.

> **카메라 좌표계는 카메라 자신을 원점으로 삼는다.**

즉 카메라 위치 $\mathbf{c}$(세계 좌표)를 카메라 좌표로 변환하면 반드시 $\mathbf{0}$이 나온다. 이걸 위 식에 그대로 대입하면 **미지수 $\mathbf{c}$에 대한 방정식**이 생긴다:

$$\mathbf{0} = R\,\mathbf{c} + \mathbf{t}$$

이제 일차방정식 $0 = ax + b$를 $x = -b/a$로 푸는 것과 똑같이 풀면 된다. 다만 "$a$로 나누기"가 행렬에서는 "**역행렬을 왼쪽에 곱하기**"다:

$$
R\,\mathbf{c} = -\mathbf{t}
\quad\Longrightarrow\quad
R^{-1}R\,\mathbf{c} = -R^{-1}\mathbf{t}
\quad\Longrightarrow\quad
\boxed{\ \mathbf{c} = -R^{-1}\mathbf{t}\ }
$$

($R^{-1}R = I$이고 $I\mathbf{c} = \mathbf{c}$이므로.) 여기까지가 절반이다. 남은 것은 $R^{-1}$을 실제로 어떻게 계산하느냐다.

---

## 3. 회전행렬의 마법: $R^{-1} = R^\top$

일반적인 $3\times3$ 행렬의 역행렬은 계산이 꽤 번거롭다(여인수, 행렬식 …). 그런데 **회전행렬**은 역행렬이 공짜다. 그냥 **전치**(행과 열을 바꾸기, $R^\top_{ij} = R_{ji}$)하면 끝이다.

### 왜 그런가

회전행렬의 열벡터들을 $\mathbf{r}_1,\mathbf{r}_2,\mathbf{r}_3$이라 하자. 회전은 길이와 각도를 보존하므로, 이 열벡터들은

- 각각 **단위벡터**: $\mathbf{r}_i\cdot\mathbf{r}_i = 1$
- 서로 **수직**: $i\neq j$일 때 $\mathbf{r}_i\cdot\mathbf{r}_j = 0$

이다(고교 기하의 정규직교 기저). 그런데 $R^\top R$의 $(i,j)$ 성분은 정확히 "$R^\top$의 $i$행" $\cdot$ "$R$의 $j$열" $= \mathbf{r}_i\cdot\mathbf{r}_j$다. 따라서

$$R^\top R =
\begin{pmatrix}
\mathbf{r}_1\!\cdot\!\mathbf{r}_1 & \mathbf{r}_1\!\cdot\!\mathbf{r}_2 & \mathbf{r}_1\!\cdot\!\mathbf{r}_3\\
\mathbf{r}_2\!\cdot\!\mathbf{r}_1 & \mathbf{r}_2\!\cdot\!\mathbf{r}_2 & \mathbf{r}_2\!\cdot\!\mathbf{r}_3\\
\mathbf{r}_3\!\cdot\!\mathbf{r}_1 & \mathbf{r}_3\!\cdot\!\mathbf{r}_2 & \mathbf{r}_3\!\cdot\!\mathbf{r}_3
\end{pmatrix}
=
\begin{pmatrix}1&0&0\\0&1&0\\0&0&1\end{pmatrix} = I$$

즉 $R^\top$이 바로 $R^{-1}$이다.

### 2×2 회전으로 직접 확인

고교에서 익숙한 2차원 회전행렬로 확인해 보자.

$$R = \begin{pmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{pmatrix},
\qquad
R^\top = \begin{pmatrix}\cos\theta & \sin\theta\\ -\sin\theta & \cos\theta\end{pmatrix}$$

$$R^\top R = \begin{pmatrix}\cos^2\theta+\sin^2\theta & -\cos\theta\sin\theta+\sin\theta\cos\theta\\ -\sin\theta\cos\theta+\cos\theta\sin\theta & \sin^2\theta+\cos^2\theta\end{pmatrix} = \begin{pmatrix}1&0\\0&1\end{pmatrix}$$

$\sin^2\theta+\cos^2\theta=1$ 하나로 끝난다. 게다가 $R^\top$은 $\theta$ 자리에 $-\theta$를 넣은 행렬($\cos(-\theta)=\cos\theta$, $\sin(-\theta)=-\sin\theta$)이므로, "**전치 = 반대 방향 회전 = 역회전**"이라는 기하적 의미까지 정확히 맞는다.

### 결론

$$\boxed{\ \mathbf{c} = -R^{-1}\mathbf{t} = -R^\top\mathbf{t}\ }$$

이것이 답의 $\text{cam\_pos} = -R^\top t$다. 역행렬 계산 없이 **전치 + 행렬-벡터 곱** 한 번이면 된다.

---

## 4. $4\times4$ 역행렬 `inverse(viewmat)[:3,3]`과 같은 결과인 이유

`gsplat/rendering.py` 쪽에서는 다른 방식으로 같은 값을 얻는다:

```python
camtoworlds = torch.inverse(viewmats)      # camera→world
dirs = means - camtoworlds[:, :3, 3]       # μ − cam_pos
```

즉 $4\times4$ 행렬을 통째로 뒤집은 뒤 **4열의 위쪽 3개**를 카메라 위치로 쓴다. 왜 이게 $-R^\top\mathbf{t}$와 같을까?

$4\times4$ 행렬의 역행렬을 추측해서 곱해보면 된다. 후보:

$$M = \begin{pmatrix} R & \mathbf{t}\\ \mathbf{0}^\top & 1\end{pmatrix},
\qquad
M' = \begin{pmatrix} R^\top & -R^\top\mathbf{t}\\ \mathbf{0}^\top & 1\end{pmatrix}$$

블록끼리 곱하면(블록을 하나의 원소처럼 보고 행×열):

$$M'M = \begin{pmatrix} R^\top R & R^\top\mathbf{t} + (-R^\top\mathbf{t})\\ \mathbf{0}^\top & 1\end{pmatrix}
= \begin{pmatrix} I & \mathbf{0}\\ \mathbf{0}^\top & 1\end{pmatrix} = I_4$$

따라서 $M' = M^{-1}$이고, 그 **4열 위쪽 3개**가 정확히 $-R^\top\mathbf{t}$다.

의미로도 자연스럽다. $M^{-1}$은 camera→world 변환이므로, 카메라 좌표 원점 $\mathbf{0}$을 넣으면 세계 좌표의 카메라 위치가 나온다. 그런데 동차좌표에서 $(\mathbf{0},1)$을 곱하는 것은 **4열을 그대로 꺼내는 것**과 같다. 그래서 "camera→world 행렬의 마지막 열 = 카메라 위치"다.

정리하면 두 코드는 같은 값을 계산한다.

| 방식 | 코드 | 비용 |
|---|---|---|
| 구조를 이용 | `-torch.einsum("cij,ci->cj", R_cam, t_cam)` | 전치 + 곱 (싸다, 수치적으로 안정) |
| 일반 역행렬 | `torch.inverse(viewmats)[:, :3, 3]` | $4\times4$ 역행렬 (불필요하게 비쌈) |

실제로 랜덤한 직교 $R$ 두 개로 두 값을 비교하면 차이가 $\sim 10^{-7}$(float32 반올림 수준)로 일치한다.

---

## 5. einsum 표기 `"cij,ci->cj"` 읽는 법

```python
R_cam, t_cam = viewmats[:, :3, :3], viewmats[:, :3, 3]   # [C,3,3], [C,3]
campos = -torch.einsum("cij,ci->cj", R_cam, t_cam)       # −Rᵀ t   [C,3]
```

`einsum`은 **시그마 기호를 문자열로 쓴 것**이다. 규칙은 딱 두 줄:

1. 입력들에 붙은 첨자를 그대로 곱한다.
2. **화살표 오른쪽에 없는 첨자는 모두 $\sum$으로 더한다.**

여기서 `c`는 카메라 번호(배치 축)로 양쪽과 출력에 모두 있으니 그냥 카메라마다 따로 계산한다는 뜻이고, 실제 계산은 각 $c$에 대해

$$\text{campos}_j = -\sum_{i} R_{ij}\, t_i$$

이다(`i`가 출력에 없으니 $i$에 대해 합). 이제 이게 왜 $R^\top\mathbf{t}$인지 보자. 전치의 정의는 $(R^\top)_{ji} = R_{ij}$이므로

$$\big(R^\top \mathbf{t}\big)_j = \sum_i (R^\top)_{ji}\, t_i = \sum_i R_{ij}\, t_i$$

두 식이 완전히 같다. **첫 번째 첨자 `i`를 따라 더한다 = $R$의 열 방향으로 훑는다 = 전치를 곱한다.** 전치 행렬을 실제로 만들지 않고 "더하는 축만 바꿔서" 전치 효과를 낸 것이다.

### 왜 `"cij,cj->ci"`가 아닌가

`"cij,cj->ci"`였다면

$$\text{out}_i = \sum_j R_{ij}\,t_j = (R\mathbf{t})_i$$

즉 전치 없는 **$R\mathbf{t}$**가 된다. 이건 완전히 다른 벡터다. 기억법:

- `cij,cj->ci` : 두 번째 첨자(열, `j`)로 합 → 평범한 $R\mathbf{t}$
- `cij,ci->cj` : 첫 번째 첨자(행, `i`)로 합 → $R^\top\mathbf{t}$ ← **이쪽이 정답**

$R$이 직교행렬이라 $R\mathbf{t}$도 길이는 $|\mathbf{t}|$로 같아서 "그럴듯해 보이는" 값이 나온다. 그래서 이 실수는 오류 없이 조용히 잘못된 그림을 만든다. 실제로 랜덤 회전으로 두 결과를 비교하면 차이가 $3.28$ 정도로 완전히 다른 점이 나온다.

그리고 코드에서 $R^\top$이 곧 $R^{-1}$이라는 사실을 알기 때문에, 이 한 줄은 결국 "**world→camera 변환을 거꾸로 돌려 카메라의 원점을 세계로 보내라**"는 문장이다.

---

## 6. 2차원 수치 예로 검산 ($\theta = 30^\circ$, $\mathbf{t}=(1,2)$)

3차원은 손으로 계산하기 번거로우니, 원리가 똑같은 2차원으로 직접 확인하자.

$$R = \begin{pmatrix}\cos 30^\circ & -\sin 30^\circ\\ \sin 30^\circ & \cos 30^\circ\end{pmatrix}
= \begin{pmatrix}0.8660 & -0.5\\ 0.5 & 0.8660\end{pmatrix},
\qquad \mathbf{t} = \begin{pmatrix}1\\2\end{pmatrix}$$

**1단계 — 전치:**

$$R^\top = \begin{pmatrix}0.8660 & 0.5\\ -0.5 & 0.8660\end{pmatrix}$$

**2단계 — $R^\top\mathbf{t}$ 계산:**

$$R^\top\mathbf{t} = \begin{pmatrix}0.8660\cdot 1 + 0.5\cdot 2\\ -0.5\cdot 1 + 0.8660\cdot 2\end{pmatrix}
= \begin{pmatrix}0.8660 + 1\\ -0.5 + 1.7321\end{pmatrix}
= \begin{pmatrix}1.8660\\ 1.2321\end{pmatrix}$$

**3단계 — 부호 뒤집기:**

$$\mathbf{c} = -R^\top\mathbf{t} = \begin{pmatrix}-1.8660\\ -1.2321\end{pmatrix}$$

**검산 — 이 점을 카메라 좌표로 보내면 원점이어야 한다:**

$$R\mathbf{c} + \mathbf{t}
= \begin{pmatrix}0.8660\cdot(-1.8660) + (-0.5)\cdot(-1.2321)\\ 0.5\cdot(-1.8660) + 0.8660\cdot(-1.2321)\end{pmatrix} + \begin{pmatrix}1\\2\end{pmatrix}$$

$$= \begin{pmatrix}-1.6160 + 0.6160\\ -0.9330 - 1.0670\end{pmatrix} + \begin{pmatrix}1\\2\end{pmatrix}
= \begin{pmatrix}-1\\-2\end{pmatrix} + \begin{pmatrix}1\\2\end{pmatrix}
= \begin{pmatrix}0\\0\end{pmatrix}\ \checkmark$$

카메라 자신이 카메라 좌표계 원점으로 정확히 떨어진다.

**틀린 쪽과 비교:** 만약 $-R\mathbf{t}$를 썼다면

$$-R\mathbf{t} = -\begin{pmatrix}0.8660 - 1\\ 0.5 + 1.7321\end{pmatrix} = \begin{pmatrix}0.1340\\ -2.2321\end{pmatrix}$$

으로 $(-1.8660,\ -1.2321)$과 전혀 다른 점이다. 길이는 둘 다 $|\mathbf{t}|=\sqrt5\approx 2.236$으로 같지만 방향이 다르다 — "그럴듯하지만 틀린" 값의 정체다.

---

## 7. 한 문장 요약

> **카메라는 자기 좌표계의 원점**이므로 $\mathbf{0} = R\mathbf{c}+\mathbf{t}$이고, 회전행렬은 $R^{-1}=R^\top$이라서 $\mathbf{c} = -R^\top\mathbf{t}$. 코드로는 `-torch.einsum("cij,ci->cj", R_cam, t_cam)`이며, 여기서 첨자 `i`로 합하는 것이 곧 전치다.

### 체크리스트

- [ ] $\mathbf{t}$는 카메라 위치가 **아니다**(세계 원점의 카메라 좌표다)
- [ ] $R^{-1}=R^\top$은 회전행렬(정규직교)이라서 성립한다 — 아무 행렬에나 쓰면 안 된다
- [ ] `cij,ci->cj`는 $R^\top\mathbf{t}$, `cij,cj->ci`는 $R\mathbf{t}$ — 첨자 하나 차이로 결과가 달라진다
- [ ] `torch.inverse(viewmats)[:, :3, 3]`과 같은 값이지만, 구조를 아는 쪽이 더 싸다
