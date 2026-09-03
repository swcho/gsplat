# SH 평가를 행렬 곱으로 — 고교 수학에서 출발하기

> 카드의 답: SH 평가는 **$1\times K$ 행벡터** $[Y_0(\mathbf d)\ \cdots\ Y_{K-1}(\mathbf d)]$ 와 **$K\times 3$ 계수 행렬** $[\mathbf c_0^\top;\ \cdots;\ \mathbf c_{K-1}^\top]$ 의 곱이고, 결과는 **$1\times 3$ RGB 색**이다.
>
> "행렬 곱"을 처음 본다면 낯설 수 있다. 하지만 고교 기하에서 배운 **벡터 내적** 하나만 알면 아래 다섯 단계로 그대로 이어진다.

---

## 0. 먼저, SH 평가가 뭘 계산하는 건지

3DGS의 Gaussian 하나는 색을 RGB 값 하나로 저장하지 않고, **"어느 방향에서 보느냐"에 따라 달라지는 색 함수** $\mathbf c(\mathbf d)$ 로 저장한다. $\mathbf d$ 는 카메라가 그 Gaussian을 바라보는 방향(단위벡터)이다.

이 함수는 미리 정해진 **기저 함수** $Y_0(\mathbf d), Y_1(\mathbf d), \dots, Y_{K-1}(\mathbf d)$ (Spherical Harmonics) 에 **계수** $\mathbf c_0, \dots, \mathbf c_{K-1}$ 를 곱해 더한 꼴이다.

$$
\mathbf c(\mathbf d) = \sum_{k=0}^{K-1} \mathbf c_k\, Y_k(\mathbf d)
$$

- $Y_k(\mathbf d)$ 는 **숫자 하나** (방향을 넣으면 실수가 나오는 함수. 예: $Y_0 = 0.2821$ 상수, $Y_2(\mathbf d)=0.4886\,z$ 등 — 노트북 1.2절의 다항식).
- $\mathbf c_k$ 는 **RGB 세 숫자** $(c_k^R, c_k^G, c_k^B)$ — 이것이 학습으로 맞추는 파라미터다.
- 3DGS는 3차까지 쓰므로 $K=(3+1)^2=16$.

"평가"란 방향 $\mathbf d$ 가 주어졌을 때 이 합을 **실제로 계산해서 색 하나를 얻는 것**이다. 이 합을 "행렬 곱"이라는 한 줄로 쓰는 것이 카드의 내용이다.

---

## 1단계. 벡터 내적 (고교 기하 복습)

두 벡터 $\vec a=(a_1,a_2,a_3)$, $\vec b=(b_1,b_2,b_3)$ 의 내적은

$$
\vec a\cdot\vec b = a_1b_1+a_2b_2+a_3b_3
$$

"같은 자리끼리 곱해서 모두 더한다"는 규칙이다. 성분이 3개가 아니라 $K$개여도 똑같다.

$$
\vec a\cdot\vec b=\sum_{k=1}^{K}a_kb_k
$$

**이제 위 SH 평가 식을 채널 하나(예: R)만 떼어 보자.**

$$
c^R(\mathbf d)=\sum_{k=0}^{K-1} Y_k(\mathbf d)\,c_k^R
= \big(Y_0(\mathbf d),\dots,Y_{K-1}(\mathbf d)\big)\cdot\big(c_0^R,\dots,c_{K-1}^R\big)
$$

즉 **R 채널 색 = "기저값 벡터" 와 "R 계수 벡터" 의 내적**이다. G, B 채널도 마찬가지로 각각 내적 하나씩이다. SH 평가는 결국 **내적 세 번**이다.

---

## 2단계. 행렬 곱 = "행 하나 × 열 하나 = 내적 하나"

내적 여러 개를 한 번에 표기하는 도구가 **행렬 곱**이다. 규칙은 하나뿐이다.

> **결과의 $(i, j)$ 자리 = 왼쪽 행렬의 $i$번째 행 과 오른쪽 행렬의 $j$번째 열 의 내적**

$2\times2$ 예로 확인해 보자.

$$
\begin{bmatrix} 1 & 2\\ 3 & 4\end{bmatrix}
\begin{bmatrix} 5 & 6\\ 7 & 8\end{bmatrix}
=
\begin{bmatrix}
(1,2)\cdot(5,7) & (1,2)\cdot(6,8)\\
(3,4)\cdot(5,7) & (3,4)\cdot(6,8)
\end{bmatrix}
=
\begin{bmatrix} 19 & 22\\ 43 & 50\end{bmatrix}
$$

- 왼쪽 1행 $(1,2)$ 와 오른쪽 1열 $(5,7)$ 의 내적 $1\cdot5+2\cdot7=19$ → 결과 (1,1).
- 왼쪽 1행 $(1,2)$ 와 오른쪽 2열 $(6,8)$ 의 내적 $6+16=22$ → 결과 (1,2).
- 나머지도 같은 식.

**크기 규칙**: $(m\times K)$ 행렬 × $(K\times n)$ 행렬 = $(m\times n)$ 행렬. 안쪽의 $K$ 가 같아야 내적이 정의된다(성분 개수가 맞아야 하니까). 바깥쪽 $m, n$ 이 결과의 크기다.

행렬이 행 하나짜리 $(1\times K)$ 이면 그냥 **행벡터**, 열 하나짜리 $(K\times1)$ 이면 **열벡터**라고 부른다. 행벡터 × 열벡터 $=(1\times1)$ = 숫자 하나 = 내적. 이것이 "행렬 곱은 내적의 묶음"이라는 말의 뜻이다.

---

## 3단계. SH 평가 = $(1\times K)$ 행벡터 × $(K\times3)$ 행렬

이제 카드의 식을 읽어 보자.

$$
\underbrace{\big[\,Y_0(\mathbf d)\ \ Y_1(\mathbf d)\ \cdots\ Y_{K-1}(\mathbf d)\,\big]}_{1\times K}
\;
\underbrace{\begin{bmatrix}
c_0^R & c_0^G & c_0^B\\
c_1^R & c_1^G & c_1^B\\
\vdots & \vdots & \vdots\\
c_{K-1}^R & c_{K-1}^G & c_{K-1}^B
\end{bmatrix}}_{K\times 3}
=
\underbrace{\big[\,c^R(\mathbf d)\ \ c^G(\mathbf d)\ \ c^B(\mathbf d)\,\big]}_{1\times 3}
$$

- **왼쪽**: 방향 $\mathbf d$ 에서 계산한 기저값 $K$개를 한 줄로 늘어놓은 행벡터. 방향이 정해지면 그냥 숫자 $K$개다.
- **오른쪽**: $k$번째 행이 $\mathbf c_k^\top=(c_k^R, c_k^G, c_k^B)$. 즉 계수 벡터를 **가로로 눕혀** 위에서부터 쌓은 것($\top$ 는 "세로를 가로로 뒤집는다"는 전치 기호). 열은 R, G, B 채널.
- **결과**: 크기 규칙 $(1\times K)(K\times3)=(1\times3)$. 세 성분은
  - 1열: 기저 행벡터 · R 계수 열 $=\sum_k Y_k c_k^R = c^R(\mathbf d)$
  - 2열: 기저 행벡터 · G 계수 열 $= c^G(\mathbf d)$
  - 3열: 기저 행벡터 · B 계수 열 $= c^B(\mathbf d)$

정확히 1단계에서 본 "내적 세 번"이 행렬 곱 한 줄에 담겼다.

**구체 크기 (3DGS, $K=16$)**: $(1\times16)\times(16\times3)=(1\times3)$. 곱셈-덧셈이 채널당 16번, 총 48번이면 Gaussian 하나의 색이 나온다. 노트북 4절이 "채널마다 16번의 곱셈-덧셈"이라 적은 것이 이것이다.

> 여기에 3DGS는 결과에 $+0.5$ 를 더하고 음수를 0으로 잘라(`clamp_min(0)`) 최종 색으로 쓴다. 행렬 곱 자체는 위가 전부다.

---

## 4단계. Gaussian이 $N$개면? — 행이 $N$개로, 하지만 "배치 내적"

장면에는 Gaussian이 수십만 개 있다. Gaussian $n$ 마다

- 카메라가 그것을 보는 방향 $\mathbf d_n$ 이 **다르고**,
- 계수 행렬 $C_n$ ($16\times3$) 도 **다르다**.

방향이 다르니 기저 행벡터 $[Y_0(\mathbf d_n)\cdots Y_{15}(\mathbf d_n)]$ 도 Gaussian마다 다르다. 이 행벡터들을 위아래로 쌓으면 $N\times16$ 행렬 $B$ 가 된다 (`sh_bases(d, 3)` 의 출력 모양 `[N, 16]`).

**주의할 점**: 2단계의 일반 행렬 곱 $B\,C$ 는 "모든 행이 **같은** 오른쪽 행렬 $C$ 와 곱해진다"는 뜻이다. 하지만 지금은 $n$번째 행은 $n$번째 계수 행렬 $C_n$ 과만 곱해야 한다. 즉 **하나의 커다란 행렬 곱이 아니라, 서로 독립인 $N$개의 $(1\times16)(16\times3)$ 곱을 나란히 수행**하는 것이다. 이를 **배치(batch) 내적**이라 부른다.

노트북의 `sh_eval` 이 바로 이것이다.

```python
def sh_eval(coeffs, dirs, degree):          # coeffs [N,16,3], dirs [N,3]
    K = (degree + 1) ** 2
    d = F.normalize(dirs, dim=-1)           # 단위벡터로
    return torch.einsum("nk,nkc->nc", sh_bases(d, degree), coeffs[:, :K])
```

`einsum("nk,nkc->nc")` 의 첨자 문자열을 읽는 법:

| 기호 | 뜻 |
|---|---|
| `nk` | 첫 입력 = 기저 행렬, 크기 $N\times K$. `n`은 Gaussian 번호, `k`는 기저 번호 |
| `nkc` | 둘째 입력 = 계수, 크기 $N\times K\times 3$. `c`는 채널(R,G,B) |
| `->nc` | 출력은 $N\times3$. **출력에서 사라진 첨자 `k` 에 대해 곱해서 더한다** |

즉 각 $n$, 각 $c$ 마다 $\sum_k B_{nk}\,C_{nkc}$ — 3단계의 내적을 Gaussian $N$개 × 채널 3개 만큼 반복한 것이다. 첨자 `n`이 두 입력에 모두 있고 출력에도 남아 있으므로 "n은 서로 짝지어진 것끼리만 곱한다"(배치)는 뜻이 되고, `k`는 출력에서 사라지므로 "k에 대해 합한다"(내적)는 뜻이 된다.

> 비교: 노트북 4.1절의 `einsum("abk,kc->abc", B, one[0])` 는 **Gaussian 하나**의 계수 `[16,3]` 를 여러 방향 격자 `[a,b,16]` 에 모두 곱한다. 여기서는 오른쪽 계수 행렬이 하나이므로 그냥 보통 행렬 곱이다. 계수에 `n` 이 붙는지 여부가 "일반 행렬 곱"과 "배치 내적"을 가른다.

---

## 5단계. 왜 굳이 이렇게 쓰는가 — GPU 병렬성과 미분

**(a) 병렬 처리.** 4단계의 $N$개 내적은 서로 완전히 독립이다 (Gaussian 3번의 색을 구하는 데 7번의 정보가 전혀 필요 없다). GPU는 수천 개의 작은 계산을 동시에 하는 장치이므로, "독립적인 $(1\times16)(16\times3)$ 곱 $N$개"는 GPU에 이상적인 작업이다. gsplat은 이 계산을 CUDA 커널 하나(`spherical_harmonics(...)`)로 만들어 모든 Gaussian을 한 번에 처리한다.

**(b) 미분이 쉽다.** 학습은 "렌더 결과와 사진의 차이"를 계수 $\mathbf c_k$ 에 대해 미분해서 계수를 고치는 과정이다(노트북 4.2절의 Adam). 3단계의 R 채널 식

$$
c^R(\mathbf d)=Y_0(\mathbf d)\,c_0^R+Y_1(\mathbf d)\,c_1^R+\cdots+Y_{15}(\mathbf d)\,c_{15}^R
$$

을 보면, 계수 $c_k^R$ 에 대해 **1차식(선형)** 이다. 고교 미분으로 $\frac{d}{dx}(ax)=a$ 이듯이

$$
\frac{\partial c^R}{\partial c_k^R}=Y_k(\mathbf d)
$$

즉 **계수에 대한 기울기는 그냥 기저 행벡터의 $k$번째 성분**이다. 앞으로 계산(forward)에서 이미 구해 둔 $[Y_0(\mathbf d)\cdots Y_{15}(\mathbf d)]$ 를 그대로 다시 쓰면 되니, 역전파(backward)에 추가 비용이 거의 없다. 채널 R,G,B 모두 같은 기저 행벡터를 공유하므로 기울기도 같은 값을 세 번 쓴다.

$Y_k(\mathbf d)$ 자체는 $x,y,z$ 의 다항식이라 $\mathbf d$ 에 대해서도 미분 가능하지만(Gaussian 위치를 학습할 때 필요), 이 카드의 핵심은 "계수에 대해서는 선형 → 기울기 = 기저값"이라는 단순함이다.

---

## 한 줄 정리

$$
\text{색}(\mathbf d)=\underbrace{[Y_0(\mathbf d)\ \cdots\ Y_{15}(\mathbf d)]}_{1\times16\ \text{방향이 정하는 행벡터}}\ \underbrace{\begin{bmatrix}\mathbf c_0^\top\\ \vdots\\ \mathbf c_{15}^\top\end{bmatrix}}_{16\times3\ \text{학습되는 계수}}=\underbrace{[R\ \ G\ \ B]}_{1\times3}
$$

- **내적 세 번**(채널별)을 행렬 곱 하나로 적은 것.
- Gaussian이 $N$개면 각각 방향·계수가 달라 `einsum("nk,nkc->nc")` — 독립적인 내적 $N\times3$개의 **배치**.
- 계수에 대해 선형이라 GPU 병렬화가 쉽고, 기울기 $\partial\text{색}/\partial \mathbf c_k = Y_k(\mathbf d)$ 가 공짜로 나온다.
