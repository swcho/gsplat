# 카메라는 어디에 있나 — world→camera 행렬 $[R\,|\,\mathbf t]$에서 위치를 되찾는 법

> 정답: $\mathbf o_{\text{cam}} = -R^\top\mathbf t$. 왜 이렇게 되는지를 고교 도형의 이동·회전에서 출발해 단계적으로 쌓아 올린다.

---

## 1. 좌표계 옮기기 — 평행이동과 회전 (2D에서 시작)

고교 기하에서 배운 **도형의 평행이동**과 **회전변환**을 떠올려 보자.

- 평행이동: 점 $(x, y)$를 $(a, b)$만큼 옮기면 $(x+a,\ y+b)$.
- 원점 중심 회전(각 $\theta$): $(x, y) \mapsto (x\cos\theta - y\sin\theta,\ x\sin\theta + y\cos\theta)$.

여기서 관점을 살짝 바꾸자. **도형을 옮기는 대신, 도형은 그대로 두고 "좌표계(눈금판)"를 옮긴다**고 생각해도 똑같은 식이 나온다. 3DGS에서 이 눈금판이 두 개다.

- **world 좌표계**: 장면(Gaussian들)이 놓여 있는 고정된 눈금판.
- **camera 좌표계**: 카메라 렌즈 중심을 원점으로 하고, 카메라가 보는 방향을 한 축($+z$)으로 잡은 눈금판.

같은 점 $P$라도 world 눈금으로 읽은 좌표 $\mathbf x_w$와 camera 눈금으로 읽은 좌표 $\mathbf x_c$는 다르다. 이 둘을 바꿔 주는 규칙이 바로 **world→camera 변환**이다.

2D 예로 감을 잡자. 카메라가 world 좌표 $(3, 1)$에 있고, 카메라 눈금판은 world 눈금판을 $\theta$만큼 돌린 것이라고 하자. 어떤 점의 camera 좌표를 얻으려면

1. 먼저 카메라 위치가 원점이 되도록 **평행이동**: $(x-3,\ y-1)$
2. 그다음 카메라 축에 맞게 **회전**: $-\theta$만큼 돌린다 (눈금판을 $+\theta$ 돌린 것은 점을 $-\theta$ 돌린 것과 같다)

"평행이동 뒤 회전" — 이 두 단계가 3D에서도 그대로 이어진다.

---

## 2. 회전을 행렬로 쓰면 $R$ — 그리고 $R^{-1} = R^\top$인 이유

### 2.1 행렬로 쓰기

위 회전식은 벡터와 행렬의 곱으로 정리된다. 행렬 곱은 "각 행과 열벡터의 내적"이라는 규칙만 알면 된다.

$$
\begin{pmatrix} x' \\ y' \end{pmatrix}
=
\underbrace{\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}}_{R}
\begin{pmatrix} x \\ y \end{pmatrix}
$$

3D에서는 $R$이 $3\times 3$ 행렬이 되고, 세 축 중 어느 쪽으로 얼마나 돌리든 결국 하나의 $3\times3$ 회전 행렬 $R$로 표현된다.

### 2.2 회전 행렬의 열은 서로 수직인 단위벡터

$R$의 열벡터를 살펴보자. 2D 예에서 첫 열은 $(\cos\theta, \sin\theta)$, 둘째 열은 $(-\sin\theta, \cos\theta)$.

- 각 열의 길이: $\cos^2\theta + \sin^2\theta = 1$ → **단위벡터**
- 두 열의 내적: $-\cos\theta\sin\theta + \sin\theta\cos\theta = 0$ → **서로 수직**

이는 당연하다. 회전은 **길이와 각도를 보존**하므로, 원래 서로 수직이던 단위벡터 $(1,0)$, $(0,1)$이 회전 후에도 서로 수직인 단위벡터로 남는다. 그리고 $R\begin{pmatrix}1\\0\end{pmatrix}$이 첫 열, $R\begin{pmatrix}0\\1\end{pmatrix}$이 둘째 열이다. 이런 행렬을 **직교 행렬**이라 부른다.

### 2.3 그래서 $R^\top R = I$

전치 $R^\top$는 행과 열을 맞바꾼 행렬이다. $R^\top R$의 $(i, j)$ 성분은 "$R$의 $i$번째 열 · $R$의 $j$번째 열"(내적)이다. 위에서 본 대로

- $i = j$일 때: 단위벡터의 자기 내적 = 1
- $i \ne j$일 때: 수직인 벡터의 내적 = 0

즉 $R^\top R = I$(단위행렬). 역행렬의 정의가 "곱해서 $I$가 되는 행렬"이므로

$$
R^{-1} = R^\top .
$$

직관: $\theta$만큼 돌린 것을 되돌리는 회전은 $-\theta$ 회전이고, $\cos(-\theta)=\cos\theta$, $\sin(-\theta) = -\sin\theta$를 대입하면 정확히 $R$의 전치가 나온다. 3D에서도 마찬가지다. **역행렬 계산(가우스 소거 등)이 필요 없고 그냥 뒤집으면 된다** — 이것이 이 카드의 핵심 도구다.

---

## 3. world→camera 변환에서 카메라 위치 뽑아내기

### 3.1 변환식

1절의 "평행이동 뒤 회전"을 3D 행렬로 쓰면, 어떤 점의 world 좌표 $\mathbf x_w$와 camera 좌표 $\mathbf x_c$ 사이에

$$
\mathbf x_c = R\,(\mathbf x_w - \mathbf o_{\text{cam}}) = R\,\mathbf x_w + \underbrace{(-R\,\mathbf o_{\text{cam}})}_{\mathbf t}
$$

가 성립한다. 여기서 $\mathbf o_{\text{cam}}$은 카메라 중심의 **world 좌표**(우리가 찾고 싶은 것)이고, 코드에 저장되는 것은 $R$과 $\mathbf t = -R\,\mathbf o_{\text{cam}}$이다. 즉 **$\mathbf t$는 카메라 위치가 아니다.** 회전이 끼어 있는 "가공된" 값이다.

### 3.2 카메라 자신은 camera 좌표로 원점

카메라 좌표계는 카메라 렌즈 중심을 원점으로 잡았다. 따라서 카메라 위치 $\mathbf o_{\text{cam}}$을 변환식에 넣으면 결과는 반드시 $\mathbf 0$이어야 한다.

$$
\mathbf 0 = R\,\mathbf o_{\text{cam}} + \mathbf t
$$

이제 $\mathbf o_{\text{cam}}$에 대해 풀자. 이것은 고교 수학의 일차방정식 $ax + b = 0 \Rightarrow x = -b/a$와 완전히 같은 구조다. 다만 "나누기" 대신 "역행렬 곱하기"를 하고, 2절 덕분에 역행렬은 $R^\top$이다.

$$
R\,\mathbf o_{\text{cam}} = -\mathbf t
\;\;\Longrightarrow\;\;
R^\top R\,\mathbf o_{\text{cam}} = -R^\top\mathbf t
\;\;\Longrightarrow\;\;
\boxed{\mathbf o_{\text{cam}} = -R^\top\mathbf t}
$$

### 3.3 흔한 실수 두 가지

| 잘못된 식 | 왜 틀린가 |
|---|---|
| $\mathbf o = -\mathbf t$ | $R = I$(카메라가 안 돌아간 경우)에만 우연히 맞다. 회전이 있으면 $\mathbf t$는 이미 $R$로 돌려진 값이라 되돌려 주어야 한다. |
| $\mathbf o = -R\,\mathbf t$ | 되돌리는 방향이 반대. $R$을 한 번 더 적용하면 회전이 두 배가 된다. 되돌리기는 $R^{-1} = R^\top$. |

행렬 곱은 순서에 민감하다(교환법칙이 성립하지 않는다). "$R$을 곱했으니 $R^\top$로 벗겨 낸다"는 감각을 기억하자.

---

## 4. 노트북 예제로 손계산

`sh_walkthrough.py` 4절의 예제다. 카메라가 world 좌표 $(0, 0, -5)$에서 $+z$ 방향을 바라본다. 카메라 축이 world 축과 같은 방향이므로 회전은 없다: $R = I$.

$\mathbf t = -R\,\mathbf o_{\text{cam}} = -I\,(0, 0, -5)^\top = (0, 0, 5)^\top$. 코드도 그렇게 만든다.

```python
viewmat = torch.eye(4); viewmat[:3, 3] = torch.tensor([0., 0., 5.])   # R = I, t = (0,0,5)
```

이제 공식으로 카메라 위치를 복원하면

$$
\mathbf o_{\text{cam}} = -R^\top\mathbf t = -I\,(0,0,5)^\top = (0, 0, -5)^\top . \checkmark
$$

의미를 확인하자. world 원점에 있는 Gaussian $(0,0,0)$을 camera 좌표로 옮기면 $R\,\mathbf x_w + \mathbf t = (0, 0, 5)$ — "카메라 앞 5만큼 떨어진 곳"이다. 카메라가 $z=-5$에 있으니 원점은 카메라보다 $+5$ 앞에 있는 것이 맞다.

그리고 SH 평가에 필요한 **시점 방향**은 카메라 위치에서 Gaussian 중심을 향하는 벡터

$$
\mathbf d = \frac{\boldsymbol\mu - \mathbf o_{\text{cam}}}{\|\boldsymbol\mu - \mathbf o_{\text{cam}}\|}
$$

이다. $\boldsymbol\mu = (0,0,0)$이면 $\mathbf d = (0,0,5)/5 = (0,0,1)$; $\boldsymbol\mu = (2,1,3)$이면 $(2,1,8)/\sqrt{69} \approx (0.241, 0.120, 0.963)$. 노트북의 `view_dirs` 함수가 정확히 `means - (-R.T @ t)`를 계산한다. 카메라 위치를 $-\mathbf t = (0,0,-5)$로 잘못 잡아도 이 예제에서는 우연히 맞지만($R = I$), 카메라를 돌리는 순간 방향이 어긋나 SH가 잘못된 색을 낸다.

---

## 5. $4\times 4$ 동차 행렬과 c2w의 4번째 열

3D 그래픽스에서는 회전과 평행이동을 한 행렬에 담기 위해 **동차 좌표** $(x, y, z, 1)$을 쓴다. 벡터 끝에 1을 붙이면 "행렬 곱 하나"로 평행이동까지 표현할 수 있다.

$$
\begin{pmatrix}\mathbf x_c \\ 1\end{pmatrix}
=
\underbrace{\begin{pmatrix} R & \mathbf t \\ \mathbf 0^\top & 1 \end{pmatrix}}_{\text{w2c (viewmat)}}
\begin{pmatrix}\mathbf x_w \\ 1\end{pmatrix}
$$

이 행렬의 역행렬(camera→world, **c2w**)은 3절의 계산을 그대로 담고 있다. $\mathbf x_c = R\mathbf x_w + \mathbf t$를 $\mathbf x_w$에 대해 풀면 $\mathbf x_w = R^\top\mathbf x_c - R^\top\mathbf t$이므로

$$
\text{c2w} = \text{w2c}^{-1} =
\begin{pmatrix} R^\top & -R^\top\mathbf t \\ \mathbf 0^\top & 1 \end{pmatrix}.
$$

여기서 **c2w의 4번째 열(평행이동 부분)이 곧 $-R^\top\mathbf t = \mathbf o_{\text{cam}}$**이다. 그 이유는 직관적이다. c2w는 "camera 좌표 → world 좌표" 변환인데, camera 원점 $(0,0,0,1)$을 넣으면 결과는 4번째 열 그 자체다. camera 원점이 world에서 어디냐 — 바로 카메라 위치다.

정리하면 카메라 위치를 얻는 두 경로는 같은 답을 준다.

1. w2c에서 직접: $\mathbf o_{\text{cam}} = -R^\top\mathbf t$
2. w2c의 $4\times4$ 역행렬을 구한 뒤 4번째 열의 위 세 성분을 읽기 (`torch.linalg.inv(viewmat)[:3, 3]`)

경로 1은 역행렬 계산 없이 전치와 곱셈 한 번으로 끝나므로, gsplat의 SH 커널처럼 카메라·Gaussian마다 수백만 번 반복되는 곳에서는 이쪽을 쓴다.

---

## 한 줄 정리

카메라는 자기 좌표계의 원점이므로 $\mathbf 0 = R\,\mathbf o_{\text{cam}} + \mathbf t$; 회전 행렬의 역은 전치($R^{-1} = R^\top$)이므로 $\mathbf o_{\text{cam}} = -R^\top\mathbf t$. 이 값이 c2w 행렬의 4번째 열이며, SH 시점 방향 $\mathbf d = (\boldsymbol\mu - \mathbf o_{\text{cam}})/\|\cdot\|$의 출발점이다.
