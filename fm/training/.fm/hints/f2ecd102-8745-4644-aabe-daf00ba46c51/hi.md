# 카메라 위치는 왜 `camtoworlds[:, :3, 3]` 인가

## 0. 한 줄 답

`parser.camtoworlds`는 각 학습 이미지에 대한 **camera-to-world 4×4 행렬** $N$개를 담은 배열 `[N,4,4]`이다.
이 행렬의 오른쪽 위 3×1 열(= 평행이동 성분 $\mathbf{t}$)이 **그 카메라가 월드 좌표계에서 놓인 지점**이다.
그래서 `[:, :3, 3]` 슬라이스 한 번으로 `[N,3]` 크기의 카메라 위치 배열이 나온다.

```python
cam_pos = parser.camtoworlds[:, :3, 3]   # [N,3] 월드 좌표계의 카메라 위치
ax.scatter(*cam_pos.T, c="red", s=12, marker="^", label="cameras")
```

아래에서 "왜 평행이동 성분이 곧 위치인가"를 고교 수학(벡터, 행렬의 회전변환)에서 출발해 쌓아 올린다.

---

## 1. 출발점: 고교에서 이미 아는 두 가지

**(1) 벡터와 좌표계.** 공간의 한 점은 기준점(원점)과 세 축을 정하면 숫자 세 개 $(x,y,z)$로 표현된다.
중요한 건 **같은 점이라도 기준을 바꾸면 숫자가 달라진다**는 사실이다.

**(2) 행렬은 변환이다.** 기하 과정에서 배운 평면의 회전변환

$$
R(\theta)=\begin{pmatrix}\cos\theta & -\sin\theta\\[2pt] \sin\theta & \cos\theta\end{pmatrix},
\qquad
\begin{pmatrix}x'\\y'\end{pmatrix} = R(\theta)\begin{pmatrix}x\\y\end{pmatrix}
$$

처럼, 행렬을 벡터에 곱하면 벡터가 **돌아간다**. 3차원에서도 똑같이 $3\times3$ 회전행렬 $R$이 있고,
회전은 길이와 각도를 보존한다. 이 "길이 보존"이 뒤(6절)에서 결정적으로 쓰인다.

---

## 2. 좌표계가 두 개 있다

3DGS 학습에서는 좌표계를 두 개 동시에 쓴다.

| 좌표계 | 원점 | 축 | 누가 쓰나 |
|---|---|---|---|
| **월드(world)** | 장면 어딘가에 고정된 한 점 | 장면에 고정 | Gaussian의 `means`, SfM 포인트 `parser.points` |
| **카메라(camera)** | 그 사진을 찍은 렌즈 중심 | $z$축이 바라보는 방향 | 투영·렌더링 계산 |

사진이 $N$장이면 **카메라 좌표계도 $N$개**다. 그래서 "$i$번째 카메라 좌표계 → 월드 좌표계"로
번역해 주는 변환이 사진마다 하나씩 필요하다. 그 번역기가 바로 `camtoworlds[i]`다.

---

## 3. 번역기의 정체: 회전 + 평행이동

카메라 좌표 $\mathbf{x}_c$를 월드 좌표 $\mathbf{x}_w$로 바꾸는 일은 두 단계다.

1. 축 방향을 맞춘다 → 회전행렬 $R$ ($3\times3$)
2. 원점을 옮긴다 → 평행이동 벡터 $\mathbf{t}$ ($3\times1$)

$$
\mathbf{x}_w \;=\; R\,\mathbf{x}_c \;+\; \mathbf{t}
$$

문제는 이 식이 **행렬 곱 하나로 안 써진다**는 것이다. $+\mathbf{t}$ 라는 덧셈이 붙어 있어서
$3\times3$ 행렬 하나로 표현할 수 없다(원점이 원점으로 안 가므로 선형변환이 아니다).

### 해결책: 동차좌표(homogeneous coordinates)

좌표 뒤에 숫자 $1$을 하나 덧붙여 4차원으로 만드는 트릭을 쓴다.

$$
\begin{pmatrix}\mathbf{x}_w\\[2pt] 1\end{pmatrix}
=
\underbrace{\begin{pmatrix} R & \mathbf{t}\\[2pt] \mathbf{0}^{\top} & 1\end{pmatrix}}_{M_{c\to w}\;(4\times4)}
\begin{pmatrix}\mathbf{x}_c\\[2pt] 1\end{pmatrix}
$$

곱셈을 손으로 풀어 보면 위쪽 3줄이 $R\mathbf{x}_c + 1\cdot\mathbf{t}$, 맨 아래 줄이 $0\cdot\mathbf{x}_c+1\cdot1=1$이다.
즉 **마지막 열의 $\mathbf{t}$가 "항상 1인 성분"과 곱해지면서 덧셈 역할을 대신한다.**
이게 4×4 행렬을 쓰는 유일한 이유이고, `camtoworlds`가 `[N,4,4]`인 이유다.

성분을 다 펼치면 이렇게 생겼다.

$$
M_{c\to w}=
\begin{pmatrix}
r_{11} & r_{12} & r_{13} & t_x\\
r_{21} & r_{22} & r_{23} & t_y\\
r_{31} & r_{32} & r_{33} & t_z\\
0 & 0 & 0 & 1
\end{pmatrix}
$$

---

## 4. 핵심 논증: 카메라 원점을 그냥 대입해 보자

카메라는 **자기 좌표계의 원점**에 있다. 정의상 그렇다(카메라 좌표계의 원점 = 렌즈 중심).
그러니 카메라 자신의 카메라 좌표는

$$
\mathbf{x}_c=\begin{pmatrix}0\\0\\0\end{pmatrix}
$$

이다. 이 값을 3절의 식에 넣으면

$$
\mathbf{x}_w = R\begin{pmatrix}0\\0\\0\end{pmatrix} + \mathbf{t} = \mathbf{0}+\mathbf{t} = \mathbf{t}
$$

**회전 항이 통째로 사라진다.** 남는 건 $\mathbf{t}$뿐이다.

> 따라서 camera-to-world 행렬의 평행이동 성분 $\mathbf{t}$는 **그 자체로** 월드 좌표계에서의 카메라 위치다.
> 별도의 계산이 필요 없고, 그래서 "행렬에서 뽑아낸다(슬라이싱)"는 표현이 정확하다.

직관적으로도 자연스럽다. $M_{c\to w}$는 "카메라 좌표계를 월드 좌표계 위에 어떻게 놓았는가"를
기술하는 행렬이고, 물체를 놓을 때 필요한 정보는 **어디에**($\mathbf{t}$) 그리고 **어느 방향으로**($R$)
두 가지뿐이다. 앞의 것이 위치, 뒤의 것이 자세(orientation)다.

---

## 5. 슬라이싱 문법 해부: `[:, :3, 3]`

`parser.camtoworlds`는 축이 3개인 배열이다. 대괄호 안 콤마로 구분된 세 자리가 각 축에 대응한다.

| 자리 | 대상 축 | 쓴 값 | 뜻 |
|---|---|---|---|
| 1번째 | 이미지 인덱스 ($N$) | `:` | 모든 카메라 |
| 2번째 | 행 (4) | `:3` | 0,1,2번 행 — 즉 위 3줄만 (마지막 `0 0 0 1` 줄 버림) |
| 3번째 | 열 (4) | `3` | 3번 열 — 즉 **네 번째(마지막) 열** |

- `:3`은 "0 이상 3 미만" = 0,1,2 → **범위**이므로 그 축이 남는다.
- `3`은 정수 하나 → **그 축이 사라진다**(차원 축소).

그래서 결과 shape은 `[N, 3]`. `cam_pos[i]`가 $i$번째 카메라의 $(x,y,z)$다.

혼동하기 쉬운 이웃들:

```python
camtoworlds[:, :3, 3]    # [N,3]   평행이동 t  → 카메라 위치     ✅
camtoworlds[:, :3, :3]   # [N,3,3] 회전 R      → 카메라가 보는 자세
camtoworlds[:, 3, :3]    # [N,3]   맨 아랫줄 → 항상 (0,0,0). 무의미  ❌
camtoworlds[:, :3, 2]    # [N,3]   R의 3번째 열 → 카메라 z축(광축) 방향
```

마지막 줄은 덤이지만 유용하다. 카메라의 **바라보는 방향**은 $R$의 세 번째 열($z$축을 월드로 옮긴 것)이다.
위치는 마지막 열, 방향은 회전 블록의 열 — 이렇게 짝지어 기억하면 헷갈리지 않는다.

시각화 코드의 `*cam_pos.T`는 `[N,3]`을 전치해 `[3,N]`으로 만든 뒤 언패킹해서
`ax.scatter(xs, ys, zs)`에 넘기는 관용구다.

---

## 6. 가장 흔한 함정: world-to-camera였다면 이 코드는 틀린다

**"평행이동 = 위치"는 camera-to-world일 때만 성립한다.** 방향이 반대인
world-to-camera(= extrinsic, view matrix)에서는 성립하지 않는다.

실제로 COLMAP이 파일에 저장하는 값은 world-to-camera다. gsplat 파서는 그것을 읽어서
(`examples/datasets/colmap.py:110-117`, `_image_w2c`) **명시적으로 역행렬을 취해** 뒤집는다.

```python
# examples/datasets/colmap.py:190
camtoworlds = np.linalg.inv(w2c_mats)   # Convert extrinsics to camera-to-world.
```

역행렬을 손으로 구해 보면 왜 조심해야 하는지 바로 보인다. $R$은 회전행렬이라
길이를 보존하고, 그 결과 열벡터들이 서로 수직인 단위벡터가 되어 $R^{-1}=R^{\top}$ (직교행렬)이다.
이 사실을 쓰면

$$
M_{w\to c} \;=\; M_{c\to w}^{-1} \;=\;
\begin{pmatrix} R^{\top} & -R^{\top}\mathbf{t}\\[2pt] \mathbf{0}^{\top} & 1\end{pmatrix}
$$

(검산: 두 행렬을 곱하면 회전 블록은 $RR^\top=I$, 평행이동 블록은 $R(-R^\top\mathbf{t})+\mathbf{t}=\mathbf{0}$ → 단위행렬.)

즉 world-to-camera 행렬의 평행이동 성분은 $\mathbf{t}$가 아니라 $-R^{\top}\mathbf{t}$다.
여기서 위치를 되찾으려면 한 단계를 더 거쳐야 한다.

$$
\mathbf{t} \;=\; -\,R_{w\to c}^{\top}\,\mathbf{t}_{w\to c}
\quad\Longleftrightarrow\quad
\text{(코드로는)}\;\;\texttt{-R.T @ t}
$$

### 숫자로 확인

카메라가 월드의 $\mathbf{t}=(2,0,1)$에 있고, $z$축 기준 $90^\circ$ 회전한 자세라고 하자.

$$
R=\begin{pmatrix}0&-1&0\\1&0&0\\0&0&1\end{pmatrix},\qquad \mathbf{t}=\begin{pmatrix}2\\0\\1\end{pmatrix}
$$

world-to-camera의 평행이동 성분은

$$
-R^{\top}\mathbf{t}
=-\begin{pmatrix}0&1&0\\-1&0&0\\0&0&1\end{pmatrix}\begin{pmatrix}2\\0\\1\end{pmatrix}
=-\begin{pmatrix}0\\-2\\1\end{pmatrix}
=\begin{pmatrix}0\\2\\-1\end{pmatrix}
$$

$(0,2,-1)$과 실제 위치 $(2,0,1)$은 **부호만 다른 것도 아니고 완전히 다른 점**이다.
회전이 섞여 있으면 눈으로 봐서 "부호 뒤집힌 거겠지" 하고 넘길 수도 없다.
행렬을 받았을 때 **어느 방향 변환인지 먼저 확인**하는 습관이 필요한 이유다.

렌더링 쪽에서 이 뒤집기가 다시 등장한다.

```python
viewmats=torch.linalg.inv(camtoworlds),      # world→cam
```

래스터화는 world-to-camera가 필요하고, 파서가 들고 있는 건 camera-to-world이므로 매번 역행렬을 취한다.

---

## 7. 이 슬라이스는 그림 그리는 데만 쓰이지 않는다

같은 `[:, :3, 3]`이 학습 하이퍼파라미터의 기준을 정하는 데 쓰인다
(`examples/datasets/colmap.py:437` 부근).

```python
camera_locations = camtoworlds[:, :3, 3]
scene_center = np.mean(camera_locations, axis=0)
dists = np.linalg.norm(camera_locations - scene_center, axis=1)
self.scene_scale = np.max(dists)
```

읽어 보면 고교 통계·기하 수준의 계산이다.

1. 카메라 위치들의 **평균점**을 장면 중심으로 잡는다.
2. 각 카메라에서 중심까지의 거리(피타고라스, $\lVert\cdot\rVert$)를 구한다.
3. 그 중 **최댓값**을 `scene_scale`로 삼는다 — "카메라들이 퍼져 있는 반경".

이 값이 `means`의 학습률과 밀도화(densification) 임계값의 **단위**가 된다
(워크스루의 `scene_scale = parser.scene_scale * 1.1`). 장면이 방 하나든 야외든 같은
하이퍼파라미터로 학습되게 만드는 정규화 장치다. `normalize=True`일 때 월드 좌표를 정렬하는
`similarity_from_cameras`도 결국 카메라 위치 집합을 입력으로 쓴다.

그러니 이 카드의 슬라이스를 잘못 쓰면 산점도만 이상해지는 게 아니라, **학습률 스케일이 통째로
어긋난다.** 그래서 학습 전에 SfM 포인트와 카메라 위치를 한 번 찍어 보는 이 시각화 셀이
"디버깅 체크포인트"로서 값을 한다. 카메라 삼각형들이 포인트클라우드를 **둘러싸듯이** 배치되면
정상이고, 한 점에 뭉쳐 있거나 포인트클라우드에서 멀리 떨어져 있으면 좌표 규약을 잘못 읽은 것이다.

---

## 8. 요약

| 질문 | 답 |
|---|---|
| `camtoworlds` shape | `[N,4,4]` — 사진 $N$장, 각각 camera→world 동차변환 |
| 4×4인 이유 | 회전 $R$ + 평행이동 $\mathbf{t}$를 **한 번의 행렬 곱**으로 합치는 동차좌표 트릭 |
| 위치가 왜 마지막 열인가 | 카메라는 자기 좌표계 원점 $\mathbf{0}$에 있고, $R\mathbf{0}+\mathbf{t}=\mathbf{t}$ |
| `[:, :3, 3]`의 뜻 | 모든 카메라 / 위 3행 / 4번째 열 → `[N,3]` |
| 자세는 어디에 | `[:, :3, :3]` ($R$), 광축 방향은 $R$의 3번째 열 |
| world→cam이면 | 평행이동이 $-R^{\top}\mathbf{t}$ → 위치는 `-R.T @ t`로 복원해야 함 |
| 같은 슬라이스의 다른 용도 | `scene_scale` 계산 → 학습률·밀도화 임계값의 기준 단위 |
