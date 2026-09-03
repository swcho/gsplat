# `viewmats`는 어떻게 만드는가 — `torch.linalg.inv(camtoworlds)`

## 한 줄 요약

`rasterization()`은 **월드 좌표를 카메라 좌표로 바꾸는 행렬**(world→cam)을 원한다.
그런데 데이터셋이 들고 있는 건 그 **반대 방향**인 `camtoworlds`(cam→world)다.
방향이 반대인 변환은 곧 **역행렬**이므로, 코드는 이렇게 쓴다.

```python
return rasterization(
    ...,
    viewmats=torch.linalg.inv(camtoworlds),   # world→cam
    Ks=Ks, width=width, height=height, ...
)
```

(`fm/training/.fm/assets/training_walkthrough.py`의 `rasterize_splats()`, 그리고 원본
`examples/simple_trainer.py:728`에서는 같은 뜻의 `torch.linalg.inv_ex(camtoworlds).inverse`)

---

## 1. 고교 개념에서 출발: "좌표를 바꾸는 것 = 행렬을 곱하는 것"

기하와 벡터에서 배운 회전변환을 떠올려 보자. 2차원에서 점을 각 $\theta$ 만큼 회전시키는 변환은

$$
R(\theta)=\begin{pmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{pmatrix},
\qquad
\begin{pmatrix}x'\\y'\end{pmatrix}=R(\theta)\begin{pmatrix}x\\y\end{pmatrix}
$$

3차원에서도 똑같다. 회전은 $3\times3$ 행렬 $R$ 하나로 표현된다.
문제는 **평행이동**이다. "$\mathbf{x}$를 $\mathbf{t}$만큼 옮겨라"는 $\mathbf{x}+\mathbf{t}$인데,
이건 행렬 곱셈만으로는 쓸 수 없다(원점이 원점으로 가지 않으므로).

### 트릭: 차원을 하나 늘린다 (동차좌표, homogeneous coordinates)

점 $(x,y,z)$ 뒤에 숫자 $1$을 붙여 4차원 벡터 $(x,y,z,1)^\top$로 쓰면, 회전과 평행이동을
**한 개의 $4\times4$ 행렬**로 묶을 수 있다.

$$
\underbrace{\begin{pmatrix} R & \mathbf{t}\\ \mathbf{0}^\top & 1\end{pmatrix}}_{4\times4}
\begin{pmatrix}\mathbf{x}\\ 1\end{pmatrix}
=\begin{pmatrix} R\mathbf{x}+\mathbf{t}\\ 1\end{pmatrix}
$$

왜 되는지는 직접 성분을 곱해 보면 바로 확인된다. 마지막 행 $(0,0,0,1)$ 덕분에 네 번째 성분은
계속 $1$로 유지되고, 위쪽 세 성분이 정확히 "회전 후 평행이동"이 된다.
이것이 `camtoworlds`가 `[N, 4, 4]` 모양인 이유다. $N$은 학습 이미지(=카메라) 개수.

---

## 2. 두 행렬은 서로 무엇이 다른가

3DGS에는 좌표계가 두 개 있다.

| 좌표계 | 뜻 |
|---|---|
| **월드(world)** | 장면 전체가 놓인 고정 좌표계. Gaussian의 `means`가 여기 산다. |
| **카메라(camera)** | 카메라 렌즈를 원점으로 삼는 좌표계. OpenCV/COLMAP 규약이라 $+z$가 "카메라가 보는 앞쪽". |

- $M_{c\to w}$ = `camtoworld` : 카메라 좌표의 점을 월드 좌표로 옮긴다. → **"카메라가 월드의 어디에 어떤 자세로 놓였는가(=포즈)"**
- $M_{w\to c}$ = `viewmat` : 월드 좌표의 점을 카메라 좌표로 옮긴다. → **"카메라 눈으로 볼 때 이 점은 어디에 있는가"**

두 변환은 서로를 되돌리므로 $M_{w\to c}M_{c\to w}=I$, 즉

$$
\boxed{\;M_{w\to c}=M_{c\to w}^{-1}\;}
$$

역행렬은 고교 과정에서 $2\times2$로 배운 그 개념 그대로다.
$A^{-1}$은 "$A$가 한 일을 정확히 되돌리는 행렬"이고, $AA^{-1}=A^{-1}A=I$.
크기가 $4\times4$로 커졌을 뿐, 역할은 같다. PyTorch에서는
`torch.linalg.inv`가 이걸 계산해 준다(배치 차원 `[N,4,4]`도 한 번에 처리).

### 직관적으로 왜 "포즈의 역"이 "보는 변환"인가

카메라가 원점에서 $z=5$ 위치로 뒤로 물러났다고 하자. 그러면 카메라 눈에서 보면
**세상이 앞으로 5만큼 다가온 것**과 같다. 카메라를 오른쪽으로 돌리면, 카메라 눈에서는
**세상이 왼쪽으로 돌아간 것**처럼 보인다. 즉 카메라를 움직이는 변환과, 그 시점에서 세상을
보는 변환은 정확히 서로의 반대 — 그래서 역행렬이다.

---

## 3. 손으로 검산해 보기 (그리고 강체변환의 역행렬 공식)

일반 $4\times4$ 역행렬은 계산이 번거롭지만, 여기 나오는 행렬은 **강체변환**(회전+평행이동)이라
아주 깔끔한 공식이 있다. $R$이 회전행렬이면 $R^{-1}=R^\top$이기 때문이다
(회전행렬의 열벡터들이 서로 수직인 단위벡터이므로 $R^\top R = I$ — 내적으로 확인 가능).

$$
\begin{pmatrix} R & \mathbf{t}\\ \mathbf{0}^\top & 1\end{pmatrix}^{-1}
=\begin{pmatrix} R^\top & -R^\top\mathbf{t}\\ \mathbf{0}^\top & 1\end{pmatrix}
$$

**검산**: 두 행렬을 곱하면 왼쪽 위 블록은 $RR^\top=I$, 오른쪽 위 블록은
$R(-R^\top\mathbf{t})+\mathbf{t}=-\mathbf{t}+\mathbf{t}=\mathbf{0}$. 정말 $I$가 된다.

**구체적 예**: 카메라가 회전 없이($R=I$) 월드의 $(0,0,5)$에 있다면
$M_{c\to w}$의 평행이동 성분은 $\mathbf{t}=(0,0,5)$이고,

$$
M_{w\to c}=\begin{pmatrix} I & -(0,0,5)^\top\\ \mathbf{0}^\top & 1\end{pmatrix}
$$

월드 원점 $(0,0,0)$을 넣으면 카메라 좌표 $(0,0,-5)$. $z<0$ 이니 "카메라 뒤"라서 안 보인다 —
$+z$가 앞쪽인 규약과 잘 맞는 결과다. 여기서 알 수 있는 유용한 사실:

> `camtoworlds[:, :3, 3]` 은 **월드에서의 카메라 위치**다.
> 워크스루가 카메라 궤적을 3D로 그릴 때 `cam_pos = parser.camtoworlds[:, :3, 3]`을
> 쓰는 것이 바로 이 때문이고, SH 색 계산의 시선 방향도
> `dirs = means - camtoworlds[:, :3, 3]` 로 얻는다(`simple_trainer.py:677`).
> 반대로 `viewmats[:, :3, 3]`은 카메라 위치가 **아니다**($-R^\top\mathbf{t}$ 를 거쳐야 위치가 된다).

---

## 4. 넘겨준 `viewmats`가 렌더러 안에서 실제로 하는 일

`gsplat/cuda/_torch_impl.py`의 `_world_to_cam()`이 참조 구현이다. `viewmats`를 회전과
평행이동으로 쪼개서 두 가지에 쓴다.

```python
R = viewmats[..., :3, :3]        # [C,3,3]
t = viewmats[..., :3, 3]         # [C,3]
means_c  = R @ means + t         # 중심점 변환
covars_c = R @ covars @ R.T      # 공분산(타원체 모양) 변환
```

- **중심점**: $\boldsymbol{\mu}_c = R\boldsymbol{\mu}_w+\mathbf{t}$ — 위에서 본 동차좌표 곱 그대로.
- **공분산**: Gaussian은 점이 아니라 "퍼진 덩어리"이고 그 모양은 공분산행렬 $\Sigma$가 정한다.
  확률과 통계에서 확률변수 $X$에 상수를 곱하면 분산이 $\mathrm{Var}(aX)=a^2\mathrm{Var}(X)$로
  변하는 것을 배웠는데, 벡터 버전이 바로 $\Sigma_c = R\,\Sigma_w R^\top$ 이다
  ($a^2$ 대신 $R(\cdot)R^\top$).

이렇게 카메라 좌표로 옮긴 다음에야 내부 파라미터 $K$로 화면에 투영하고
($u=f_x x/z+c_x$ 꼴), near/far 평면 컬링과 타일 래스터화가 진행된다.
즉 `viewmats`가 방향을 거꾸로 주면 장면이 전부 카메라 뒤로 가서 **아무것도 렌더되지 않는다**
— 초기 학습에서 검은 화면이 나오면 가장 먼저 의심할 지점이다.

---

## 5. 그런데 왜 처음부터 world→cam을 저장하지 않았나?

재미있는 사실: COLMAP이 원래 주는 값은 world→cam이다. 데이터 로더가 그걸 한 번 뒤집어
`camtoworlds`로 저장한다(`examples/datasets/colmap.py:190`, `camtoworlds = np.linalg.inv(w2c_mats)`).
그리고 학습 루프에서 다시 뒤집어 `viewmats`로 만든다. 왕복하는 이유는 c2w 형태가
**사람과 데이터 처리에 훨씬 편하기** 때문이다.

- 카메라 위치를 바로 읽을 수 있다(§3) → 씬 정규화(`similarity_from_cameras`), `scene_scale` 계산,
  카메라 궤적 보간·타원 궤도 생성 등이 모두 c2w 기준으로 돌아간다.
- 포즈 최적화(`pose_adjust`)처럼 카메라를 "움직이는" 조작도 c2w에서 자연스럽다.

반면 렌더러는 오직 "월드 점 → 카메라 좌표" 계산만 하므로 w2c를 원한다. 그래서 경계면에서
한 번 `inv`를 넣어 주는 것이다.

---

## 6. 실수하기 쉬운 지점

1. **전치와 역행렬을 혼동**: $R^{-1}=R^\top$은 $3\times3$ **회전 블록에만** 성립한다.
   $4\times4$ 전체를 전치하면 평행이동 성분이 엉뚱한 곳으로 간다. 굳이 손으로 만들려면
   §3의 $\begin{pmatrix}R^\top & -R^\top\mathbf{t}\end{pmatrix}$ 공식을 쓸 것.
2. **배치 차원**: `camtoworlds`는 `[C,4,4]`(단일 카메라도 `data0["camtoworld"][None]`으로
   앞에 축을 하나 붙인다). `torch.linalg.inv`는 마지막 두 축에만 작용하므로 그대로 쓰면 된다.
3. **`inv` vs `inv_ex`**: `simple_trainer.py`가 쓰는 `torch.linalg.inv_ex(...).inverse`는
   특이행렬일 때 예외를 던지지 않고 `info`로 알려 주는 버전이라 GPU 동기화를 피할 수 있어 조금 빠르다.
   결과값의 의미는 `inv`와 같다.
4. **매 스텝 재계산**: 포즈를 최적화하면 `camtoworlds`가 매 이터레이션 바뀌므로 역행렬도 매번
   다시 계산해야 하고, 이 `inv`는 **미분 가능해야** 한다(그래서 numpy가 아니라 `torch.linalg.inv`).

---

## 암기용 압축

> 데이터셋은 **cam→world**(포즈)를 준다. 래스터라이저는 **world→cam**(뷰)을 원한다.
> 반대 방향 = 역행렬 ⇒ `viewmats = torch.linalg.inv(camtoworlds)`.
> 부산물로 `camtoworlds[:, :3, 3]` = 카메라 월드 위치.
