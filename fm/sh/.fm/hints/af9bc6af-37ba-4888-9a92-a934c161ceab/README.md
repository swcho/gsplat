# 3DGS에서 SH 평가는 얼마나 자주, 어떤 방향으로 수행되는가

> **답**: **카메라마다·Gaussian마다 한 번씩** 수행된다. 방향 $\mathbf d$는 카메라 광학 중심 $\mathbf o_{\text{cam}}$에서
> Gaussian 중심 $\boldsymbol\mu_i$를 바라보는 벡터 $\boldsymbol\mu_i-\mathbf o_{\text{cam}}$을 단위 길이로 정규화한 것이다.

---

## 1. 왜 "픽셀마다"가 아니라 "Gaussian마다"인가

SH 평가란 계수 $\mathbf c_{i,k}$와 방향 $\mathbf d$가 주어졌을 때 급수를 실제로 더하는 일이다.

$$
\mathbf c_i(\mathbf d) = \max\!\Big(0,\ \sum_{k=0}^{15} \mathbf c_{i,k}\,Y_k(\mathbf d) + 0.5\Big)
$$

NeRF는 **광선(=픽셀)마다** 샘플 위치와 시점 방향을 MLP에 넣어 색을 얻는다. 반면 3DGS는 구조가 다르다.

1. 각 Gaussian $i$에 대해 **색 하나** $\mathbf c_i$를 결정한다.
2. 래스터라이저는 그 Gaussian이 덮는 픽셀들에 **같은 색** $\mathbf c_i$를 불투명도 $\alpha_i$와 2D 가우시안 가중치로 알파 블렌딩한다.

즉 색은 Gaussian 단위의 속성이고, 픽셀 단계에서는 이미 정해진 색을 섞기만 한다. 그래서 SH를 픽셀마다 평가할 자리가 없고,
"이 카메라에서 이 Gaussian은 어떤 색인가"를 Gaussian마다 한 번 계산하면 끝이다.

이때 쓰는 근사가 하나 있다. 실제로는 Gaussian이 덮는 각 픽셀마다 광선 방향이 조금씩 다르지만, 3DGS는
**한 Gaussian 안에서는 시점 방향이 일정하다**고 보고 **Gaussian 중심을 향하는 방향 하나로 대표**한다.
Gaussian이 화면에서 작을수록 이 근사는 정확하고, 화면을 크게 덮는 Gaussian(가까이 있거나 스케일이 큰 것)일수록
가장자리 픽셀의 실제 방향과 중심 방향의 차이가 커져 근사 오차가 늘어난다. 실제 장면에서는 Gaussian이 대부분 작고
SH 3차가 표현하는 색 변화가 부드럽기 때문에 이 오차는 눈에 잘 띄지 않는다.

## 2. 연산 규모 — 많아 보이지만 GPU에서는 수 ms

한 번의 평가는 두 단계다.

1. 방향 $\mathbf d$에서 기저값 $Y_0(\mathbf d),\dots,Y_{15}(\mathbf d)$ 16개 계산 (다항식 몇 개)
2. 채널(R, G, B)마다 계수 16개와 내적 → 곱셈-덧셈(FMA) 약 48회

규모를 어림해 보면:

| 항목 | 값 |
|---|---|
| Gaussian 수 | 300만 개 |
| 카메라 수 | 1대 (한 프레임) |
| Gaussian당 연산 | 약 48 FMA + 기저 계산 |
| 프레임당 총 연산 | $3\times10^6 \times 48 \approx 1.4\times10^8$ FMA → 수억 연산 |

수억 연산이지만 Gaussian마다 완전히 독립적이라 GPU 스레드 하나가 Gaussian 하나(또는 채널 하나)를 맡으면 되고,
분기·메모리 접근도 단순해서 실제로는 **수 ms 이내**에 끝난다. 뉴럴 네트워크 없이 다항식 덧셈·곱셈만 쓰는 것이
3DGS가 실시간 렌더링을 달성하는 핵심 이유 중 하나다. gsplat의 CUDA 커널은 이 hot path(K=16, D=3)를 위한
전용 커널(`spherical_harmonics_fwd_kernel_k16_3channel`)까지 두고 있는데, 스레드 하나가 Gaussian 하나의 RGB 세 채널을
한꺼번에 처리하고 계수를 wide load로 읽어 메모리 대역폭을 아낀다.

학습 시에는 카메라 1대(또는 소수 배치)로 한 스텝을 돌리므로 매 스텝 $N$번, 여러 카메라를 배치로 렌더할 때는 $C\times N$번 평가된다.

## 3. 방향 $\mathbf d$의 정의

### 3.1 카메라 → Gaussian 중심 벡터

$$
\mathbf d_i = \frac{\boldsymbol\mu_i - \mathbf o_{\text{cam}}}{\|\boldsymbol\mu_i - \mathbf o_{\text{cam}}\|}
$$

- $\mathbf o_{\text{cam}}$: 카메라 광학 중심(투영 중심)의 **world 좌표**
- $\boldsymbol\mu_i$: Gaussian $i$의 중심(mean)의 **world 좌표**

부호 방향에 주의한다. 카메라에서 물체로 **나가는** 방향이며, "물체에서 카메라를 보는" 방향의 반대다. 어느 쪽 부호를 쓰든
일관되기만 하면 학습은 되지만(계수가 그에 맞게 학습됨), 학습된 모델을 다른 구현으로 렌더링할 때 부호가 어긋나면
시점 의존 색이 전부 뒤집히므로 관례를 지키는 것이 중요하다. Inria 원본과 gsplat 모두 이 부호를 쓴다.

### 3.2 왜 world 좌표계에서 계산하는가

SH 계수 $\mathbf c_{i,k}$는 Gaussian에 붙어 저장되는 학습 파라미터이고, 카메라가 어디 있든 고정된 값이다.
따라서 이 계수가 나타내는 "방향별 색 함수"의 좌표계도 카메라와 무관하게 고정되어야 한다. 방향을 camera 좌표계에서
계산하면 카메라가 회전할 때마다 같은 물리적 방향이 다른 $(x,y,z)$로 표현되어 계수가 의미를 잃는다.
world 좌표계에서 방향을 만들면 계수도 world 기준으로 학습되고, 어떤 카메라로 보더라도 일관된 함수가 된다.

### 3.3 왜 정규화가 필요한가

SH 기저 $Y_\ell^m$은 **단위구 $S^2$ 위의 함수**다. $\mathbf d=(x,y,z)$를 넣어 다항식으로 계산하는 `sh_bases`는
$x^2+y^2+z^2=1$을 전제로 유도된 식이다. 정규화하지 않은 $\boldsymbol\mu_i-\mathbf o_{\text{cam}}$을 그대로 넣으면
거리에 따라 기저값의 크기가 제멋대로 커져서(1차 항은 거리에 비례, 2차 항은 거리의 제곱에 비례) 색이 카메라와의
거리에 따라 폭주한다. 시점 **방향**만 중요하고 거리는 무관해야 하므로 반드시 단위 벡터로 만든다.

### 3.4 카메라 위치를 viewmat에서 복원하기

gsplat은 카메라를 world→camera 행렬 `viewmat` $=\begin{bmatrix}R & \mathbf t\\ 0 & 1\end{bmatrix}$로 받는다.
이 행렬은 world 점 $\mathbf p$를 $\mathbf p_{\text{cam}} = R\,\mathbf p + \mathbf t$로 보낸다.
카메라 광학 중심은 camera 좌표계의 원점이므로 $R\,\mathbf o_{\text{cam}} + \mathbf t = \mathbf 0$에서

$$
\mathbf o_{\text{cam}} = -R^{-1}\mathbf t = -R^\top\mathbf t \qquad (R\text{이 회전 행렬이므로 } R^{-1}=R^\top)
$$

이 식은 $R$이 정규직교(orthonormal)일 때만 정확하다. gsplat 문서도 viewmat에 스케일이나 shear가 섞이거나,
카메라 행렬을 raw로 최적화해서 정규직교성이 깨지면 결과가 근사가 된다고 명시한다.

노트북(`sh_walkthrough.py` 4절)의 재현 코드:

```python
def view_dirs(means, viewmat):
    """3DGS의 시점 방향: 카메라 위치 −Rᵀt 에서 각 Gaussian 중심을 향하는 벡터."""
    R, t = viewmat[:3, :3], viewmat[:3, 3]
    cam_pos = -R.T @ t
    return means - cam_pos          # 정규화는 sh_eval 안에서 F.normalize로
```

예: 카메라가 world $(0,0,-5)$에서 $+z$를 보면 $R=I$, $\mathbf t=(0,0,5)$이고 $-R^\top\mathbf t=(0,0,-5)$로 카메라 위치가 복원된다.

## 4. gsplat 코드에서의 모습

### 4.1 `dirs` 텐서의 shape [C, N, 3]과 배치 카메라

카메라 $C$대를 한 번에 렌더하면 방향은 (카메라, Gaussian) 쌍마다 하나씩 필요하므로 `[C, N, 3]`이 된다.
PyTorch로 쓰면 브로드캐스팅으로 이렇게 만들어진다.

```python
camtoworlds = torch.linalg.inv(viewmats)              # [C, 4, 4]
dirs = means[None] - camtoworlds[:, None, :3, 3]      # [1, N, 3] - [C, 1, 3] → [C, N, 3]
colors = _spherical_harmonics(sh_degree, dirs, coeffs)  # coeffs [N, K, 3] → [C, N, 3]
```

`camtoworlds[:, :3, 3]`은 camera→world 행렬의 평행이동 열, 곧 world 좌표의 카메라 위치 $\mathbf o_{\text{cam}}$이며
$-R^\top\mathbf t$와 같은 값이다. 참조 구현 `gsplat/cuda/_torch_impl.py`의 `_spherical_harmonics`는 `dirs`를
`[..., N, 3]`으로 받아 `F.normalize(dirs, dim=-1)` 후 기저 × 계수 합을 계산한다. 계수는 `[N, K, 3]`으로 카메라 축이 없다는
점에 주목하자. **계수는 Gaussian마다 하나이고, 카메라마다 달라지는 것은 방향뿐**이다.

현재 gsplat의 CUDA 경로(`gsplat/cuda/_wrapper.py` `spherical_harmonics(degrees_to_use, means, viewmats, coeffs, masks)`)는
파이썬에서 `dirs`를 만들지 않고 `means [..., N, 3]`과 `viewmats [..., C, 4, 4]`를 그대로 넘긴다. 커널 안에서 스레드마다
`view_direction_from_world_to_camera(mean, viewmat)`이 `mean + R^T t` (= $\boldsymbol\mu_i - (-R^\top\mathbf t)$)를 계산하고,
`sh_coeffs_to_color_fast`가 `rsqrtf`로 정규화한 뒤 계수와 내적한다. 즉 `[C, N, 3]` 방향 텐서를 메모리에 만들지 않고
스레드 인덱스 `(batch_id, camera_id, gaussian_id)`로 즉석에서 방향을 계산한다. 결과 shape는 `[..., C, N, 3]`으로 같다.
롤링 셔터 카메라(`viewmats_rs`)가 주어지면 두 끝점의 카메라 위치를 평균해 사용한다.

### 4.2 컬링 마스크 — 화면 밖 Gaussian은 평가 생략

`gsplat/rendering.py`의 `_maybe_evaluate_sh`:

```python
masks = (radii > 0).all(dim=-1)  # [..., C, N]
features = spherical_harmonics(sh_degree, means, viewmats, features, masks=masks)  # [..., C, N, D]
if clamp:
    features = torch.clamp_min(features + 0.5, 0.0)   # Inria 구현과 동일
```

투영 단계에서 카메라 절두체 밖이거나 너무 작아 잘려 나간 Gaussian은 `radii`가 0으로 기록된다. 그 (카메라, Gaussian) 쌍은
어차피 픽셀에 기여하지 않으므로 마스크를 넘겨 SH 평가를 건너뛴다. CUDA 커널은 `if (masks != nullptr && !masks[output_id]) return;`
로 해당 스레드를 바로 종료한다. 이 마스크가 `[C, N]`인 것 자체가 "카메라마다·Gaussian마다 한 번"이라는 평가 단위를 그대로 보여 준다.
같은 Gaussian이라도 카메라 A에서는 보이고 카메라 B에서는 컬링될 수 있기 때문이다.

packed 모드에서는 `[C, N]` 격자 대신 보이는 쌍만 `nnz`개 모아 `camera_ids`, `gaussian_ids`로 넘기고 결과도 `[nnz, D]`가 된다.
이 경우에도 단위는 여전히 (카메라, Gaussian) 쌍이다.

### 4.3 이후 처리

평가 결과 `[C, N, 3]`에 `+0.5`를 더하고 `clamp_min(0)`을 적용한 뒤 래스터라이저 `rasterize_to_pixels`에 넘긴다.
래스터라이저는 이 색을 더 이상 방향에 따라 바꾸지 않고 알파 블렌딩만 한다. 학습 초기에 `sh_degree_to_use`를 낮게 두면
(`min(step // sh_degree_interval, sh_degree)`) 고차 기저를 0으로 두고 계산해 사실상 DC만 평가된다.

## 5. 시점 방향 vs 표면 법선 — SH 색은 법선을 쓰지 않는다

전통적인 실시간 렌더링에서 SH는 주로 **조명(irradiance)**을 표현하고, 표면 한 점의 밝기는 그 점의 **법선 $\mathbf n$** 방향에서
조명 SH를 평가해 얻는다(Ramamoorthi & Hanrahan 2001). 그래서 "SH를 평가하는 방향 = 법선"이라는 인상을 가질 수 있다.

3DGS는 다르다.

| | 전통 SH 조명 | 3DGS의 SH 색 |
|---|---|---|
| SH가 표현하는 것 | 방향별 입사 광량 | Gaussian이 **보이는 방향에 따라 내는 색**(outgoing radiance) |
| 평가 방향 | 표면 법선 $\mathbf n$ | 시점 방향 $\mathbf d = \text{normalize}(\boldsymbol\mu_i - \mathbf o_{\text{cam}})$ |
| 법선 필요 여부 | 필수 | **불필요** |

3DGS의 Gaussian은 반투명한 덩어리(volumetric primitive)일 뿐 **표면이 없고, 따라서 법선도 없다**. 조명과 재질을 분리해 계산하는 대신,
"이 방향에서 보면 이 색"이라는 관측 결과 자체를 SH에 담는다(NeRF가 시점 방향을 MLP 입력으로 받는 것과 같은 역할). 하이라이트, 프레넬,
반투명 같은 시점 의존 효과가 모두 이 함수 안에 뭉쳐 들어간다.

이것이 3DGS에서 relighting이 곧바로 되지 않는 이유이기도 하다. 조명이 색에 baked-in 되어 있고 법선·재질이 분리되어 있지 않다.
법선을 도입해 조명을 분리하려는 후속 연구(예: GaussianShader, Relightable 3D Gaussian, 2DGS 계열)는 별도의 법선 파라미터나
가장 짧은 축을 법선으로 쓰는 등의 장치를 추가로 둔다.

---

## 요약

- **빈도**: (카메라, Gaussian) 쌍마다 1회. 픽셀마다가 아닌 이유는 3DGS가 Gaussian 하나에 색 하나를 주고 그 색을 픽셀에 알파 블렌딩하기 때문.
  한 Gaussian 안에서는 시점 방향을 중심 방향 하나로 근사한다.
- **규모**: Gaussian 300만 × 48 FMA ≈ 수억 연산/프레임이지만 완전 병렬이라 GPU에서 수 ms.
- **방향**: $\mathbf d = \text{normalize}(\boldsymbol\mu_i - \mathbf o_{\text{cam}})$, world 좌표계. 카메라 위치는 viewmat에서 $-R^\top\mathbf t$로 복원.
  SH는 단위구 위 함수이므로 정규화 필수.
- **gsplat**: 방향(개념적으로 `[C, N, 3]`)은 CUDA 커널 안에서 즉석 계산, 계수는 `[N, K, 3]`으로 카메라 축 없음,
  `masks = (radii > 0)` `[C, N]`으로 컬링된 쌍은 평가 생략, 결과 `[C, N, 3]`에 `+0.5`, `clamp_min(0)`.
- **법선 아님**: 3DGS에는 표면 법선이 없다. SH는 시점 방향으로 평가하는 outgoing radiance 함수다.
