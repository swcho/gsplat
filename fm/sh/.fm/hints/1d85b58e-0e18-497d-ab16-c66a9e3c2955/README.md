# 3DGS에서 Gaussian $i$의 시점 의존 색 공식

## 질문

3DGS에서 Gaussian $i$의 시점 의존 색 공식은?

## 답

$$
\mathbf c_i(\mathbf d)=\max\!\Big(0,\ \sum_{k=0}^{15}\mathbf c_{i,k}\,Y_k(\mathbf d)+0.5\Big),
\qquad
\mathbf d=\frac{\boldsymbol\mu_i-\mathbf o_{\text{cam}}}{\|\boldsymbol\mu_i-\mathbf o_{\text{cam}}\|}
$$

$\mathbf d$는 카메라 위치 $\mathbf o_{\text{cam}}$에서 Gaussian 중심 $\boldsymbol\mu_i$를 향하는 단위 방향이다.

---

## 1. 왜 색이 "방향의 함수"인가

실제 물체는 보는 각도에 따라 색이 달라진다(하이라이트, 프레넬 반사, 반투명). Gaussian 하나에 RGB 값 하나만 두면 이런 현상을 표현할 수 없다. 그래서 3DGS는 Gaussian마다 **구면 위 함수** $\mathbf c_i:\ S^2\to\mathbb R^3$를 저장한다. "구면 위 함수"를 소수의 숫자로 압축하는 표준 도구가 Spherical Harmonics(SH)이고, 3DGS는 3차까지 총 $(3+1)^2 = 16$개의 기저를 쓴다.

핵심은 **뉴럴 네트워크가 없다**는 점이다. 색 하나를 얻는 데 필요한 연산은 채널당 16번의 곱셈-덧셈이라, 수백만 개 Gaussian을 매 프레임 실시간으로 처리할 수 있다(NeRF 계열이 MLP를 픽셀마다 수백 번 호출하는 것과 대비된다).

## 2. 공식의 각 부분

### 2.1 기저 $Y_k(\mathbf d)$ — 방향만으로 결정되는 16개의 숫자

$Y_k$는 실수형 Spherical Harmonics를 $k=\ell^2+\ell+m$ 순서로 일렬로 늘어놓은 것이다($\ell$ = 차수, $-\ell\le m\le\ell$).

| $k$ | $(\ell, m)$ | $Y_k(x,y,z)$ | 성질 |
|---|---|---|---|
| 0 | (0, 0) | $\tfrac{1}{2\sqrt\pi}\approx 0.2821$ | **상수** — 방향 무관 |
| 1, 2, 3 | (1, −1), (1, 0), (1, 1) | $-\sqrt{\tfrac{3}{4\pi}}\,y,\ \sqrt{\tfrac{3}{4\pi}}\,z,\ -\sqrt{\tfrac{3}{4\pi}}\,x$ | 1차(선형) — 앞/뒤 밝기 차 |
| 4 … 8 | $\ell=2$ | $xy,\ yz,\ 3z^2-1,\ xz,\ x^2-y^2$ 에 상수 곱 | 2차 — 더 잦은 변동 |
| 9 … 15 | $\ell=3$ | $x,y,z$의 3차 동차 다항식 | 3차 — 가장 세부 |

즉 $\ell$차 SH는 **$x,y,z$의 $\ell$차 다항식을 단위구 $\|\mathbf d\|=1$에 제한한 것**이다. 여기서 $x,y,z$는 방향 $\mathbf d$의 성분이므로, $\mathbf d$가 단위벡터라는 조건이 붙는다(그렇지 않으면 다항식 값이 길이에 따라 멋대로 커진다). 기저들은 구면 적분에 대해 정규직교하며, 부호는 Condon–Shortley 규약(3DGS 원본과 gsplat 모두 동일)이다.

$Y_k$는 학습 파라미터가 아니다. 방향 $\mathbf d$가 정해지면 값이 확정되는 고정 함수이고, 모든 Gaussian이 공유한다.

### 2.2 계수 $\mathbf c_{i,k}\in\mathbb R^3$ — Gaussian이 실제로 "기억하는" 것

계수는 Gaussian $i$마다 $16\times 3 = 48$개의 실수이며, 학습으로 갱신되는 파라미터다. 채널 R, G, B 각각이 자기 몫의 16개 계수를 가진다. 합 $\sum_k \mathbf c_{i,k}Y_k(\mathbf d)$는 행렬 형태로 쓰면

$$
\underbrace{[\,Y_0(\mathbf d)\ \cdots\ Y_{15}(\mathbf d)\,]}_{1\times16}\;
\underbrace{\begin{bmatrix}\mathbf c_{i,0}^\top\\ \vdots\\ \mathbf c_{i,15}^\top\end{bmatrix}}_{16\times3}
\;=\;\underbrace{[\,r\ \ g\ \ b\,]}_{1\times 3}
$$

이므로 "16차원 방향-특징 벡터와 계수 행렬의 곱"이다.

- **$k=0$ (DC 계수)**: $Y_0$이 상수이므로 $\mathbf c_{i,0}Y_0$은 방향과 무관한 **기본색**이다. 나머지 $\ell\ge1$ 항은 구면 평균이 0인 "변동"만 담는다. asset 노트북 3장에서 DC 항만으로 복원한 값이 함수의 구면 평균과 정확히 일치함을 수치로 확인한다.
- **$k\ge1$ (고차 계수)**: 시점 의존성. gsplat의 `examples/simple_trainer.py`는 파라미터를 `sh0` (`[N,1,3]`)와 `shN` (`[N,15,3]`)으로 나누고 `shN_lr = sh0_lr / 20`으로 둔다 — 기본색이 먼저 잡히고 시점 의존성은 천천히 배우라는 뜻이다. 또 `sh_degree_to_use = min(step // sh_degree_interval, sh_degree)`로 1000스텝마다 한 차수씩 활성화한다.

### 2.3 $+0.5$ 오프셋 — "계수가 전부 0이면 중간 회색"

계수를 모두 0으로 두면 SH 합은 0이 되어 색이 검정이 된다. 0.5를 더하면 **계수가 0일 때의 기준점이 중간 회색 $(0.5,0.5,0.5)$**이 되어, 계수가 양·음 어느 쪽으로도 대칭적으로 움직일 수 있다. 이 오프셋이 초기화 공식 `sh0 = (rgb − 0.5) / C0`를 만든다. 고차 계수가 0인 상태에서

$$
\mathbf c(\mathbf d)=c_0\,Y_0^0+0.5=\mathbf{rgb}
\quad\Longrightarrow\quad
c_0=\frac{\mathbf{rgb}-0.5}{Y_0^0}=\frac{\mathbf{rgb}-0.5}{C_0},\qquad C_0=\tfrac{1}{2\sqrt\pi}
$$

이므로 학습 시작 시 각 Gaussian이 SfM 포인트 색을 그대로 재현한다(asset 노트북 3장 코드 셀에서 검증).

수학적으로 $+0.5$는 DC 계수에 흡수될 수 있다($0.5 = (0.5/C_0)\,Y_0^0$). 그럼에도 별도 상수로 두는 이유는 순전히 **최적화 편의**(초깃값이 0 근방에서 대칭이 되도록)이며, Inria 원본 구현과의 호환성을 유지하기 위해서다. 노트북 4.2에서 학습된 계수를 정답과 비교할 때 `coeffs_gd[0] += 0.5 / C0`로 이 오프셋을 DC로 흡수하는 장면이 바로 그 동치성을 보여준다.

### 2.4 $\max(0,\cdot)$ 클램프 — 목적과 부작용

**목적.** SH는 직교 다항식의 합이므로 계수에 따라 얼마든지 **음수**가 나올 수 있다. 노트북 1.3의 환경 조명 실험에서 날카로운 태양 봉우리 주변에 음의 물결(ringing)이 생기는 것이 그 예다. 음의 색은 물리적 의미가 없고, 뒤따르는 알파 블렌딩에서 다른 Gaussian의 기여를 "빼는" 이상한 동작을 낳는다. `max(0, ·)`은 이를 0으로 잘라 색을 음이 아닌 값으로 보장한다. (상한 1은 자르지 않는다 — HDR스러운 밝은 값은 블렌딩 후 이미지 단계에서 처리된다.)

**부작용.**
1. **기울기 소실**: 클램프된 영역($\sum + 0.5 < 0$)에서는 미분이 0이므로, 그 방향에서 관측되는 계수에는 학습 신호가 전달되지 않는다. 어떤 Gaussian이 특정 시점에서 "검정으로 잘린" 상태에 빠지면 그 시점에서는 스스로 빠져나오기 어렵다.
2. **표현력 제한**: 함수를 ReLU로 자른 형태이므로 SH의 선형·정규직교 성질(계수 = 사영 적분)이 깨진다. 계수를 해석할 때 "$\mathbf c_i(\mathbf d)$가 정확히 SH 급수"라고 볼 수 없다.
3. **비대칭 경계**: 아래쪽만 자르므로 밝은 하이라이트 쪽으로는 자유롭지만 어두운 쪽은 0에서 포화된다.

gsplat은 색(`colors`)에는 클램프를 적용하지만, 함께 래스터화되는 `extra_signals`(깊이·법선·임의 특징 등)에는 적용하지 않는다 — 부호가 의미를 갖는 신호이기 때문이다(아래 5장 코드 참조).

## 3. 방향 $\mathbf d$: 왜 "카메라 → Gaussian"이고, 카메라 위치는 어떻게 얻나

### 3.1 방향의 정의

$$
\mathbf d=\frac{\boldsymbol\mu_i-\mathbf o_{\text{cam}}}{\|\boldsymbol\mu_i-\mathbf o_{\text{cam}}\|}
$$

분자 $\boldsymbol\mu_i-\mathbf o_{\text{cam}}$은 카메라 위치에서 Gaussian 중심을 향하는 벡터, 분모는 그 길이다. 즉 **"카메라가 이 Gaussian을 어느 방향에서 바라보고 있는가"**를 단위벡터로 표현한 것이다.

주의할 점 두 가지:
- **월드 좌표계**다. 방향은 카메라 좌표계로 회전하지 않는다. 그래야 카메라가 돌아가더라도 같은 물리 방향에서 보면 같은 $Y_k$ 값이 나오고, 계수 $\mathbf c_{i,k}$가 카메라 자세와 무관한 "물체의 성질"로 학습된다.
- **Gaussian마다, 카메라마다 다르다.** 같은 카메라라도 Gaussian 위치가 다르면 $\mathbf d$가 다르다(핀홀 카메라의 시선은 화면 위치마다 다르다). 따라서 SH 평가는 $C\times N$번 수행된다.

부호 규약(카메라→Gaussian인지 그 반대인지)은 사실 학습으로 흡수될 수 있지만(홀수 차수 $\ell=1,3$의 부호만 뒤집힌다), Inria 원본·gsplat·노트북이 모두 "카메라→Gaussian"을 쓰므로 계수를 서로 호환하려면 이 규약을 지켜야 한다.

### 3.2 카메라 위치 $\mathbf o_{\text{cam}}=-R^\top\mathbf t$

3DGS/gsplat이 받는 `viewmat`는 **world→camera** 변환 $[R\,|\,\mathbf t]$다. 월드 점 $\mathbf p_w$를 카메라 좌표로 옮기는 식은

$$
\mathbf p_c = R\,\mathbf p_w+\mathbf t .
$$

카메라 중심은 카메라 좌표에서 원점 $\mathbf p_c=\mathbf 0$이므로 $R\,\mathbf o_{\text{cam}}+\mathbf t=\mathbf 0$. $R$은 회전행렬이라 $R^{-1}=R^\top$이므로

$$
\mathbf o_{\text{cam}}=-R^\top\mathbf t .
$$

노트북의 예제: `viewmat[:3,3] = (0,0,5)`, $R=I$이면 카메라 위치는 $-(0,0,5)=(0,0,-5)$이다 — 즉 원점 앞쪽 5만큼 뒤에서 $+z$를 바라보는 카메라다. gsplat CUDA 커널은 이를 한 번에 계산한다(`gsplat/cuda/csrc/SphericalHarmonics.cuh`의 `camera_offset_from_world_to_camera`): `viewmat[0]*tx + viewmat[4]*ty + viewmat[8]*tz, …`는 row-major 4×4 행렬에서 $R^\top\mathbf t$를 구하는 식이고, `view_direction_from_world_to_camera`는 `mean + (R^T t)` $=\boldsymbol\mu_i-(-R^\top\mathbf t)=\boldsymbol\mu_i-\mathbf o_{\text{cam}}$를 반환한다. 정규화는 그 뒤 SH 평가 직전에 이루어진다(PyTorch 참조 구현에서는 `F.normalize(dirs, p=2, dim=-1)`).

## 4. 그 다음: 색이 래스터라이저에서 알파 블렌딩되는 흐름

SH 평가는 렌더링 파이프라인에서 **투영 후, 픽셀 합성 전**에 위치한다. gsplat `rasterization()`의 순서는 다음과 같다.

1. **투영**(`fully_fused_projection`): 3D Gaussian → 2D means, conics, depths, `radii`. 시야 밖 Gaussian은 `radii = 0`.
2. **타일 교차·정렬**(`isect_tiles`, `isect_offset_encode`): 각 Gaussian이 어느 16×16 타일에 겹치는지, 깊이 순 정렬.
3. **SH 평가** → 여기서 위의 색 공식이 적용된다. 출력은 `colors [..., C, N, 3]` — 카메라 $C$개 × Gaussian $N$개마다 **하나의 RGB**. 이후 단계에서는 이 Gaussian이 화면 어디에 그려지든 이 색 하나만 쓴다(Gaussian 내부에서는 색이 변하지 않는다).
4. **픽셀 래스터화**(`rasterize_to_pixels`): 픽셀 $\mathbf p$마다 겹치는 Gaussian을 깊이순 앞→뒤로 순회하며

$$
\alpha_i=o_i\,\exp\!\big(-\tfrac12(\mathbf p-\boldsymbol\mu_i')^\top\Sigma_i'^{-1}(\mathbf p-\boldsymbol\mu_i')\big),
\qquad
\mathbf C(\mathbf p)=\sum_i \mathbf c_i(\mathbf d)\,\alpha_i\prod_{j<i}(1-\alpha_j)
$$

   를 누적한다. 여기서 $\mathbf c_i(\mathbf d)$가 바로 3단계에서 나온 값이다. 남은 투과율 $\prod(1-\alpha_j)$가 임계값 이하가 되면 조기 종료한다.

이 흐름에서 클램프의 역할이 드러난다: 알파 블렌딩은 $\mathbf c_i\alpha_i T_i$의 **가중합**이므로, $\mathbf c_i$가 음수면 앞선 Gaussian의 색을 상쇄하게 된다. `max(0,·)`로 가중합의 각 항이 음이 아님을 보장한다.

역전파 때는 이 경로를 거꾸로 탄다: 픽셀 손실 → `rasterize_to_pixels`의 $\partial/\partial\mathbf c_i$ → SH 평가의 역전파(클램프 통과 후 $\partial\mathbf c_i/\partial\mathbf c_{i,k}=Y_k(\mathbf d)$, 그리고 $\partial\mathbf c_i/\partial\mathbf d$를 통해 `means`에도 기울기가 흐른다) → 계수 `sh0`, `shN` 갱신.

## 5. gsplat 코드에서 정확히 어디에서 일어나는가

### `gsplat/rendering.py` — `_maybe_evaluate_sh`

`rasterization()` 내부(`_rasterization`)에서 교차 정렬 직후 호출된다.

```python
# gsplat/rendering.py
def _maybe_evaluate_sh(
    sh_degree, features, means, radii, viewmats, batch_dims, C, N, clamp
):
    ...
    else:
        masks = (radii > 0).all(dim=-1)  # [..., C, N]
        features = spherical_harmonics(
            sh_degree, means, viewmats, features, masks=masks
        )  # [..., C, N, D]
        if clamp:
            # make it apple-to-apple with Inria's CUDA Backend.
            features = torch.clamp_min(features + 0.5, 0.0)
        else:
            features = features + 0.5
    return features
```

- `spherical_harmonics(...)`가 $\sum_k \mathbf c_{i,k}Y_k(\mathbf d)$까지(방향 계산·정규화·기저 평가·계수 곱 포함)를 CUDA 커널 하나로 처리한다.
- `torch.clamp_min(features + 0.5, 0.0)`이 정확히 $\max(0,\cdot+0.5)$다. 주석대로 Inria 원본 CUDA와 수치를 맞추기 위한 것이다.
- `masks = (radii > 0)`: 투영 단계에서 컬링된(화면 밖) Gaussian은 SH 평가를 건너뛴다 — 계산 절약.
- 호출부(`_rasterization` 내):

```python
if has_color:
    colors = _maybe_evaluate_sh(
        sh_degree, colors, means, radii, viewmats, batch_dims, C, N, True   # clamp=True
    )
if extra_signals is not None:
    # Do not clamp it.
    extra_signals = _maybe_evaluate_sh(
        extra_signals_sh_degree, extra_signals, means, radii, viewmats, batch_dims, C, N, False
    )
```

색에는 `clamp=True`, 부가 신호에는 `clamp=False`. 또 파일 상단 `_POST_CODE = {"none": 0, "shift": 1, "shift_relu": 2}`는 같은 세 가지 후처리(항등 / $+0.5$ / $\max(x+0.5,0)$)를 CUDA에 융합한 경로(`SHPostOp`)용 코드다.

### `gsplat/cuda/_torch_impl.py` — `_spherical_harmonics` (PyTorch 참조 구현)

CUDA 커널과 같은 값을 내는 순수 PyTorch 버전. 테스트와 이해용으로 쓰인다.

```python
# gsplat/cuda/_torch_impl.py
def _spherical_harmonics(
    degrees_to_use: int,
    dirs: torch.Tensor,  # [..., N, 3]
    coeffs: torch.Tensor,  # [N, K, D]
):
    K = coeffs.shape[1]
    dirs = F.normalize(dirs, p=2, dim=-1)                       # d ← (μ − o_cam)/‖·‖
    num_bases = (degrees_to_use + 1) ** 2                       # 활성화 차수까지만 사용
    bases = coeffs.new_zeros(dirs.shape[:-1] + (K,))
    bases[..., :num_bases] = _eval_sh_bases_fast(num_bases, dirs)   # Y_k(d)
    return (bases[..., None] * coeffs).sum(dim=-2)              # Σ_k c_k Y_k(d)
```

- `F.normalize`가 공식의 분모 $\|\boldsymbol\mu_i-\mathbf o_{\text{cam}}\|$ 역할이다(입력 `dirs`는 정규화되지 않은 $\boldsymbol\mu_i-\mathbf o_{\text{cam}}$).
- `_eval_sh_bases_fast`(같은 파일)는 Sloan(2013)의 점화식으로 $Y_k$를 4차(25개)까지 계산한다. 노트북의 `sh_bases`는 이를 $(\ell,m)$별로 풀어 쓴 것이며, 5장 교차 검증 셀에서 두 구현의 차이가 부동소수 오차 수준임을 확인한다.
- `degrees_to_use < 3`이면 나머지 기저를 0으로 두어 **고차 계수를 사실상 끈다** — 학습 초기 `sh_degree_interval` 스케줄이 이 인자로 들어온다.
- `+0.5`와 클램프는 이 함수에 **없다**. 그것은 호출부(`_maybe_evaluate_sh`)의 책임이다.

### `gsplat/cuda/csrc/SphericalHarmonics.cuh` — 방향 계산 (CUDA)

```cpp
__device__ vec3 camera_offset_from_world_to_camera(const float* viewmat, ...) {
    const float tx = viewmat[3], ty = viewmat[7], tz = viewmat[11];   // t
    vec3 camera_offset(viewmat[0]*tx + viewmat[4]*ty + viewmat[8]*tz,  // (Rᵀt).x
                       viewmat[1]*tx + viewmat[5]*ty + viewmat[9]*tz,  // (Rᵀt).y
                       viewmat[2]*tx + viewmat[6]*ty + viewmat[10]*tz); // (Rᵀt).z
    ...
}
__device__ vec3 view_direction_from_world_to_camera(const float* mean, const float* viewmat) {
    return vec3(mean[0], mean[1], mean[2]) + camera_offset_from_world_to_camera(viewmat);  // μ + Rᵀt = μ − o_cam
}
```

`camera_offset` $= R^\top\mathbf t = -\mathbf o_{\text{cam}}$이므로 `mean + camera_offset` $=\boldsymbol\mu_i-\mathbf o_{\text{cam}}$이다. `viewmat_rs`가 주어지면(롤링 셔터) 시작·끝 자세의 평균을 쓴다.

### 흐름 요약

```
examples/simple_trainer.py     sh0 [N,1,3] + shN [N,15,3]  → colors [N,16,3]
        │                      sh_degree_to_use = min(step // 1000, 3)
        ▼
gsplat/rendering.py::rasterization()
   ├─ fully_fused_projection → radii, means2d, conics, depths
   ├─ isect_tiles / isect_offset_encode
   ├─ _maybe_evaluate_sh(clamp=True)
   │     └─ spherical_harmonics(sh_degree, means, viewmats, coeffs, masks=radii>0)
   │           └─ CUDA: d = normalize(μ + Rᵀt); Σ_k c_k Y_k(d)      ← SphericalHarmonics.cuh / .cu
   │     └─ torch.clamp_min(· + 0.5, 0.0)                              ← max(0, Σ + 0.5)
   └─ rasterize_to_pixels(colors, ...)  → 알파 블렌딩 Σ c_i α_i Π(1−α_j)
```

## 6. 한계 — 왜 16개에서 멈추고, 무엇을 못 그리나

노트북 1.3의 실험이 보여주듯, 차수 3(16개 계수)까지의 SH는 **부드러운** 방향 변화(넓은 광택, 프레넬, 하늘 그라디언트)는 잘 맞추지만, 좁고 날카로운 봉우리는 흐릿하게 퍼지고 ringing이 생긴다. 계수 수는 $(L+1)^2$로 늘어나므로 고주파를 잡으려면 메모리가 급격히 커진다(Gaussian 수백만 개 × 48 실수가 이미 상당한 비중). 그래서 3DGS는 거울 반사·굴절 같은 날카로운 시점 의존성을 잘 못 그리며, 후속 연구는 SH 대신 작은 MLP 디코더, Spherical Gaussians, 반사 방향 기반 인코딩 등으로 이를 보완한다.

## 요약 표

| 기호 | 뜻 | 코드 |
|---|---|---|
| $\mathbf c_{i,k}\in\mathbb R^3$ | Gaussian $i$의 $k$번째 SH 계수 (학습 파라미터, 16×3) | `sh0`(k=0), `shN`(k=1..15) → `colors [N,16,3]` |
| $Y_k(\mathbf d)$ | 고정된 SH 기저, $k=\ell^2+\ell+m$, $\ell\le3$ | `_eval_sh_bases_fast` |
| $\mathbf d$ | 카메라→Gaussian 단위 방향 (월드 좌표) | `normalize(mean + Rᵀt)` |
| $\mathbf o_{\text{cam}}=-R^\top\mathbf t$ | world→camera `viewmat`에서 복원한 카메라 위치 | `camera_offset_from_world_to_camera` (부호 반대로 저장) |
| $+0.5$ | 계수 0 ↔ 중간 회색; 초기화 `(rgb−0.5)/C0`의 근거 | `features + 0.5` |
| $\max(0,\cdot)$ | 음의 색 방지, Inria 호환; 기울기 0 영역이라는 부작용 | `torch.clamp_min(…, 0.0)` (색만, `extra_signals`는 제외) |
