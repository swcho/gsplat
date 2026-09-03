# 3DGS는 Gaussian의 색을 어떻게 저장하는가 — SH 계수 16 × 3

**질문**: 3D Gaussian Splatting(3DGS)은 각 Gaussian의 색을 어떤 형태로 저장하는가?

**답**: RGB 값 하나가 아니라 **Spherical Harmonics(SH) 계수 16개 × 3채널(총 48개 실수)** 로 저장한다. 덕분에 *보는 방향*에 따라 색이 달라지는 **시점 의존 색(view-dependent color)** 을 표현할 수 있다.

---

## 1. 왜 RGB 하나로는 부족한가

실제 물체의 색은 보는 각도에 따라 바뀐다.

- 광택 표면의 **하이라이트**는 카메라가 움직이면 따라 움직인다.
- 비스듬히 볼수록 반사가 강해지는 **프레넬 효과**.
- 반투명 물질, 미세한 표면 요철 등.

Gaussian마다 RGB 값 하나만 두면 어느 카메라에서 봐도 똑같은 색이 나와 이런 현상을 재현할 수 없다. 그래서 3DGS는 "색"을 숫자 3개가 아니라 **방향 → 색 함수** $\mathbf c(\mathbf d)$로 저장한다. 이 함수를 소수의 숫자로 압축해 두는 도구가 SH다.

## 2. SH란 — 구면 위 함수의 "푸리에 급수"

단위 방향 벡터 $\mathbf d$ ($\|\mathbf d\|=1$)에 대해 정의된 함수는 SH 기저 $Y_\ell^m$의 선형결합으로 근사할 수 있다.

$$
f(\mathbf d)\approx\sum_{\ell=0}^{L}\sum_{m=-\ell}^{\ell} c_\ell^m\,Y_\ell^m(\mathbf d)
$$

- $\ell$ = 차수(degree). 클수록 구면 위에서 더 빠르게 진동하는 고주파 성분.
- 차수 $\ell$에는 $2\ell+1$개 기저가 있어, 차수 $L$까지 쓰면 총 $(L+1)^2$개.
- **3DGS는 $L=3$** → $1+3+5+7 = 16$개 기저.

각 기저마다 R, G, B 계수가 따로 필요하므로 Gaussian 하나의 색 파라미터는

$$
16 \text{(기저)} \times 3 \text{(채널)} = 48 \text{개 실수}
$$

이다. 기저는 $x,y,z$의 $\ell$차 다항식이라 계산이 매우 싸다(0차는 상수, 1차는 $x,y,z$ 자체, 2차는 $xy, yz, 3z^2-1, \dots$).

## 3. 색은 어떻게 계산되나 — SH 평가

렌더 시 카메라 위치 $\mathbf o_{\text{cam}}$에서 Gaussian 중심 $\boldsymbol\mu_i$를 보는 방향으로 급수를 더한다.

$$
\mathbf c_i(\mathbf d)=\max\!\Big(0,\ \sum_{k=0}^{15}\mathbf c_{i,k}\,Y_k(\mathbf d)+0.5\Big),\qquad
\mathbf d=\frac{\boldsymbol\mu_i-\mathbf o_{\text{cam}}}{\|\boldsymbol\mu_i-\mathbf o_{\text{cam}}\|}
$$

- 카메라마다·Gaussian마다 한 번씩, 채널당 곱셈-덧셈 16번이면 끝. 뉴럴 네트워크가 없어서 수백만 Gaussian을 실시간으로 그릴 수 있다.
- `+0.5`는 계수가 전부 0일 때 색이 검정이 아닌 **중간 회색**이 되도록 하는 오프셋, `max(0, ·)`은 음수 색을 잘라내는 클램프.

## 4. 계수 배열의 구조 — DC(`sh0`)와 나머지(`shN`)

| 부분 | 인덱스 | 형태 | 의미 |
|---|---|---|---|
| **DC 계수** `sh0` | $k=0$ ($\ell=0$) | `[N, 1, 3]` | 방향 무관 **기본색**. $c_0 Y_0^0$은 색 함수의 구면 평균 |
| 고차 계수 `shN` | $k=1\ldots15$ ($\ell=1,2,3$) | `[N, 15, 3]` | 평균이 0인 **시점 의존 변동**(하이라이트 등) |

"DC"는 신호처리의 direct current, 즉 주파수 0 성분에서 온 용어다. $Y_0^0 = \frac{1}{2\sqrt\pi}\approx 0.2821$은 상수이므로 DC 항만 있으면 어느 방향에서 봐도 같은 색이 나온다.

### 초기화: `sh0 = (rgb − 0.5) / C0`

학습 시작 시 SfM 포인트 색 $\mathbf{rgb}$를 재현하려면 고차 계수를 0으로 두고

$$
c_0 Y_0^0 + 0.5 = \mathbf{rgb}\ \Longrightarrow\ c_0=\frac{\mathbf{rgb}-0.5}{C_0},\quad C_0 = Y_0^0
$$

로 두면 된다. 즉 처음에는 "RGB 하나"와 동일하게 시작하고, 학습이 진행되며 `shN`이 시점 의존성을 채워 나간다.

## 5. gsplat 코드에서의 모습

`examples/simple_trainer.py`:

```python
colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))   # [N, 16, 3]  (sh_degree=3)
params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))          # [N, 1, 3]
params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))          # [N, 15, 3]
...
colors = torch.cat([splats["sh0"], splats["shN"]], 1)  # [N, K, 3] 로 합쳐 렌더러에 전달
```

- `sh0_lr = 2.5e-3`, `shN_lr = 2.5e-3 / 20` — 시점 의존성은 기본색이 잡힌 뒤 천천히 배우도록 학습률을 1/20로 둔다.
- `sh_degree_to_use = min(step // sh_degree_interval, sh_degree)` — 처음 1000스텝은 DC만 쓰고, 1000스텝마다 한 차수씩 활성화한다(coarse-to-fine).
- `gsplat/rendering.py`의 `_maybe_evaluate_sh`가 `spherical_harmonics(sh_degree, means, viewmats, coeffs)`로 급수를 더한 뒤 `torch.clamp_min(features + 0.5, 0.0)`을 적용한다.

## 6. 왜 3차(16개)에서 멈추나

SH는 **부드러운 구면 함수**를 소수의 계수로 압축하는 데 강하다(Ramamoorthi & Hanrahan, 2001: 확산 조명은 2차 9개 계수로 오차 약 1%). 그러나 좁고 날카로운 봉우리(거울 반사 등)를 표현하려면 차수를 급격히 올려야 하고, 그만큼 파라미터·메모리가 늘어난다. 16개는 광택·프레넬 수준의 부드러운 시점 의존성을 잡는 데 충분하면서 Gaussian당 48개 실수로 비용이 감당되는 타협점이다. 이것이 3DGS가 날카로운 반사·굴절을 잘 못 그리는 이유이기도 하며, 후속 연구는 작은 MLP, 구면 가우시안(Spherical Gaussians), 반사 방향 인코딩 등으로 이를 보완한다.

## 요약

- 저장 형태: **SH 계수 `[16, 3]` = 48개 실수** / Gaussian (RGB 하나 아님).
- 구성: DC 1개(`sh0`, 기본색) + 고차 15개(`shN`, 시점 의존 변동).
- 렌더 시: 카메라→Gaussian 방향으로 SH를 평가해 `+0.5`, `clamp_min(0)` → 그 카메라에서의 색.
- 효과: MLP 없이 곱셈-덧셈 48번으로 시점 의존 색 → 실시간 렌더링.

## 참고

- 원본 노트북: `fm/sh/.fm/assets/sh_walkthrough.py` (§1 SH 정의, §2 3DGS에서의 역할, §3 DC 계수, §4 SH 평가)
- Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023
- gsplat: `examples/simple_trainer.py`, `gsplat/rendering.py` (`_maybe_evaluate_sh`), `gsplat/cuda/_torch_impl.py` (`_eval_sh_bases_fast`, `_spherical_harmonics`)
