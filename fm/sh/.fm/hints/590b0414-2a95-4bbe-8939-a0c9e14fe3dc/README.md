# gsplat `_eval_sh_bases_fast` — Sloan(2013) 점화식으로 SH 기저를 계산하는 방법과 4차(25개) 한계

> **Q.** gsplat의 `_eval_sh_bases_fast`는 어떤 방식으로 기저를 계산하며 최대 몇 차까지 지원하는가?
> **A.** Sloan(2013)의 **점화식**으로 기저를 계산한다. **4차, 즉 25개 기저**까지 지원한다.

이 문서는 `gsplat/cuda/_torch_impl.py`의 `_eval_sh_bases_fast`(968행부터)를 한 줄씩 따라가며,
왜 그것이 "점화식"이고, 왜 빠르며, 왜 25개에서 멈추는지를 정리한다.
비교 대상으로 노트북 `sh_walkthrough.py`의 하드코딩 다항식 `sh_bases`와 CUDA 커널 `sh_coeffs_to_color_fast`를 함께 본다.

---

## 1. 함수 시그니처와 `basis_dim`

```python
def _eval_sh_bases_fast(basis_dim: int, dirs: Tensor):
    """
    Evaluate spherical harmonics bases at unit direction for high orders
    using approach described by
    Efficient Spherical Harmonic Evaluation, Peter-Pike Sloan, JCGT 2013
    https://jcgt.org/published/0002/02/06/

    :param basis_dim: int SH basis dim. Currently, only 1-25 square numbers supported
    :param dirs: torch.Tensor (..., 3) unit directions
    :return: torch.Tensor (..., basis_dim)
    """
```

- 첫 인자는 **차수(degree)가 아니라 기저 개수 `basis_dim`**이다. 차수 $L$까지 쓰면 기저는 $(L+1)^2$개이므로
  허용값은 **1, 4, 9, 16, 25** (각각 $L=0,1,2,3,4$)의 다섯 개뿐이다. docstring의 "1-25 square numbers"가 그 뜻이다.
- 코드는 `if basis_dim <= 1: return`, `<= 4`, `<= 9`, `<= 16`의 조기 반환으로 차수를 구분한다. 제곱수가 아닌 값
  (예: 5)을 넣으면 검사 없이 `result[..., 8]`까지 쓰려 하므로 인덱스 오류가 난다 — 즉 제곱수 제한은 assert가 아니라
  **구조적으로** 강제된다.
- `dirs`는 **단위 방향**이어야 한다. 정규화는 호출자 `_spherical_harmonics`가 `F.normalize(dirs, dim=-1)`로 미리 한다.
  호출부는 다음과 같다.

```python
num_bases = (degrees_to_use + 1) ** 2
bases = coeffs.new_zeros(dirs.shape[:-1] + (K,))
bases[..., :num_bases] = _eval_sh_bases_fast(num_bases, dirs)
return (bases[..., None] * coeffs).sum(dim=-2)
```

  `K`(계수 텐서의 기저 수, 3DGS는 16)와 `num_bases`(이번 스텝에 활성화된 차수까지의 기저 수)를 분리해서,
  아직 켜지지 않은 고차 기저는 0으로 남겨 계수와 곱해도 기여가 없게 한다. 학습 초기 `sh_degree_interval` 스케줄이
  이 경로로 동작한다.

- 출력의 인덱스 순서는 3DGS/gsplat 규약 $k = \ell^2 + \ell + m$ 이다. 그래서 코드가 `result[..., 2]`, `[3]`, `[1]`처럼
  순서가 뒤섞여 보이는데, 이는 $m=0 \to +1 \to -1$ 순으로 **계산**한 뒤 각자 자기 슬롯 $k$에 **저장**하기 때문이다.

---

## 2. 코드 구조 — "z-종속 항 먼저, 그다음 $\pm m$ 쌍"

실수 SH의 정의를 떠올리면 구조가 바로 보인다 (구면좌표 $\mathbf d=(\sin\theta\cos\varphi,\ \sin\theta\sin\varphi,\ \cos\theta)$).

$$
Y_\ell^{\pm m}(\theta,\varphi) \;\propto\; K_\ell^m\,P_\ell^m(\cos\theta)\times
\begin{cases}\cos m\varphi \\ \sin m\varphi\end{cases}
\qquad (m>0)
$$

즉 모든 기저는 **[z만의 함수] × [$\varphi$만의 함수]** 로 인수분해된다. Sloan의 코드는 이 두 인자를 따로 만들어 곱한다.

### 2.1 z-종속 인자: `fTmpA/B/C/D`와 zonal 항

각 차수 블록의 앞부분은 $z=\cos\theta$의 다항식만 만든다. 3차 블록을 보자.

```python
fTmpC = -2.285228997322329 * z2 + 0.4570457994644658    # m=±1 용:  ∝ P_3^1(z)/sinθ
fTmpB = 1.445305721320277 * z                            # m=±2 용:  ∝ P_3^2(z)/sin²θ
fTmpA = -0.5900435899266435                              # m=±3 용:  ∝ P_3^3(z)/sin³θ  (상수)
result[..., 12] = z * (1.865881662950577 * z2 - 1.119528997770346)   # m=0 (zonal):  K_3^0 P_3(z)
```

- `result[..., ℓ²+ℓ]`(m=0)은 **zonal harmonic** — 방위각에 무관하고 z에만 의존한다. 이것은 르장드르 다항식
  $K_\ell^0 P_\ell(z)$을 그대로 쓴 것이다. 4차에서는 `1.9843·z·pSH12 − 1.0062·pSH6` 꼴로, **아래 차수의 zonal 항 두 개로
  다음 차수를 만드는 르장드르 3항 점화식** $P_\ell = \frac{(2\ell-1)zP_{\ell-1} - (\ell-1)P_{\ell-2}}{\ell}$ 의 정규화 버전이다
  (torch 코드는 이를 풀어서 썼고, CUDA 코드는 `pSH12`, `pSH6` 변수를 그대로 재사용한다).
- `fTmp*`는 연관 르장드르 함수 $P_\ell^m(\cos\theta)$에서 $\sin^m\theta$ 인자를 **미리 떼어낸 것**이다.
  $P_\ell^m(\cos\theta) = \sin^m\theta \cdot (\text{z의 } (\ell-m)\text{차 다항식})$ 이므로, 남는 부분은 z의 다항식이고
  떼어낸 $\sin^m\theta$는 아래 2.2의 $\varphi$ 인자에 흡수된다.
- 이름 규칙: 같은 문자(`A`)는 "z 다항식이 상수인 것", 즉 $m=\ell$ 항(sectoral)이고, `B`, `C`, `D`로 갈수록 z의 차수가
  1, 2, 3으로 올라가며 $m$은 하나씩 내려간다. 각 차수 블록에서 `fTmpA`가 새 상수로 덮어써진다.

### 2.2 $\varphi$-종속 인자: `fC/fS` 점화식 — 삼각함수 호출 없이 $\cos m\varphi$, $\sin m\varphi$

핵심이 되는 네 줄이다.

```python
fC1 = x * x - y * y          # ℓ=2 블록
fS1 = 2 * x * y
fC2 = x * fC1 - y * fS1      # ℓ=3 블록
fS2 = x * fS1 + y * fC1
fC3 = x * fC2 - y * fS2      # ℓ=4 블록
fS3 = x * fS2 + y * fC2
```

$x = \sin\theta\cos\varphi,\ y=\sin\theta\sin\varphi$ 이므로 복소수 $w = x + iy = \sin\theta\,e^{i\varphi}$ 로 두면
$w^m = \sin^m\theta\,(\cos m\varphi + i\sin m\varphi)$ 이다. 위 점화식은 정확히 **$w^{m} = w \cdot w^{m-1}$ 의 실수부/허수부**다.

$$
\underbrace{\mathrm{fC}_m}_{\Re(w^{m+1})} = x\,\mathrm{fC}_{m-1} - y\,\mathrm{fS}_{m-1},\qquad
\underbrace{\mathrm{fS}_m}_{\Im(w^{m+1})} = x\,\mathrm{fS}_{m-1} + y\,\mathrm{fC}_{m-1}
$$

따라서

- `fC1` $= \sin^2\theta\cos 2\varphi$, `fS1` $= \sin^2\theta\sin 2\varphi$
- `fC2` $= \sin^3\theta\cos 3\varphi$, `fS2` $= \sin^3\theta\sin 3\varphi$
- `fC3` $= \sin^4\theta\cos 4\varphi$, `fS3` $= \sin^4\theta\sin 4\varphi$

이고 $m=1$은 `x`, `y` 자체가 $\sin\theta\cos\varphi$, $\sin\theta\sin\varphi$ 다. (이 항등식을 무작위 방향 5개에서 수치로
확인하면 차이가 $10^{-16}$ 수준이다.) 즉 **삼각함수를 한 번도 부르지 않고**, $x, y$의 곱셈·뺄셈만으로
$\cos m\varphi\sin^m\theta$ 와 $\sin m\varphi \sin^m\theta$ 를 차례로 점화해 낸다. 2.1에서 떼어낸 $\sin^m\theta$ 인자가
여기서 정확히 복원되므로 둘을 곱하면 원래의 $P_\ell^m(\cos\theta)\cos m\varphi$ 가 된다.

### 2.3 조립: 각 $\pm m$ 쌍은 같은 z 인자를 공유한다

```python
result[..., 13] = fTmpC * x      # (3,+1) :  z다항식 × cosφ sinθ
result[..., 11] = fTmpC * y      # (3,−1) :  z다항식 × sinφ sinθ
result[..., 14] = fTmpB * fC1    # (3,+2) :  z다항식 × cos2φ sin²θ
result[..., 10] = fTmpB * fS1    # (3,−2)
result[..., 15] = fTmpA * fC2    # (3,+3) :  상수    × cos3φ sin³θ
result[..., 9]  = fTmpA * fS2    # (3,−3)
```

$+m$과 $-m$은 **z 인자를 공유**하고 $\varphi$ 인자만 `fC`/`fS`로 다르다. 그래서 각 쌍은 곱셈 2번으로 끝난다.
정리하면 한 차수 블록의 계산 순서는

1. `z2`, `fTmp*` — z-종속 인자 (차수마다 1~3개)
2. `fC_m`, `fS_m` — 새 $\varphi$ 인자 한 쌍 (곱셈 4, 덧셈 2)
3. zonal 항 `result[ℓ²+ℓ]`
4. $m=\pm1, \pm2, \dots, \pm\ell$ 쌍을 `fTmp × fC/fS`로 채움

이며, 위 차수는 아래 차수의 `fC/fS`와 `z2`를 그대로 재사용한다. 이것이 "점화식으로 계산한다"의 실제 의미다.

---

## 3. 왜 이 방식이 빠른가

| 관점 | 일반적인 정의식 평가 | Sloan 방식 |
|---|---|---|
| 각도 변환 | $\theta=\arccos z,\ \varphi=\mathrm{atan2}(y,x)$ 필요 | 없음 — 직교좌표 $x,y,z$를 그대로 사용 |
| $\cos m\varphi,\ \sin m\varphi$ | $m$마다 `cos`/`sin` 호출 또는 Chebyshev 점화 | `fC/fS` 점화: 곱셈 4 + 덧셈 2 로 다음 $m$ |
| $P_\ell^m$ | 계승·제곱근 포함한 일반식 | 정규화 상수를 **컴파일 타임에 접어 넣은** 소수 리터럴 하나 |
| 분기 | 루프 + `m` 부호 분기 | 차수 경계에서만 `if basis_dim <= n` (모든 스레드가 같은 방향으로 분기) |

- 수치적으로 세어 보면 3차 블록 전체(7개 기저)는 곱셈 약 12번, 덧셈 약 4번이다. 16개 기저 전체가 곱셈 30여 번,
  덧셈 10여 번 정도로 끝난다. 초월함수(`sin`, `cos`, `sqrt`, `acos`)가 **0번**이다.
- 상수(`0.5462742…` 등)는 $\sqrt2\,K_\ell^m$ 에 연관 르장드르 함수의 선행 계수(예: $P_2^2$의 3, $P_3^3$의 15, $P_4^4$의 105)를
  곱한 값이다. 실제로 $\sqrt2 K_2^2\cdot 3 = 0.54627$, $\sqrt2 K_3^3\cdot 15 = 0.59004$, $\sqrt2 K_4^4\cdot 105 = 0.62584$ 로 일치한다.
  Sloan은 이 상수들을 심볼릭 계산으로 미리 뽑아 **코드 생성기**로 C 코드를 찍어냈고(논문의 `code.zip`), gsplat은 그 출력을
  Python/CUDA로 옮겼다.
- GPU 친화적인 이유: 모든 연산이 FMA(융합 곱셈-덧셈)로 컴파일되고, 데이터 의존 분기가 없어 warp가 갈라지지 않으며,
  중간값이 레지스터 몇 개(`fC`, `fS`, `fTmp`, `z2`)로 끝나 메모리 접근이 없다. 3DGS는 카메라×Gaussian마다 이 계산을 하므로
  수백만 회/프레임이 되는데, 이 비용이 사실상 계수 내적(16×3 FMA)에 묻힌다.
- PyTorch 구현에서도 이점이 있다: 텐서 연산 개수 = 커널 런치 횟수이므로, 삼각함수·`atan2`·거듭제곱을 부르지 않고
  곱셈·덧셈 몇 십 개로 끝나는 것이 그대로 속도로 이어진다. 또 미분이 다항식이라 autograd가 단순하다.

논문 제목 그대로 "Efficient Spherical Harmonic Evaluation"(Peter-Pike Sloan, JCGT vol. 2 no. 2, 2013)의 요지는
**"SH 평가를 각도가 아닌 직교좌표 다항식으로, 그리고 $m$ 방향은 복소수 거듭제곱 점화식으로 하면 고차에서도 분기 없는
직선 코드가 된다"** 는 것이고, 논문은 임의 차수까지 이런 코드를 생성하는 방법을 제시한다.

---

## 4. 하드코딩 다항식(노트북 `sh_bases`)과의 관계

노트북 `sh_walkthrough.py`의 `sh_bases`는 같은 기저를 **$(\ell,m)$별로 완전히 전개한 다항식**으로 쓴다. 3차의 예:

```python
c1, c2, c3, c4, c5 = 0.5900435899266435, 2.890611442640554, 0.4570457994644658, 0.3731763325901154, 1.445305721320277
out += [-c1 * y * (3 * x * x - y * y), c2 * x * y * z, -c3 * y * (5 * z * z - 1), c4 * z * (5 * z * z - 3),
        -c3 * x * (5 * z * z - 1), c5 * z * (x * x - y * y), -c1 * x * (x * x - 3 * y * y)]
```

두 구현은 **수학적으로 동일**하다. 대응을 직접 확인해 보자.

| k | `sh_bases` (전개형) | `_eval_sh_bases_fast` (점화형) | 일치 근거 |
|---|---|---|---|
| 9 $(3,-3)$ | $-c_1\,y(3x^2-y^2)$ | `fTmpA * fS2` $= -0.5900\,(x\cdot 2xy + y(x^2-y^2))$ | $x\cdot2xy + y(x^2-y^2) = 3x^2y - y^3 = y(3x^2-y^2)$ |
| 15 $(3,+3)$ | $-c_1\,x(x^2-3y^2)$ | `fTmpA * fC2` $= -0.5900\,(x(x^2-y^2) - y\cdot 2xy)$ | $x^3 - 3xy^2 = x(x^2-3y^2)$ |
| 10 $(3,-2)$ | $c_2\,xyz = 2.8906\,xyz$ | `fTmpB * fS1` $= 1.4453\,z \cdot 2xy$ | $1.4453\times2 = 2.8906$ |
| 11 $(3,-1)$ | $-c_3\,y(5z^2-1)$ | `fTmpC * y` $= (-2.2852 z^2 + 0.4570)\,y$ | $0.4570\times5 = 2.2852$ |
| 12 $(3,0)$ | $c_4\,z(5z^2-3) = 0.3732\,z(5z^2-3)$ | $z(1.8659 z^2 - 1.1195)$ | $0.3732\times5=1.8659,\ 0.3732\times3=1.1195$ |

- 상수는 모두 같은 $K_\ell^m$(과 $\sqrt2$, 르장드르 선행계수)에서 나오며, 다만 어느 인자에 곱해 두었느냐만 다르다.
  전개형은 "$(\ell,m)$ 하나 = 다항식 하나"라 읽기 쉽고, 점화형은 "공통 인자 재사용"이라 연산이 적다.
- 부호(Condon–Shortley 규약)도 같다. 예를 들어 1차에서 `fTmpA = -0.4886`으로 잡고 `result[3] = fTmpA * x`,
  `result[1] = fTmpA * y`, `result[2] = -fTmpA * z` 로 쓴 것이 노트북의 `[-c*y, c*z, -c*x]`와 정확히 대응한다.
- 노트북이 하드코딩을 택한 이유는 교육 목적이다: $\ell$차 SH가 "$x,y,z$의 $\ell$차 동차다항식"이라는 사실이 눈에 보이도록.
  `sh_bases`의 docstring도 "`_eval_sh_bases_fast`와 같은 값·순서를 $(\ell, m)$별로 풀어 쓴 것"이라고 명시한다.

---

## 5. 왜 4차(25개)까지만 구현했나

1. **3DGS의 기본은 3차(16개)** 다. 원논문과 gsplat `simple_trainer.py`의 기본값 `sh_degree=3`이며, 노트북 1.3절이 보여주듯
   SH는 부드러운 시점 의존성(광택, 프레넬)까지가 표현 한계라 차수를 더 올려도 이득이 급격히 줄고 파라미터(Gaussian당
   $(L+1)^2\times3$)와 메모리만 늘어난다. 4차(25개, 75 실수)는 "한 단계 여유"로 넣은 것이다.
2. **코드 크기와 레지스터**. Sloan 방식은 루프가 없는 직선 코드라 차수마다 코드가 $2\ell+1$줄씩 늘고, 역전파(`_vjp`)는
   더 길어진다(CUDA의 `sh_coeffs_to_color_fast_vjp`는 4차 처리에만 100줄 이상). 필요 없는 차수를 넣으면 컴파일 시간과
   레지스터 압력만 커진다.
3. **인터페이스가 명시적으로 상한을 갖는다**. CUDA 쪽 `gsplat/cuda/csrc/SphericalHarmonics.h`에
   `inline constexpr int SH_MAX_DEGREE = 4;` 가 있고, `SphericalHarmonics.cpp`의 `check_spherical_harmonics_degree`가
   `0 <= degrees_to_use <= SH_MAX_DEGREE`를 `TORCH_CHECK`로 강제한다. 커널 디스패치도 `dispatch::IntParam<0,1,2,3,4>`
   로 0~4차만 템플릿 인스턴스화한다. torch 구현의 "1-25 square numbers"는 이 상한과 짝을 맞춘 것이다.
4. Sloan의 원 코드 생성기는 임의 차수를 만들 수 있으므로, 더 높은 차수가 필요하면 같은 패턴으로 블록을 추가하면 된다.
   다만 위 세 곳(`_eval_sh_bases_fast`, `SH_MAX_DEGREE`, `IntParam`)과 `_vjp`를 함께 늘려야 한다.

---

## 6. 노트북 `COMPARE_WITH_GSPLAT` 셀이 검증하는 것

```python
COMPARE_WITH_GSPLAT = False
if COMPARE_WITH_GSPLAT:
    from gsplat.cuda._torch_impl import _eval_sh_bases_fast, _spherical_harmonics
    d = F.normalize(torch.randn(10000, 3, device=DEVICE), dim=-1)
    print("기저 차이 :", (sh_bases(d, 3) - _eval_sh_bases_fast(16, d)).abs().max().item())
    c = torch.randn(10000, 16, 3, device=DEVICE)
    print("평가 차이 :", (sh_eval(c, d, 3) - _spherical_harmonics(3, d, c)).abs().max().item())
```

- **기저 차이**: 무작위 단위 방향 10,000개에서 노트북의 전개형 `sh_bases(d, 3)`와 gsplat의 점화형 `_eval_sh_bases_fast(16, d)`
  (인자가 차수 3이 아니라 **기저 수 16**임에 주목) 의 16개 값이 원소별로 같은지를 최대 절대 오차로 본다. 두 식이 4절처럼
  대수적으로 동일하므로 float32 반올림 수준($10^{-7}$ 안팎)이 나와야 한다. 이는 (a) 상수 $K_\ell^m$, (b) 부호 규약, (c) 인덱스
  순서 $k=\ell^2+\ell+m$ 세 가지가 모두 일치함을 한 번에 확인한다.
- **평가 차이**: 계수 `[N,16,3]`까지 곱해 색을 내는 `sh_eval`과 `_spherical_harmonics`를 비교한다. 여기서는 정규화 방식
  (`F.normalize`)과 축소 순서(`einsum` vs `(bases[...,None]*coeffs).sum(-2)`)까지 같은지 확인하는 셈이다.
- 기본값이 `False`인 이유는 `import gsplat`이 CUDA 확장의 JIT 빌드를 유발할 수 있어서다.

---

## 7. CUDA 커널도 같은 점화식을 쓰는가 — 확인 결과: **예, 동일하다**

`gsplat/cuda/csrc/SphericalHarmonicsCUDA.cu`의 `sh_coeffs_to_color_fast`(49행부터)는 주석부터 같은 논문을 인용한다.

```cpp
// Evaluate spherical harmonics bases at unit direction for high orders using
// approach described by Efficient Spherical Harmonic Evaluation, Peter-Pike
// Sloan, JCGT 2013 See https://jcgt.org/published/0002/02/06/ for reference
// implementation
```

본문을 torch 버전과 나란히 두면 변수명(`fTmp0B`, `fC1`, `fS1`, `fC2 = x*fC1 - y*fS1`, …)과 상수 리터럴이 그대로다.

```cpp
float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
float fTmp1B = 1.445305721320277f * z;
float fC2    = x * fC1 - y * fS1;
float fS2    = x * fS1 + y * fC1;
float pSH12  = z * (1.865881662950577f * z2 - 1.119528997770346f);
float pSH13  = fTmp0C * x;
float pSH11  = fTmp0C * y;
float pSH14  = fTmp1B * fC1;
float pSH10  = fTmp1B * fS1;
float pSH15  = -0.5900435899266435f * fC2;
float pSH9   = -0.5900435899266435f * fS2;
```

차이점은 구현 형태에서만 나온다.

| 항목 | torch `_eval_sh_bases_fast` | CUDA `sh_coeffs_to_color_fast` |
|---|---|---|
| 차수 인자 | `basis_dim` (1/4/9/16/25) | `degree` (0~4), `if(degree >= n)` 중첩 |
| 출력 | 기저 텐서 `[..., basis_dim]`를 반환, 계수 곱은 호출자가 | 기저를 만드는 즉시 `coeffs[k*D + c]`와 **곱해 누적**해 색 하나를 반환(기저 배열을 메모리에 쓰지 않음) |
| 정규화 | 호출자(`F.normalize`) | 함수 안에서 `rsqrtf`로 직접 |
| 4차 zonal 항 | 상수를 풀어 쓴 식 | `pSH20 = 1.9843f * z * pSH12 - 1.0062f * pSH6` — 아래 차수 변수를 재사용해 르장드르 점화가 더 명시적 |
| 상한 | docstring "1-25" + 구조적 | `SH_MAX_DEGREE = 4` + `TORCH_CHECK` + `IntParam<0,1,2,3,4>` |

즉 CUDA 커널은 **같은 Sloan 점화식, 같은 상수, 같은 4차 상한**을 쓰며, 기저를 따로 저장하지 않고 계수와 바로 내적한다는
점만 다르다. 역전파 `sh_coeffs_to_color_fast_vjp`도 같은 중간변수(`fC1_x`, `fS1_y`, `pSH7_z` …)의 편미분을 손으로 전개한 것이다.
참고로 같은 파일에는 `K == 16 && D == 3`인 3DGS 기본 형태만을 위한 특화 경로(`spherical_harmonics_fwd_kernel_k16_3channel`,
차수를 템플릿 인자로 받아 `std::min(degrees_to_use, 3)`)도 있는데, 이 역시 내부 기저 계산은 동일한 점화식이다.

---

## 요약

- `_eval_sh_bases_fast(basis_dim, dirs)`: **기저 개수**(1·4·9·16·25)를 받아 단위 방향에서 SH 기저를 계산한다.
- 방식: Sloan(2013). 각 기저를 **z-다항식(`fTmp*`, zonal은 르장드르 점화) × $\varphi$-인자**로 나누고, $\varphi$-인자는
  `fC_m = x·fC_{m-1} − y·fS_{m-1}`, `fS_m = x·fS_{m-1} + y·fC_{m-1}` (복소수 $w^m$의 실·허부) 로 **삼각함수 없이 점화**한다.
- 빠른 이유: 초월함수 0회, 곱셈·덧셈 수십 번, 데이터 의존 분기 없음, 상수는 미리 접어 넣음 — GPU FMA 직선 코드.
- 노트북 `sh_bases`와 대수적으로 동일(상수 $\sqrt2K_\ell^m\times$르장드르 선행계수), `COMPARE_WITH_GSPLAT` 셀이 이를 수치로 확인.
- **4차(25개)까지**: 3DGS 기본 3차 + 한 단계 여유, 코드/레지스터 비용, `SH_MAX_DEGREE = 4`와 정합. CUDA 커널도 같은 점화식·상한을 쓴다.
