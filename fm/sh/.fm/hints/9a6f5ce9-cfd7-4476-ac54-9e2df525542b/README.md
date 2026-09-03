# 실제 코드에서 SH 기저는 구면 좌표 대신 어떤 형태로 계산하는가?

**답:** 직교좌표 $(x,y,z)$의 **다항식**으로 풀어 쓴다. 단위 방향 벡터의 성분을 곱하고 더하기만 해서
$Y_1^0=\sqrt{3/4\pi}\,z$, $Y_2^{-2}=\sqrt{15/4\pi}\,xy$ 처럼 계산하며, 삼각함수나 르장드르 함수는 호출하지 않는다.

---

## 1. 교과서 정의와 코드 구현의 차이

교과서(그리고 `sh_walkthrough.py` 1.2절)의 실수형 SH 정의는 구면 좌표 $(\theta,\varphi)$로 쓰여 있다.

$$
Y_\ell^m(\theta,\varphi)=
\begin{cases}
\sqrt{2}\,K_\ell^{m}\cos(m\varphi)\,P_\ell^{m}(\cos\theta) & m>0\\
K_\ell^{0}\,P_\ell^{0}(\cos\theta) & m=0\\
\sqrt{2}\,K_\ell^{|m|}\sin(|m|\varphi)\,P_\ell^{|m|}(\cos\theta) & m<0
\end{cases}
$$

이 식을 그대로 구현하려면 방향 벡터에서 $\theta=\arccos z$, $\varphi=\operatorname{atan2}(y,x)$를 구하고,
$\cos(m\varphi)$, $\sin(m\varphi)$, 연관 르장드르 함수 $P_\ell^m$을 차례로 평가해야 한다.

하지만 $\mathbf d=(x,y,z)=(\sin\theta\cos\varphi,\ \sin\theta\sin\varphi,\ \cos\theta)$ 를 대입하면
모든 항이 $x,y,z$의 **$\ell$차 동차 다항식**으로 정리된다. 예를 들어

| $(\ell,m)$ | 구면 좌표 형태 | 직교좌표 다항식 |
|---|---|---|
| $(1,0)$ | $\sqrt{3/4\pi}\cos\theta$ | $\sqrt{3/4\pi}\,z$ |
| $(1,-1)$ | $-\sqrt{3/4\pi}\sin\theta\sin\varphi$ | $-\sqrt{3/4\pi}\,y$ |
| $(2,-2)$ | $\sqrt{15/16\pi}\sin^2\theta\sin 2\varphi$ | $\sqrt{15/4\pi}\,xy$ |
| $(2,0)$ | $\sqrt{5/16\pi}(3\cos^2\theta-1)$ | $\sqrt{5/16\pi}(3z^2-1)$ |
| $(2,2)$ | $\sqrt{15/16\pi}\sin^2\theta\cos 2\varphi$ | $\sqrt{15/16\pi}(x^2-y^2)$ |

핵심 항등식은 $\sin\theta\cos\varphi = x$, $\sin\theta\sin\varphi=y$, $\cos\theta=z$ 이고,
$\sin^2\theta\sin 2\varphi = 2(\sin\theta\cos\varphi)(\sin\theta\sin\varphi)=2xy$ 같은 배각 공식이다.
즉 "구면 위 함수"라는 추상적 정의를 실제로 **단위구에 제한된 다항식**으로 바꿔 쓴 것이 코드의 형태다.

## 2. 왜 다항식 형태를 쓰는가

1. **정규화된 방향 벡터가 이미 있다.** 3DGS에서 SH를 평가하는 방향은 "카메라 중심 → Gaussian 중심" 벡터다
   (`gsplat/rendering.py`의 `_maybe_evaluate_sh` → `spherical_harmonics(sh_degree, means, viewmats, ...)`).
   각도 $(\theta,\varphi)$는 어디에도 없고, 벡터를 정규화하면 곧바로 $x,y,z$가 나온다.
   각도로 바꾸는 것은 `acos`/`atan2`를 추가로 부르는 순수한 낭비다.
2. **삼각함수·르장드르 함수 호출이 사라진다.** 다항식에는 곱셈과 덧셈만 남는다. GPU에서 곱셈·덧셈은
   FMA(fused multiply-add) 한 명령으로 처리되지만, `sin`/`cos`/`acos`는 수십 사이클짜리 특수 함수다.
   Gaussian 수백만 개 × 카메라마다 매 스텝 SH를 평가하므로 이 차이가 그대로 렌더링 시간에 반영된다.
3. **특이점이 없다.** 구면 좌표는 극($z=\pm1$, $\sin\theta=0$)에서 $\varphi$가 정의되지 않고,
   `atan2`의 불연속·gradient 폭주 문제가 있다. 다항식은 단위구 전체에서 매끄럽고,
   역전파(계수 gradient, 방향 gradient)도 단순한 곱셈 규칙으로 안정적으로 계산된다.
4. **수치 상수를 미리 접어 둘 수 있다.** 정규화 상수 $K_\ell^m$과 다항식 계수를 곱한 값
   (예: $0.4886 = \sqrt{3/4\pi}$, $1.0925 = \sqrt{15/4\pi}$, $0.9462 = 3\sqrt{5/16\pi}$)을 하드코딩한 리터럴로 두면
   런타임에는 팩토리얼도 제곱근도 계산하지 않는다.
5. **차수별로 재사용이 가능하다.** $\ell$차 다항식은 $\ell-1$차에서 만든 중간값($z^2$, $x^2-y^2$, $2xy$ …)을
   그대로 이어받아 만들 수 있어, 곱셈 몇 번씩만 추가해 차수를 올릴 수 있다(아래 점화식).

## 3. `sh_walkthrough.py`의 `sh_bases` — $(\ell,m)$별로 풀어 쓴 버전

노트북은 교육용으로 각 기저를 $(\ell,m)$ 순서대로 그대로 적었다. 인덱스 규약은 $k=\ell^2+\ell+m$.

```python
def sh_bases(dirs, degree):
    x, y, z = dirs.unbind(-1)
    out = [torch.full_like(x, C0)]                                   # (0,0)  C0 = 1/(2√π)
    if degree >= 1:
        c = 0.4886025119029199                                       # √(3/4π)
        out += [-c * y, c * z, -c * x]                               # (1,-1) (1,0) (1,1)
    if degree >= 2:
        c1, c2, c3 = 1.0925484305920792, 0.31539156525252005, 0.5462742152960396
        out += [c1 * x * y, -c1 * y * z, c2 * (3 * z * z - 1), -c1 * x * z, c3 * (x * x - y * y)]
    if degree >= 3:
        ...
        out += [-c1 * y * (3 * x * x - y * y), c2 * x * y * z, -c3 * y * (5 * z * z - 1),
                c4 * z * (5 * z * z - 3), -c3 * x * (5 * z * z - 1), c5 * z * (x * x - y * y),
                -c1 * x * (x * x - 3 * y * y)]
    return torch.stack(out, dim=-1)
```

보이는 대로 `sin`, `cos`, `acos`가 한 번도 등장하지 않는다. 음의 부호는 Condon–Shortley 규약(3DGS/gsplat과 동일).

## 4. gsplat 실제 구현 — Sloan(2013) 점화식으로 하드코딩

gsplat은 Peter-Pike Sloan, *Efficient Spherical Harmonic Evaluation* (JCGT 2013)의 방식을 그대로 옮겼다.
PyTorch 참조 구현은 `gsplat/cuda/_torch_impl.py`의 `_eval_sh_bases_fast`, CUDA 커널은
`gsplat/cuda/csrc/SphericalHarmonicsCUDA.cu`의 `sh_coeffs_to_color_fast`이다.

### 4.1 구조: $z$만의 다항식 × $(x,y)$의 "방위각 다항식"

구면 좌표 정의에서 $P_\ell^m(\cos\theta)$는 $z$만의 함수, $\cos(m\varphi)$/$\sin(m\varphi)$는 방위각만의 함수다.
직교좌표에서 이것은

- $\sin^m\theta\cos(m\varphi) \to$ `fC_m`, $\quad \sin^m\theta\sin(m\varphi)\to$ `fS_m` (둘 다 $x,y$의 $m$차 다항식)
- $P_\ell^m(z)/\sin^m\theta \to$ `fTmp*` ($z$만의 다항식)

으로 나뉜다. 그리고 `fC_m`, `fS_m`은 복소수 $(x+iy)^m$ 의 실수부·허수부이므로 **한 단계씩 곱해 나가는 점화식**으로 얻는다.

```python
# _eval_sh_bases_fast (발췌)
result[..., 0] = 0.2820947917738781                    # ℓ=0

fTmpA = -0.48860251190292                              # ℓ=1: √(3/4π) 부호 포함
result[..., 2] = -fTmpA * z                            # (1,0)
result[..., 3] =  fTmpA * x                            # (1,1)
result[..., 1] =  fTmpA * y                            # (1,-1)

z2   = z * z                                           # ℓ=2
fTmpB = -1.092548430592079 * z                         # z 다항식
fTmpA =  0.5462742152960395
fC1  = x * x - y * y                                   # Re[(x+iy)^2]
fS1  = 2 * x * y                                       # Im[(x+iy)^2]
result[..., 6] = 0.9461746957575601 * z2 - 0.3153915652525201   # (2,0)  = √(5/16π)(3z²−1)
result[..., 7] = fTmpB * x                             # (2,1)
result[..., 5] = fTmpB * y                             # (2,-1)
result[..., 8] = fTmpA * fC1                           # (2,2)
result[..., 4] = fTmpA * fS1                           # (2,-2)

fTmpC = -2.285228997322329 * z2 + 0.4570457994644658   # ℓ=3
fTmpB =  1.445305721320277 * z
fTmpA = -0.5900435899266435
fC2 = x * fC1 - y * fS1                                # Re[(x+iy)^3]  ← 점화식
fS2 = x * fS1 + y * fC1                                # Im[(x+iy)^3]
result[..., 12] = z * (1.865881662950577 * z2 - 1.119528997770346)  # (3,0)
result[..., 13] = fTmpC * x;  result[..., 11] = fTmpC * y
result[..., 14] = fTmpB * fC1; result[..., 10] = fTmpB * fS1
result[..., 15] = fTmpA * fC2; result[..., 9]  = fTmpA * fS2
```

- `fC{m+1} = x*fC{m} - y*fS{m}`, `fS{m+1} = x*fS{m} + y*fC{m}` 은 복소수 곱 $(x+iy)\cdot(x+iy)^m$ 이다.
  이것이 $\cos((m{+}1)\varphi)$, $\sin((m{+}1)\varphi)$ 의 가법정리를 삼각함수 없이 구현한 것.
- $z$ 다항식(`fTmpA/B/C/D`)은 르장드르 함수를 미리 전개해 계수를 상수로 박아 둔 것.
- 같은 `fC1`, `fS1`이 $\ell=2$의 $(2,\pm2)$와 $\ell=3$의 $(3,\pm2)$, $\ell=4$의 $(4,\pm2)$에서 모두 재사용된다.
- `basis_dim <= 4/9/16`에서 early return하므로 `sh_degree_to_use = min(step // sh_degree_interval, 3)`처럼
  차수를 점진적으로 올리는 학습 스케줄에 그대로 대응한다. 4차(25개)까지 지원.

### 4.2 CUDA 커널: 기저를 만들지 않고 곧바로 색으로 접는다

`SphericalHarmonicsCUDA.cu`의 `sh_coeffs_to_color_fast`는 위와 완전히 같은 다항식·상수를 쓰지만,
기저 배열을 메모리에 저장하지 않고 **기저값 × 계수를 즉시 누적**한다.

```cpp
opmath_t result = 0.2820947917738781f * coeffs[c];
if (degree >= 1) {
    float inorm = rsqrtf(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);   // 정규화: 여기서만 특수 함수 1회
    float x = dir.x * inorm, y = dir.y * inorm, z = dir.z * inorm;
    result += 0.48860251190292f * (-y * coeffs[1*D+c] + z * coeffs[2*D+c] - x * coeffs[3*D+c]);
    if (degree >= 2) {
        float z2 = z * z;
        float fTmp0B = -1.092548430592079f * z;
        float fC1 = x*x - y*y, fS1 = 2.f*x*y;
        float pSH6 = 0.9461746957575601f * z2 - 0.3153915652525201f;
        float pSH7 = fTmp0B * x, pSH5 = fTmp0B * y;
        float pSH8 = 0.5462742152960395f * fC1, pSH4 = 0.5462742152960395f * fS1;
        result += pSH4*coeffs[4*D+c] + pSH5*coeffs[5*D+c] + pSH6*coeffs[6*D+c] + ...;
        if (degree >= 3) { float fC2 = x*fC1 - y*fS1, fS2 = x*fS1 + y*fC1; ... }
    }
}
```

- 스레드 하나가 (Gaussian, 채널) 한 쌍을 맡아 레지스터만으로 끝낸다. 전 과정에서 특수 함수는 `rsqrtf` 한 번뿐이고
  나머지는 컴파일러가 FMA로 묶을 수 있는 곱셈·덧셈이다.
- `if (degree >= k)` 중첩 구조는 `_eval_sh_bases_fast`의 early return과 같은 역할이며,
  워프 내 모든 스레드가 같은 `degree`를 공유하므로 분기 발산도 없다.
- 역전파 커널도 같은 다항식을 미분한 형태(예: $\partial(xy)/\partial x = y$)라서 마찬가지로 곱셈·덧셈만 쓴다.

## 5. 정리

| 항목 | 구면 좌표 $(\theta,\varphi)$ 방식 | 직교좌표 다항식 방식 (실제 코드) |
|---|---|---|
| 입력 | 각도 2개 — 벡터에서 `acos`, `atan2`로 변환 필요 | 정규화된 방향 벡터 $(x,y,z)$ 그대로 |
| 연산 | $\cos m\varphi$, $\sin m\varphi$, 르장드르 $P_\ell^m$ | 곱셈·덧셈(FMA) + 하드코딩 상수 |
| 특이점 | 극($z=\pm1$)에서 $\varphi$ 미정의 | 없음, 구면 전체에서 매끄러움 |
| 차수 확장 | $P_\ell^m$ 재계산 | `fC/fS` 복소수 점화식과 중간값 재사용 |
| gsplat 구현 | — | `_eval_sh_bases_fast` (Sloan 2013), `sh_coeffs_to_color_fast` CUDA 커널 |

한 줄로: **SH의 $\ell$차 기저 = $x,y,z$의 $\ell$차 동차 다항식을 단위구에 제한한 것**이므로,
방향 벡터가 이미 있는 렌더러에서는 각도로 되돌아갈 이유가 없고, 상수를 접어 둔 다항식을 곱셈·덧셈으로 평가하는 것이 정확하고도 가장 빠르다.

## 참고
- 노트북: `.fm/assets/sh_walkthrough.py` 1.2절(정의·다항식 표), `sh_bases`, 5절(gsplat 교차 검증)
- `gsplat/cuda/_torch_impl.py` — `_eval_sh_bases_fast`, `_spherical_harmonics`
- `gsplat/cuda/csrc/SphericalHarmonicsCUDA.cu` — `sh_coeffs_to_color_fast`
- Peter-Pike Sloan, "Efficient Spherical Harmonic Evaluation", JCGT 2(2), 2013. https://jcgt.org/published/0002/02/06/
