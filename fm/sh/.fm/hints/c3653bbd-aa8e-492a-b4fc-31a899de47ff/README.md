# SH는 1차원 신호 처리의 어떤 개념에 대응되는가 — 푸리에 급수

**질문**: Spherical Harmonics(SH)는 1차원 신호 처리의 어떤 개념에 대응되는가?

**답**: 1차원 주기 함수의 **푸리에 급수**에 대응한다. 사인·코사인이 주기 함수의 기저 역할을 하듯, SH $Y_\ell^m$은 구면 위에 정의된 함수 $f(\mathbf d)$의 정규직교 기저 역할을 한다.

---

## 1. 대응 관계 한눈에 보기

| | 1차원 푸리에 급수 | Spherical Harmonics |
|---|---|---|
| 정의역 | 원 $S^1$ (주기 $2\pi$인 각 $\varphi$) | 구면 $S^2$ (단위 방향 벡터 $\mathbf d$, $\|\mathbf d\|=1$) |
| 기저 함수 | $1,\ \cos n\varphi,\ \sin n\varphi$ | $Y_\ell^m(\mathbf d)$, $\ell\ge 0$, $-\ell\le m\le\ell$ |
| "주파수" 인덱스 | $n = 0,1,2,\dots$ | 차수(degree, band) $\ell = 0,1,2,\dots$ |
| 주파수 $n$(또는 $\ell$)당 기저 개수 | 2개 ($\cos, \sin$; $n=0$은 1개) | $2\ell+1$개 |
| 차수 $L$까지 총 기저 수 | $2L+1$ | $(L+1)^2$ (3DGS의 $L=3$ → 16개) |
| 전개식 | $f(\varphi)=\sum_n a_n\cos n\varphi + b_n \sin n\varphi$ | $f(\mathbf d)=\sum_{\ell}\sum_{m} c_\ell^m Y_\ell^m(\mathbf d)$ |
| 정규직교성 | $\frac1\pi\int_0^{2\pi}\cos n\varphi\cos k\varphi\,d\varphi=\delta_{nk}$ | $\int_{S^2} Y_\ell^m Y_{\ell'}^{m'}\,d\Omega=\delta_{\ell\ell'}\delta_{mm'}$ |
| 계수 구하기(사영) | $a_n=\frac1\pi\int f\cos n\varphi\,d\varphi$ | $c_\ell^m=\int_{S^2} f\,Y_\ell^m\,d\Omega$ |
| 0번째 항의 뜻 | $a_0/2$ = 함수의 평균 (DC 성분) | $c_0^0 Y_0^0$ = 함수의 구면 평균 (DC 계수) |
| 절단(truncation)의 효과 | 저역 통과 필터 → 부드러운 근사, 날카로운 곳에 링잉(Gibbs) | 저역 통과 필터 → 부드러운 근사, 날카로운 봉우리 주변에 링잉 |
| 미분방정식과의 관계 | $\frac{d^2}{d\varphi^2}$의 고유함수 | 구면 라플라시안 $\Delta_{S^2}$의 고유함수 (고유값 $-\ell(\ell+1)$) |

핵심은 **"부드러운 함수를 유한 개의 직교 기저 계수로 압축한다"**는 아이디어가 원(1차원 주기)에서 구면(2차원 곡면)으로 그대로 옮겨진 것이라는 점이다.

## 2. 왜 "대응"인가 — 세 가지 공통 성질

### 2.1 정규직교 기저
푸리에 급수의 편리함은 사인·코사인이 서로 직교한다는 데서 나온다. 직교하면 각 계수를 **내적 한 번**으로 독립적으로 뽑아낼 수 있고, 계수 하나를 바꿔도 다른 항에 영향을 주지 않는다.

SH도 정확히 같다. 구면 전체에 대한 적분 $d\Omega=\sin\theta\,d\theta\,d\varphi$에 대해

$$
\int_{S^2} Y_\ell^m(\mathbf d)\,Y_{\ell'}^{m'}(\mathbf d)\,d\Omega=\delta_{\ell\ell'}\,\delta_{mm'}
$$

이므로 계수는 $c_\ell^m=\int_{S^2} f\,Y_\ell^m\,d\Omega$ 로 바로 구해진다. 노트북(`sh_walkthrough.py` §1)은 이것을 격자 구적으로 확인한다: 16개 기저의 Gram 행렬 $\int Y_iY_j\,d\Omega$를 계산하면 단위행렬과의 차이가 수치 오차 수준이다.

### 2.2 "주파수"의 개념
1차원에서 $n$이 크면 $\cos n\varphi$가 한 주기 안에서 더 많이 진동한다. 구면에서는 $\ell$이 그 역할을 한다. $\ell$차 SH는 **$x,y,z$의 $\ell$차 동차 다항식을 단위구에 제한한 것**이다.

- $\ell=0$: $Y_0^0=\frac{1}{2\sqrt\pi}$ — 상수 (방향 무관)
- $\ell=1$: $\pm\sqrt{3/4\pi}\,x,\ y,\ z$ — 방향 성분 그 자체 (한 번 부호가 바뀜)
- $\ell=2$: $xy,\ yz,\ 3z^2-1,\ xz,\ x^2-y^2$ 에 상수를 곱한 것 — 두 번 진동
- $\ell=3$: 3차식 — 더 잘게 진동

노트북의 등장방형 지도 그림(4행 7열)에서 아래 행(높은 $\ell$)으로 갈수록 빨강·파랑 무늬가 잦아지는 것이 그대로 "고주파"다. 주파수 개념이 이렇게 옮겨지기 때문에 우주론에서는 CMB 온도 지도의 **각도 파워 스펙트럼 $C_\ell$** 처럼 1차원 파워 스펙트럼과 똑같은 방식으로 SH 계수를 분석한다.

### 2.3 절단 = 저역 통과 필터
푸리에 급수를 $n\le N$에서 끊으면 원 함수의 저주파 성분만 남는다(부드러워지고, 불연속점 근처에서 Gibbs 링잉이 생긴다). SH를 $\ell\le L$에서 끊는 것도 완전히 같은 효과다. 노트북 §1.3의 실험이 이를 보여준다: 하늘의 부드러운 그라디언트는 $L=1$ 정도에서 이미 잘 맞지만, 태양처럼 좁고 강한 봉우리는 $L=3$(16개 계수)로도 흐릿하게 퍼지며 주변에 음의 물결(링잉)이 생긴다.

이 성질이 3DGS 설계와 직결된다. 3DGS가 **3차(16개)에서 멈추는** 것은 부드러운 시점 의존성(광택, 프레넬)까지는 소수의 계수로 충분하지만 거울 반사 같은 고주파는 차수를 급격히 올려야 하기 때문이고, 그래서 3DGS가 날카로운 반사를 잘 못 그린다.

## 3. 차이점도 알아두기

대응이 완벽한 동형은 아니다. 몇 가지 다른 점:

- **기저 개수의 증가 속도**: 1차원은 주파수 $n$당 2개(총 $2L+1$), 구면은 $\ell$당 $2\ell+1$개(총 $(L+1)^2$)로 제곱으로 늘어난다. 그래서 고주파를 표현하는 비용이 구면에서 훨씬 빨리 커진다.
- **좌표 의존성**: 구면에는 원처럼 자연스러운 "하나의 각"이 없다. $\theta,\varphi$로 쓰면 $\varphi$ 방향은 $\cos m\varphi,\sin m\varphi$ 그대로 푸리에 급수이고, $\theta$ 방향은 연관 르장드르 함수 $P_\ell^m(\cos\theta)$가 담당한다. 실제로 실수형 SH의 정의식

  $$
  Y_\ell^m(\theta,\varphi)=
  \begin{cases}
  \sqrt2\,K_\ell^{m}\cos(m\varphi)\,P_\ell^{m}(\cos\theta) & m>0\\
  K_\ell^0\,P_\ell^0(\cos\theta) & m=0\\
  \sqrt2\,K_\ell^{|m|}\sin(|m|\varphi)\,P_\ell^{|m|}(\cos\theta) & m<0
  \end{cases}
  $$

  을 보면 $\varphi$ 부분에 **푸리에 급수의 $\cos, \sin$이 문자 그대로 들어 있다**. 즉 SH는 "위도 방향으로 르장드르, 경도 방향으로 푸리에"라고 볼 수도 있다.
- **회전 불변성**: 1차원에서 신호를 평행이동하면 같은 주파수 $n$ 안에서 $\cos\leftrightarrow\sin$ 계수가 섞인다. 구면에서 함수를 회전하면 같은 차수 $\ell$ 안의 $2\ell+1$개 계수가 서로 섞이되 다른 차수와는 섞이지 않는다. 그래서 각 차수의 에너지 $\sum_m (c_\ell^m)^2$는 회전에 불변이다.
- **복소 vs 실수**: 푸리에에 $e^{in\varphi}$(복소)와 $\cos,\sin$(실수) 두 표현이 있듯 SH에도 복소형과 실수형이 있다. 그래픽스와 3DGS는 실수형을 쓴다.

## 4. 3DGS·gsplat에서 이 대응이 어떻게 쓰이는가

노트북 §2~§4의 내용을 대응 관계 관점에서 다시 읽으면:

- **저장**: Gaussian마다 색을 RGB 하나가 아닌 **SH 계수 16개 × 3채널 = 48개 실수**로 저장한다. 이는 "방향에 따른 색 함수 $\mathbf c(\mathbf d)$"를 3차까지의 푸리에 계수로 압축해 둔 것이다.
- **DC 계수**: 신호처리의 DC(direct current, 주파수 0) 용어를 그대로 가져와 $\ell=0$ 항을 부른다. $c_0^0Y_0^0$은 구면 평균, 즉 시점 무관 기본색이며, 초기값 `sh0 = (rgb − 0.5) / C0` ($C_0 = Y_0^0 \approx 0.2821$)가 여기서 나온다. gsplat은 파라미터를 `sh0`(DC)와 `shN`(나머지 15개)으로 나누고 `shN`의 학습률을 1/20로 둔다.
- **SH 평가**: 계수와 방향이 주어졌을 때 급수를 실제로 더하는 일 — 푸리에 급수의 "합성(synthesis)"에 해당한다. 방향 $\mathbf d$에서 기저 16개를 다항식으로 계산한 뒤 계수와 내적하면 끝이므로, MLP 없이 곱셈-덧셈 48번으로 색이 나온다. gsplat의 `spherical_harmonics(sh_degree, means, viewmats, coeffs)`가 이를 CUDA 커널 하나로 처리하고 `rasterization()`이 `+0.5`, `clamp_min(0)`을 적용한다.
- **차수 점진 활성화**: `sh_degree_interval`(기본 1000스텝)마다 사용 차수를 하나씩 올린다. 저주파(기본색)를 먼저 잡고 고주파(시점 의존성)를 나중에 배우는, 신호처리의 coarse-to-fine 전략과 같다.
- **계수 획득**: 3DGS는 진짜 $f$를 모르므로 적분(분석, analysis)으로 계수를 구하지 않고, 여러 카메라의 관측과 렌더 결과의 차이를 역전파해 계수를 맞춘다. 노트북 §4.2는 최소제곱 해와 Adam 경사하강 두 방법으로 이를 축소 재현하며, 관측이 적으면 고차가 과적합하고 차수가 낮으면 하이라이트를 표현 못 한다는 것을 보여준다.

## 5. 한 줄 요약

> 원 위의 함수를 $\cos n\varphi,\sin n\varphi$로 펼치는 것이 푸리에 급수라면, **구면 위의 함수를 $Y_\ell^m$으로 펼치는 것이 SH 전개**다. 직교성(내적 한 번으로 계수), 주파수 개념($\ell$), 절단 = 저역 통과 필터라는 세 성질이 그대로 이어지며, 3DGS는 이 성질을 이용해 "보는 방향에 따른 색"을 16개 계수로 압축한다.

## 참고
- 소스 노트북: `fm/sh/.fm/assets/sh_walkthrough.py` §1 (정의·정규직교성 검증·저역 통과 실험), §3 (DC 계수), §4 (SH 평가)
- Ramamoorthi & Hanrahan (2001), *An Efficient Representation for Irradiance Environment Maps* — 확산 조명은 2차(9개 계수)로 오차 ~1%
- Sloan (2013), *Efficient Spherical Harmonic Evaluation* — gsplat `_eval_sh_bases_fast`가 쓰는 점화식
- Kerbl et al. (2023), *3D Gaussian Splatting for Real-Time Radiance Field Rendering*
