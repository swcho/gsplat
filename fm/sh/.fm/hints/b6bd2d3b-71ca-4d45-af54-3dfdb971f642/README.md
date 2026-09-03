# 신경 장면 표현에서 SH를 쓰는 대표적 방법들과 그 장점

> **Q.** 신경 장면 표현에서 SH를 사용하는 대표적인 방법들과 그 장점은?
> **A.** PlenOctrees, Plenoxels, 3DGS가 시점 방향에 따른 색을 SH로 표현한다. MLP 없이 SH 계수만 저장하므로 **실시간 렌더링**이 가능하다.

이 문서는 `sh_walkthrough.py`의 2절("SH의 응용처 — 신경 장면 표현" 행)과 "정리" 표를 바탕으로, NeRF → PlenOctrees → Plenoxels → 3DGS로 이어지는 계보와 그 공통 아이디어를 정리한다.

---

## 1. 문제의 출발점 — NeRF(2020)는 왜 느린가

NeRF(Mildenhall et al., ECCV 2020)는 장면을 하나의 MLP $F_\Theta$로 표현한다.

$$
F_\Theta:\ (\mathbf x,\ \mathbf d)\ \longmapsto\ (\sigma,\ \mathbf c)
$$

- 입력: 3D 위치 $\mathbf x$ **와 시점 방향 $\mathbf d$**
- 출력: 밀도 $\sigma$와 그 방향에서 본 색 $\mathbf c$

색이 방향 $\mathbf d$의 함수이므로 하이라이트·프레넬 같은 **시점 의존 효과**를 표현할 수 있다. 문제는 렌더링 비용이다. 픽셀 하나를 그리려면 카메라 광선을 따라 샘플 점을 (coarse 64 + fine 128 =) 수백 개 뽑고, **샘플마다 MLP를 한 번씩 질의**해야 한다. 800×800 이미지 한 장이면 약 $640{,}000 \times 192 \approx 1.2\times10^8$ 번의 MLP 순전파가 필요하고, 8층 256폭 MLP를 GPU에서 돌려도 **한 장에 수십 초**가 걸린다. 학습도 장면당 1~2일이다.

핵심 병목은 두 가지로 나뉜다.

| 병목 | 원인 |
|---|---|
| 기하(밀도) 조회가 느림 | $\sigma(\mathbf x)$를 얻으려면 매번 MLP 순전파 |
| **외형(색) 조회가 느림** | $\mathbf c(\mathbf x,\mathbf d)$가 방향까지 입력으로 받는 MLP라서, 위치별로 **미리 계산해 캐시할 수 없다** — 방향이 바뀌면 다시 계산해야 하므로 |

밀도는 방향과 무관하니 격자에 미리 저장(bake)하면 된다. 그런데 색은 방향에 따라 달라지므로 "위치 → 색"으로는 표를 만들 수 없다. **"위치마다 방향의 함수를 통째로 저장"**할 방법이 필요하다. 여기서 SH가 등장한다.

---

## 2. 공통 아이디어 — 외형을 "계수 조회 + 내적"으로 바꾼다

노트북 1절이 보여준 것처럼, 구면 위의 부드러운 함수 $f(\mathbf d)$는 소수의 SH 계수로 압축된다.

$$
\mathbf c(\mathbf d) \approx \sum_{\ell=0}^{L}\sum_{m=-\ell}^{\ell} \mathbf c_\ell^m\,Y_\ell^m(\mathbf d)
= \underbrace{[\,Y_0(\mathbf d)\ \cdots\ Y_{K-1}(\mathbf d)\,]}_{1\times K}\ \underbrace{\begin{bmatrix}\mathbf c_0^\top\\ \vdots\\ \mathbf c_{K-1}^\top\end{bmatrix}}_{K\times 3},\qquad K=(L+1)^2
$$

즉 **위치마다 "방향 → 색" 함수 하나를 $K\times3$개의 숫자로 저장**할 수 있다. 렌더링 시점에는

1. 저장된 계수 $K\times 3$개를 **조회**(lookup)하고
2. 방향 $\mathbf d$에서 기저 $Y_k(\mathbf d)$를 다항식 몇 개로 계산한 뒤
3. 계수와 **내적**한다 (채널당 $K$번의 곱셈-덧셈).

MLP는 어디에도 없다. 노트북의 표현을 빌리면, 3DGS의 경우 "뉴럴 네트워크 없이 **덧셈·곱셈 48번**으로 색이 나오므로 수백만 개 Gaussian을 실시간으로 그릴 수 있다." 이것이 아래 세 방법을 하나로 묶는 아이디어이고, 각 방법은 **계수를 어느 자료구조에 얹는가**만 다르다.

| 방법 | 계수를 얹는 자료구조 | 렌더링 방식 |
|---|---|---|
| PlenOctrees (2021) | 희소 **옥트리** 리프 | 광선 행진(ray marching) |
| Plenoxels (2022) | 희소 **복셀 격자**(트라이리니어 보간) | 광선 행진 |
| 3DGS (2023) | **Gaussian 프리미티브** 각각 | 타일 기반 **래스터라이제이션** |

---

## 3. 계보

### 3.1 PlenOctrees (Yu et al., ICCV 2021) — "NeRF를 SH로 굽는다"

**아이디어**: 학습은 여전히 NeRF로 하되, 완성된 모델을 **옥트리에 미리 계산해 넣어** 렌더링에서 MLP를 없앤다.

1. **NeRF-SH 학습**: MLP의 출력을 $(\sigma,\ \mathbf c)$ 대신 **$(\sigma,\ \text{SH 계수 } \mathbf k)$**로 바꾼다. 즉 MLP는 방향 $\mathbf d$를 입력으로 받지 않고, 위치 $\mathbf x$만으로 "그 지점의 방향-색 함수"를 SH 계수로 내놓는다. 색은 $\mathbf c(\mathbf x,\mathbf d) = S\!\left(\sum_k \mathbf k_k(\mathbf x)\,Y_k(\mathbf d)\right)$(S는 시그모이드)로 평가한다.
   - 이렇게 하면 MLP의 출력이 **방향과 무관**해져 위치별로 미리 계산해 표에 넣을 수 있다.
   - 학습 중 빈 공간이 뭉개지지 않도록 sparsity prior를 추가한다.
2. **변환(bake)**: 장면을 격자로 나눠 밀도가 낮은 셀을 잘라내고, 남은 셀을 **희소 옥트리**로 만든 뒤 각 리프에 $\sigma$와 SH 계수(2차, 9개 × 3채널)를 저장한다.
3. **미세조정**: 옥트리 리프 값 자체를 학습 이미지에 대해 직접 미세조정해 변환 손실을 복구한다.

**결과**: 렌더링이 광선 행진 중 **옥트리 조회 + SH 내적**만으로 이뤄져 **800×800에서 150 FPS 이상**(원본 NeRF 대비 3000배 이상). 품질도 NeRF와 동등하거나 약간 우수하다. 웹 브라우저 데모가 유명하다.

**남은 한계**: 학습은 여전히 NeRF(수 시간~하루)에 의존한다. 옥트리는 렌더링용 캐시일 뿐이었다.

### 3.2 Plenoxels (Fridovich-Keil, Yu et al., CVPR 2022) — "MLP를 학습에서도 없앤다"

**아이디어**: PlenOctrees의 미세조정 단계가 잘 작동한다는 것은, 옥트리 값 자체가 학습 가능한 표현이라는 뜻이다. 그렇다면 **처음부터 MLP 없이** 격자 위의 값만 최적화하면 되지 않을까?

- 장면을 **희소 복셀 격자**로 두고, 각 복셀 꼭짓점에 $\sigma$와 SH 계수(2차, 9개 × 3)를 저장한다.
- 광선 위 샘플 점의 값은 인접 8개 꼭짓점의 **트라이리니어 보간**으로 얻는다(연속성과 부드러운 기울기를 위해).
- NeRF와 같은 볼륨 렌더링 식으로 색을 합성하고, 재구성 손실 + **TV(total variation) 정규화**로 격자 값을 직접 경사하강한다.
- coarse-to-fine: 낮은 해상도에서 시작해 빈 복셀을 잘라내고 남은 복셀을 세분화.

**결과**: 학습 시간이 **장면당 약 11분**으로 NeRF 대비 **100배 이상 빠르고**, 품질은 동등하다. 논문의 요지는 "NeRF의 성공 요인은 신경망이 아니라 **미분 가능한 볼륨 렌더링**이었다"는 것이다 — 표현은 단순한 격자 + SH로 충분했다.

**남은 한계**: 격자 해상도가 곧 세부 표현력이라 메모리가 크고(장면당 수 GB), 광선 행진이라 렌더링은 PlenOctrees만큼 빠르지 않다.

### 3.3 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023) — "격자 대신 프리미티브, 행진 대신 래스터"

**아이디어**: 격자는 빈 공간에도 셀을 쓰고 해상도가 고정된다. 대신 장면을 **이방성 3D Gaussian 수백만 개**로 표현하고, 각 Gaussian이 자기 색을 SH로 들고 있게 한다.

Gaussian $i$의 속성: 중심 $\boldsymbol\mu_i$, 공분산(스케일·회전), 불투명도 $\alpha_i$, **SH 계수 $\mathbf c_{i,k}\in\mathbb R^3$, $k=0..15$**(3차, 16개 × 3 = 48개 실수).

$$
\mathbf c_i(\mathbf d) = \max\!\Big(0,\ \sum_{k=0}^{15} \mathbf c_{i,k}\,Y_k(\mathbf d) + 0.5\Big),
\qquad \mathbf d = \frac{\boldsymbol\mu_i - \mathbf o_{\text{cam}}}{\|\boldsymbol\mu_i - \mathbf o_{\text{cam}}\|}
$$

- 방향 $\mathbf d$는 **카메라 위치에서 Gaussian 중심을 보는 방향**이다. 카메라 위치는 world→camera 행렬 $[R\,|\,\mathbf t]$에서 $\mathbf o_{\text{cam}} = -R^\top\mathbf t$로 복원한다(노트북 4절 `view_dirs`).
- SH 평가는 **카메라마다·Gaussian마다 한 번**이다. 광선 위 샘플마다가 아니다 — 광선 행진 자체가 없기 때문이다.
- 색이 정해진 Gaussian은 화면에 **투영(splat)**되어 타일 단위로 정렬·알파 합성된다. 이 래스터라이제이션 파이프라인이 GPU에 매우 잘 맞아 **1080p에서 30~130+ FPS**로 렌더링된다.
- 학습은 SfM 포인트에서 시작해 Gaussian을 분할·복제·삭제(densification)하며 30분~1시간 정도.

**gsplat에서의 구현 요약** (노트북 3~5절):

| 항목 | 내용 |
|---|---|
| 파라미터 | `sh0` (DC, `[N,1,3]`) + `shN` (나머지 15개, `[N,15,3]`). `shN`의 학습률은 1/20 |
| 초기화 | `sh0 = (rgb − 0.5) / C0`, `shN = 0` → 처음엔 모든 방향에서 SfM 색과 같음 |
| 평가 | `spherical_harmonics(sh_degree, means, viewmats, coeffs)` CUDA 커널 하나로 정규화 + 기저 + 내적 |
| 후처리 | `rasterization()`이 `+0.5`, `clamp_min(0)` 적용 |
| 차수 활성화 | `sh_degree_interval=1000` 스텝마다 한 차수씩 켜서 기본색부터 안정적으로 학습 |

---

## 4. 차수 선택 — 왜 2차(9개) 또는 3차(16개)인가

| 방법 | 차수 $L$ | 계수 수 $(L+1)^2$ | 채널 포함 |
|---|---|---|---|
| PlenOctrees | 2 | 9 | 27 |
| Plenoxels | 2 | 9 | 27 |
| 3DGS | 3 | 16 | 48 |

- **2차로 충분한 근거**: Ramamoorthi & Hanrahan(2001)이 확산 반사에 대한 환경 조명은 2차 SH(9개)로 오차 약 1%임을 보였다. 대부분의 표면에서 시점 의존성은 부드러운 저주파이므로 9개로 대부분 잡힌다. 격자/옥트리 방식은 **셀 수가 수백만~수억**이라 셀당 계수 수에 메모리가 민감해 2차에서 멈췄다(PlenOctrees 논문은 3차·4차도 실험했지만 이득 대비 메모리가 컸다).
- **3DGS가 3차를 택한 이유**: Gaussian 하나가 격자 셀보다 훨씬 넓은 영역을 담당하므로 프리미티브 수가 상대적으로 적고(수십만~수백만), 개당 48개 실수를 감당할 수 있다. 3차는 광택 하이라이트의 위치·폭을 좀 더 정확히 표현한다.
- **왜 그 이상은 안 가는가**: 노트북 1.3절의 실험이 답이다. 차수를 올려도 태양처럼 **좁고 날카로운 봉우리는 16개 계수로 흐릿하게 퍼지고 주변에 ringing이 생긴다**. 고주파를 표현하려면 차수를 급격히 올려야 하고($L=7$이면 64개), 그만큼 메모리와 과적합 위험(노트북 4.2절: 관측 뷰가 적으면 고차가 요동)이 커진다. 저주파 한계는 차수를 몇 개 올려서 해결되는 문제가 아니라 SH 표현 자체의 성질이다.

---

## 5. 대안과의 비교 — 모두가 SH를 쓰는 것은 아니다

NeRF를 빠르게 만드는 다른 계열은 **MLP를 없애는 대신 작게 만드는** 방향을 택했다.

| 방법 | 기하 표현 | 시점 의존 색 | 특징 |
|---|---|---|---|
| **Instant-NGP** (Müller et al., 2022) | 다해상도 **해시 격자** 특징 벡터 | 작은 MLP (2층 64폭)에 방향을 **SH 기저값으로 인코딩**해 입력 | 학습 수 초~분. 방향의 SH는 "색 계수"가 아니라 **입력 인코딩**으로만 쓰임 |
| **TensoRF** (Chen et al., 2022) | 텐서 분해(VM) 격자 | SH 또는 작은 MLP 디코더 (둘 다 제공) | 메모리 효율. SH 버전은 Plenoxels류, MLP 버전이 조금 더 좋음 |
| **Mip-NeRF 360 / Zip-NeRF** (Barron et al., 2022/2023) | 큰 MLP(+해시 격자) | MLP | 안티에일리어싱·품질 최우선. 렌더링 실시간 아님 |
| **Ref-NeRF** (Verbin et al., 2022) | MLP | 반사 방향 $\boldsymbol\omega_r$을 **IDE(integrated directional encoding)**로 인코딩해 MLP | 거울 반사 같은 고주파 시점 의존성을 표현 |
| **PlenOctrees / Plenoxels / 3DGS** | 옥트리 / 복셀 / Gaussian | **SH 계수 직접 저장** | MLP 완전 제거, 실시간 |

구분의 핵심은 **"방향을 어디서 처리하는가"**다.
- SH 계열: 방향 처리가 **저장된 계수와의 내적**으로 끝난다. 병렬·실시간에 최적이지만 표현력이 저차 SH에 묶인다.
- 작은-MLP 계열: 방향(과 위치 특징)을 MLP에 넣는다. 날카로운 반사도 학습할 수 있고 메모리가 작지만, 픽셀/샘플마다 MLP 순전파가 남아 SH 계열만큼 빠르지 않다.

3DGS 이후의 후속 연구도 이 경계를 넘나든다. 노트북 "정리" 절이 언급하듯, 날카로운 반사·굴절을 위해 **SH 대신 작은 MLP 디코더, 구면 가우시안(Spherical Gaussians), 또는 반사 방향 기반 인코딩**을 Gaussian에 붙이는 시도들이 있다(예: Scaffold-GS의 앵커별 MLP, GaussianShader/3DGS-DR의 반사 모델링). 반대로 Compact-3DGS류는 SH 계수를 벡터 양자화해 메모리를 더 줄인다.

---

## 6. 장점과 단점 정리

### 장점

| 장점 | 설명 |
|---|---|
| **실시간 렌더링** | 색 계산이 MLP 순전파에서 $K\times3$ 곱셈-덧셈으로 바뀐다. PlenOctrees 150 FPS, 3DGS 100+ FPS |
| **병렬 친화** | 각 위치/프리미티브의 색 평가가 서로 독립적이고 분기가 없어 GPU 스레드·CUDA 커널(gsplat의 `SphericalHarmonicsCUDA.cu`)에 그대로 대응된다 |
| **미분 가능** | 색이 계수에 대해 **선형**이다($\partial \mathbf c/\partial \mathbf c_k = Y_k(\mathbf d)$). 기울기가 단순하고 안정적이며, 노트북 4.2절처럼 최소제곱 닫힌 해와 Adam 결과가 거의 일치한다 |
| **해석 가능** | DC 계수 $c_0 Y_0^0$은 **구면 평균 = 시점 무관 기본색**, 고차 계수는 평균 0의 시점 의존 변동이다. 그래서 `(rgb − 0.5)/C0` 초기화, `shN` 학습률 1/20, 차수 점진 활성화 같은 직관적인 학습 전략이 가능하다 |
| **학습이 빠름(Plenoxels 이후)** | MLP가 없으니 역전파도 조회·내적의 역이다. Plenoxels는 NeRF 대비 100배, 3DGS는 수십 분 |
| **압축·전송에 유리** | 계수는 고정 길이 벡터이므로 양자화·코드북 압축이 자연스럽다 |

### 단점

| 단점 | 설명 |
|---|---|
| **메모리** | 위치/프리미티브마다 27~48개 실수를 저장한다. 3DGS 모델 크기(수백 MB~GB)의 상당 부분이 SH 계수이며, Plenoxels 격자는 수 GB에 이른다. MLP는 장면 전체를 수 MB로 압축하는 반면 SH 표는 장면 크기에 비례해 커진다 |
| **저주파 한계** | 노트북 1.3절 실험 그대로 — 좁은 하이라이트·거울 반사·굴절처럼 **방향에 따라 급격히 변하는 색**은 16개 계수로 표현 못 하고 흐려지며 ringing이 생긴다. 이것이 3DGS가 반사면을 잘 못 그리는 이유다 |
| **과적합** | 관측 카메라가 적은 영역에서 고차 계수가 관측 사이의 방향으로 요동할 수 있다(노트북 4.2절 표: n=8일 때 L=3의 MSE가 L=1보다 큼). 차수 점진 활성화와 낮은 학습률이 완화책이다 |
| **시점 방향의 근사** | 3DGS는 Gaussian **중심**을 향한 방향 하나로 Gaussian 전체의 색을 정한다. 화면에서 크게 보이는 Gaussian은 내부에서 방향이 달라져도 같은 색이다 |
| **물리적 의미 없음** | SH 계수는 BRDF·조명을 분리하지 않는 "결과 색"의 근사다. 재조명(relighting)이나 재질 편집에는 부적합하다 |

---

## 7. 한 줄 요약

> NeRF가 "위치+방향 → 색"을 **MLP에 묻던** 것을, PlenOctrees(옥트리) → Plenoxels(복셀) → 3DGS(Gaussian)는 **위치마다 SH 계수를 저장하고 방향 기저와 내적**하는 것으로 바꿨다. 계산이 조회+곱셈-덧셈 몇십 번으로 줄어 **실시간 렌더링·병렬화·단순한 미분·직관적 해석**이 가능해졌고, 그 대가로 **메모리**와 **저주파(부드러운 시점 의존성)까지만 표현되는 한계**를 얻었다.

## 참고 문헌

- Mildenhall et al., *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*, ECCV 2020.
- Yu, Li, Tancik, Li, Ng, Kanazawa, *PlenOctrees for Real-time Rendering of Neural Radiance Fields*, ICCV 2021.
- Fridovich-Keil, Yu, Tancik, Chen, Recht, Kanazawa, *Plenoxels: Radiance Fields without Neural Networks*, CVPR 2022.
- Kerbl, Kopanas, Leimkühler, Drettakis, *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, SIGGRAPH 2023.
- Müller, Evans, Schied, Keller, *Instant Neural Graphics Primitives with a Multiresolution Hash Encoding*, SIGGRAPH 2022.
- Verbin et al., *Ref-NeRF*, CVPR 2022; Barron et al., *Zip-NeRF*, ICCV 2023.
- Ramamoorthi & Hanrahan, *An Efficient Representation for Irradiance Environment Maps*, SIGGRAPH 2001.
- 이 카드의 원천: `sh_walkthrough.py` 2절(응용처 표), 3절(DC 계수), 4절(SH 평가), 정리 표.
