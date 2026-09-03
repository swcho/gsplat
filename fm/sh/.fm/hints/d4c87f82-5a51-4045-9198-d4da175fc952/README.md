# 실제 물체의 색을 RGB 하나로 표현할 수 없는 이유와 3DGS의 해결책

**한 줄 답**: 물체의 색은 "보는 방향"의 함수다 — 하이라이트(광택), 프레넬 효과, 반투명/서브서피스 산란 때문에 같은 점이라도 카메라 위치에 따라 다른 색으로 찍힌다. 그래서 시점 무관 RGB 하나로는 여러 카메라의 관측을 동시에 맞출 수 없다. 3DGS는 Gaussian마다 **SH 계수 $\mathbf c_{i,k}\in\mathbb R^3$ 16개(= 48개 실수)** 를 학습해, 색을 방향 $\mathbf d$의 함수 $\mathbf c_i(\mathbf d)=\max\!\big(0,\sum_{k=0}^{15}\mathbf c_{i,k}Y_k(\mathbf d)+0.5\big)$ 로 표현한다.

---

## 1. 물리적 배경 — 색은 왜 방향에 따라 달라지는가

카메라 픽셀에 들어오는 값은 표면 한 점 $\mathbf x$에서 카메라 방향 $\boldsymbol\omega_o$로 나가는 **출사 radiance** $L_o(\mathbf x,\boldsymbol\omega_o)$다. 렌더링 방정식은 이를

$$
L_o(\mathbf x,\boldsymbol\omega_o)=\int_{\Omega} f_r(\mathbf x,\boldsymbol\omega_i,\boldsymbol\omega_o)\;L_i(\mathbf x,\boldsymbol\omega_i)\;(\mathbf n\cdot\boldsymbol\omega_i)\,d\boldsymbol\omega_i
$$

로 쓴다. 여기서 $f_r$이 **BRDF**(Bidirectional Reflectance Distribution Function): "입사 방향 $\boldsymbol\omega_i$에서 들어온 빛이 출사 방향 $\boldsymbol\omega_o$로 얼마나 반사되는가". BRDF가 $\boldsymbol\omega_o$에 의존하는 순간, 같은 점의 색은 카메라 위치마다 달라진다.

| 성분 | 시점 의존? | 왜 그런가 |
|---|---|---|
| **확산(diffuse, Lambertian)** | 아니오 | $f_r$이 상수 $\rho/\pi$. 빛이 표면 안으로 스며들어 모든 방향으로 균일하게 재방출. 분필, 무광 벽지. |
| **광택(specular) 하이라이트** | 예 | 반사 방향 $\mathbf r=\text{reflect}(\boldsymbol\omega_i,\mathbf n)$ 근처의 좁은 로브. 카메라가 그 로브 안에 들어오면 밝은 점이 보이고, 조금 벗어나면 사라진다. 광원 이미지가 표면 위를 "미끄러진다". |
| **프레넬(Fresnel) 효과** | 예 | 반사율이 시선과 법선의 각도에 따라 변한다. 정면($\theta\approx0$)에서는 유전체 반사율이 수 %(물 ~2%, 유리 ~4%)지만, **grazing angle**($\theta\to90°$)에 가까워지면 100%에 접근. 호수를 발밑에서 보면 바닥이 보이고 멀리 보면 하늘이 비치는 이유. Schlick 근사: $F(\theta)\approx F_0+(1-F_0)(1-\cos\theta)^5$. |
| **반투명 / 서브서피스 산란** | 예 | 빛이 표면 아래로 들어가 다른 지점에서 나온다(피부, 대리석, 잎, 왁스). 역광이면 밝게 비쳐 보이고 순광이면 그렇지 않다. BRDF가 아닌 BSSRDF로만 기술되는 성분. |
| 그 밖에 | 예 | 이방성 반사(브러시드 메탈, 머리카락), 회절/간섭(CD, 비단), 재귀반사(표지판) 등 |

즉 **완전한 Lambertian 물체만** RGB 한 값으로 요약할 수 있다. 현실의 거의 모든 표면은 "확산 + 약간의 광택 + 프레넬"을 갖는다.

## 2. "시점 무관 RGB 하나"가 사진 재구성에서 남기는 문제

다중 시점 사진으로 장면을 복원한다고 하자. 장면의 한 점을 카메라 $A$, $B$, $C$가 각각 다른 각도에서 찍었다.

- 그 점이 하이라이트 위에 있으면 $A$에서는 거의 흰색, $B$·$C$에서는 원래 색으로 기록된다.
- 모델이 그 점에 RGB 하나만 부여할 수 있다면, 세 관측을 동시에 만족하는 값은 없다. 손실을 최소화하는 답은 **세 관측의 평균**이고, 그 결과 $A$의 렌더는 너무 어둡고 $B$·$C$의 렌더는 너무 밝다.
- 학습은 이 잔여 오차를 없애려고 **잘못된 방향으로** 노력한다: 하이라이트를 표현하려고 표면 앞쪽에 반투명한 흰 Gaussian을 "떠 있게" 만들거나(floater), 위치·크기를 뒤틀어 특정 시점에서만 맞는 기하를 만든다. 결과적으로 **관측 간 불일치가 기하 오류로 전이**된다.

전통 사진측량(Structure-from-Motion, MVS)의 색 일관성(photo-consistency) 기준도 Lambertian 가정 위에 세워져 있어, 광택·유리·물에서 매칭이 깨지는 것이 같은 뿌리다. 이 불일치를 없애려면 모델이 색을 **방향의 함수**로 가져야 한다.

## 3. 계보 — NeRF는 시점 방향을 입력으로 넣어 해결했다

NeRF(2020)는 장면을 $F_\Theta:(\mathbf x,\mathbf d)\mapsto(\sigma,\mathbf c)$ 인 MLP로 두었다. 밀도 $\sigma$는 위치 $\mathbf x$만의 함수지만, **색 $\mathbf c$는 위치와 시점 방향 $\mathbf d$의 함수**다(네트워크 뒷단에서 방향을 이어붙임). 논문의 ablation은 방향 입력을 빼면 광택 있는 물체(배, 드럼)에서 하이라이트가 사라지고 PSNR이 떨어짐을 보인다. 이 설계가 "시점 의존 색"을 신경 장면 표현의 표준 요소로 만들었다.

그 후 MLP 없이 빠르게 렌더링하려는 시도가 방향 함수를 **SH 계수**로 대체했다.

- **PlenOctrees**(2021): 학습된 NeRF를 옥트리에 굽되, 각 잎에 RGB 대신 SH 계수를 저장(2차 9개 또는 3차 16개). 방향 함수를 MLP 평가 없이 덧셈·곱셈으로 얻는다.
- **Plenoxels**(2022): 처음부터 MLP 없이 복셀 격자 + SH 계수를 직접 최적화.
- **3DGS**(2023): 같은 아이디어를 복셀 대신 **Gaussian 프리미티브**에 붙였다. Gaussian마다 3차 SH 16개 × RGB.

핵심 관찰: 광택·프레넬 같은 시점 의존성은 방향에 따라 **부드럽게** 변하므로, 구면 위 함수의 저주파 기저인 SH 소수 항으로 충분히 잡힌다(Ramamoorthi & Hanrahan 2001은 확산 조명이 2차 9개로 ~1% 오차임을 보였다).

## 4. 3DGS의 선택 — "조명 × BRDF"를 합쳐 outgoing radiance를 SH에 굽는다

### 4.1 무엇을 학습하는가

3DGS는 BRDF와 조명을 **따로** 복원하지 않는다. 촬영 당시 조명이 고정되어 있다고 보고, 렌더링 방정식의 적분 결과인 **출사 radiance 자체**를 방향 함수로 저장한다.

$$
\mathbf c_i(\mathbf d)\;\approx\;L_o(\boldsymbol\mu_i,\ -\mathbf d)
\quad\text{를}\quad
\mathbf c_i(\mathbf d)=\max\!\Big(0,\ \sum_{k=0}^{15}\mathbf c_{i,k}\,Y_k(\mathbf d)+0.5\Big),
\qquad
\mathbf d=\frac{\boldsymbol\mu_i-\mathbf o_{\text{cam}}}{\|\boldsymbol\mu_i-\mathbf o_{\text{cam}}\|}
$$

로 근사한다. $\mathbf d$는 카메라 중심에서 Gaussian 중심을 향하는 단위 벡터이고, $Y_k$는 $k=\ell^2+\ell+m$ 순서의 실수 SH 기저($\ell\le3$, 16개)다. 학습 파라미터는 계수 $\mathbf c_{i,k}\in\mathbb R^3$, Gaussian당 $16\times3=48$개 실수.

- **DC 계수** $\mathbf c_{i,0}$: $Y_0^0=1/(2\sqrt\pi)\approx0.2821$ 은 상수이므로 $\mathbf c_{i,0}Y_0^0$은 방향 무관 "기본색"(구면 평균). 초기화는 SfM 포인트 색으로 `sh0 = (rgb − 0.5) / C0`.
- **고차 계수** $\mathbf c_{i,1..15}$: 평균 0인 방향별 변동, 즉 하이라이트·프레넬·반투명 효과. 0으로 초기화하고 학습률을 DC의 1/20로 두어 기본색이 잡힌 뒤 천천히 배운다. `sh_degree_interval`(1000스텝)마다 한 차수씩 활성화.
- `+0.5`는 계수가 전부 0일 때 색이 검정 대신 중간 회색이 되게 하는 오프셋, `max(0,·)`은 음수 색 방지.

Gaussian 하나의 색이 **덧셈·곱셈 48번**으로 나오므로 신경망 없이 수백만 개를 실시간으로 그릴 수 있다 — 이것이 MLP(NeRF) 대신 SH를 고른 결정적 이유다.

### 4.2 이 선택의 대가 — 조명이 바뀌면 다시 학습해야 한다 (relighting 불가)

SH에 구워진 것은 "이 조명 아래에서 이 방향으로 보이는 색"이라는 **최종 결과물**이다. 확산 반사율(albedo), 광택 로브의 폭(roughness), 프레넬 $F_0$, 조명 방향·색이 모두 하나의 방향 함수로 뒤섞여 있어 분리할 수 없다.

- 조명을 바꾸거나(해가 이동, 램프를 끔), 물체를 다른 장면에 옮기면 SH 계수는 틀린 답을 낸다. 촬영 세트를 새 조명에서 다시 찍어 재학습해야 한다.
- 촬영 도중 조명·노출이 변하면(야외, 자동 노출) 학습 자체가 흔들린다. 이를 다루는 후속 연구들이 카메라별 외관 임베딩(appearance embedding)을 추가한다.
- 역렌더링 계열(Relightable 3DGS, GaussianShader, GS-IR 등)은 Gaussian마다 법선·albedo·roughness를 두고 BRDF와 환경 조명을 분리 추정해 relighting을 가능하게 하지만, 최적화가 더 어렵고 느리다. 원본 3DGS는 사진 재현(novel view synthesis) 품질과 속도를 위해 이 분리를 포기했다.

### 4.3 SH 16개의 표현 한계 — "부드러운 광택까지"

SH를 3차까지만 쓰는 것은 **구면 위 저역 통과 필터**를 거는 것과 같다. 노트북의 실험(하늘 + 좁은 태양 환경맵을 $L=0..3$으로 사영·복원)에서 보듯,

- 하늘의 완만한 그라디언트는 $L=1$~$2$에서 거의 맞는다.
- 태양처럼 **좁고 날카로운 봉우리는 16개 계수로 흐릿하게 퍼지고, 주변에 음의 물결(ringing)** 이 생긴다.
- 시점 의존 색 피팅 실험(기본색 + $(\mathbf d\cdot\mathbf h)^8$ 광택 로브)에서는 $L=3$이 넓은 광택은 재현하지만, 관측 뷰가 적으면 고차가 **과적합**(뷰 사이에서 색이 요동)하고 차수가 낮으면 하이라이트를 **표현 못 한다**.

따라서 3DGS의 SH는 넓은 플라스틱 광택, 프레넬에 의한 가장자리 밝아짐, 부드러운 반투명 정도까지 잡는다. 반면

- **거울·크롬·매끈한 유리**: 반사 방향으로 급격히 변하는 고주파 함수 — 16개로는 흐릿한 얼룩이 되거나, 학습이 반사상을 표면 뒤편의 가짜 Gaussian(floater)으로 "조각"해 버린다.
- **굴절**: 시선이 물체를 통과하며 굴절되므로 Gaussian 중심 방향 $\mathbf d$의 함수로는 원리적으로 표현이 안 된다.

그래서 후속 연구는 SH 대신 작은 MLP 시점 의존 디코더(Scaffold-GS 등), 구면 가우시안(Spherical Gaussians, 좁은 로브를 몇 개의 파라미터로), 반사 방향 기반 인코딩(Ref-NeRF 계열: 시점 대신 반사 벡터를 입력으로 넣어 하이라이트를 저주파로 만듦)을 채택한다. 또 3DGS가 $\mathbf d$를 **Gaussian 중심**으로 계산한다는 점(면적이 큰 Gaussian 내부에서 방향 변화를 무시)도 정밀한 하이라이트 위치를 흐리는 요인이다.

---

## 정리

1. **왜 RGB 하나로 안 되나**: 색 = 출사 radiance $L_o(\mathbf x,\boldsymbol\omega_o)$이고, BRDF의 광택·프레넬·서브서피스 성분이 $\boldsymbol\omega_o$에 의존하므로 카메라마다 다른 색이 찍힌다. RGB 하나(Lambertian 가정)는 관측들의 평균만 맞추고 나머지 오차가 기하 오류·floater로 새어 나간다.
2. **NeRF의 해법**: 색 네트워크에 시점 방향 $\mathbf d$를 입력. PlenOctrees/Plenoxels가 이를 SH 계수로 대체해 MLP를 제거.
3. **3DGS의 해법**: Gaussian마다 3차 SH 계수 $\mathbf c_{i,k}\in\mathbb R^3$ 16개(48개 실수)를 학습해 $\mathbf c_i(\mathbf d)=\max(0,\sum_k\mathbf c_{i,k}Y_k(\mathbf d)+0.5)$. DC = 기본색, 고차 = 방향별 변동. 계산이 덧셈·곱셈 48번이라 실시간.
4. **대가**: 고정 조명에서 "조명 × BRDF"를 합친 결과를 굽기 때문에 relighting 불가, 조명 바뀌면 재학습. 16개 SH는 저주파라 부드러운 광택·프레넬까지만, 거울 반사·굴절은 표현 불가.
