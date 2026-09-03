# Precomputed Radiance Transfer(PRT)에서 SH는 어떻게 쓰이는가?

> **답**: 물체 표면의 방향별 반사·가림 **전달 함수**를 SH 계수로 미리 저장한다.
> 런타임에는 **조명의 SH 계수**와 **전달 SH 계수**의 **내적** 한 번이 곧 그 점의 밝기가 된다.

출처: Sloan, Kautz, Snyder, *"Precomputed Radiance Transfer for Real-Time Rendering in Dynamic, Low-Frequency Lighting Environments"*, SIGGRAPH 2002.
`sh_walkthrough.py` 2절 응용처 표의 한 줄 — "조명 SH · 전달 SH의 내적 = 한 점의 밝기" — 를 풀어 쓴 것이다.

---

## 1. 문제 설정: 렌더링 방정식에서 출발

표면 위 한 점 $p$에서 방향 $\omega_o$로 나가는 빛(출사 복사휘도)은 렌더링 방정식으로 쓴다.

$$
L_o(p,\omega_o)=\int_{S^2} f_r(p,\omega_i\to\omega_o)\;L_i(p,\omega_i)\;\max(0,\mathbf n_p\!\cdot\!\omega_i)\;d\omega_i
$$

- $f_r$: BRDF(재질), $L_i$: 입사광, 코사인 항: 비스듬히 들어오는 빛은 면적당 약해진다.
- 실시간으로 이 적분을 매 프레임·매 픽셀 계산하는 것은 불가능하다. 특히 $L_i(p,\omega_i)$는 다른 표면에 **가려지거나**(self-shadow) 다른 표면에서 **튕겨 들어온**(상호반사) 빛을 포함하므로 전역 조명 문제다.

PRT의 핵심 가정 두 가지:

1. **조명은 멀리 있다(distant lighting)**: 환경맵 $L(\omega)$ 하나가 모든 점을 비춘다. 즉 $L_i$의 "원천"은 위치 $p$에 무관하다.
2. **물체는 강체(형상 고정)**: 물체 자기 자신이 만드는 가림·상호반사는 조명과 무관하게 형상만의 성질이다.

그러면 점 $p$의 밝기는 "**조명**"(매 프레임 바뀔 수 있음)과 "**물체가 그 조명을 어떻게 받아 내보내는가**"(형상·재질에만 의존, 고정)로 **분리**된다.

---

## 2. 확산(diffuse) 표면: 밝기 = 두 구면 함수의 내적

램버트 표면($f_r=\rho/\pi$, 시점 무관)이면 출사 밝기는 방향 $\omega_o$와 무관하고, 가림을 $V_p(\omega)\in\{0,1\}$(그 방향으로 하늘이 보이면 1)로 쓰면

$$
B(p)=\frac{\rho_p}{\pi}\int_{S^2} L(\omega)\;\underbrace{V_p(\omega)\,\max(0,\mathbf n_p\!\cdot\!\omega)}_{T_p(\omega)}\;d\omega
=\frac{\rho_p}{\pi}\int_{S^2} L(\omega)\,T_p(\omega)\,d\omega .
$$

여기서 $T_p(\omega)$가 **전달 함수(transfer function)**다. "방향 $\omega$에서 단위 빛이 들어오면 점 $p$가 얼마나 밝아지는가"를 나타내는 **구면 위 함수**이며, 코사인 항과 가림(그리고 아래에서 상호반사)이 모두 여기 들어 있다.

이 식의 꼴을 보면 $B(p)$는 **두 구면 함수 $L$과 $T_p$의 함수 내적** $\langle L, T_p\rangle$이다.

### 파세발(Parseval) 정리로 내적을 계수 내적으로 바꾸기

두 함수를 SH로 전개한다 (`sh_walkthrough.py` 1절의 정규직교성 $\int Y_kY_{k'}\,d\Omega=\delta_{kk'}$ 를 쓴다).

$$
L(\omega)=\sum_k l_k\,Y_k(\omega),\qquad T_p(\omega)=\sum_k t_{p,k}\,Y_k(\omega),
\qquad l_k=\int L\,Y_k\,d\Omega,\quad t_{p,k}=\int T_p\,Y_k\,d\Omega .
$$

내적에 대입하면

$$
\int L\,T_p\,d\omega
=\sum_{k}\sum_{k'} l_k\,t_{p,k'}\int Y_kY_{k'}\,d\Omega
=\sum_k\sum_{k'} l_k\,t_{p,k'}\,\delta_{kk'}
=\boxed{\sum_k l_k\,t_{p,k}=\mathbf l\cdot\mathbf t_p}
$$

즉 **구면 적분이 길이 $K$ 벡터 두 개의 내적으로 정확히 바뀐다**(정규직교 기저의 파세발 정리). $K$는 사용하는 계수 개수 — Sloan 2002는 5차까지 **25개**를 썼다(3DGS는 3차 16개).

> 이 한 줄이 PRT의 전부다: 무거운 적분 $\int L\,T_p$ 대신, 미리 계산해 둔 $\mathbf t_p$와 프레임마다 한 번 구하는 $\mathbf l$의 **내적 25번**만 하면 된다.

---

## 3. 무엇을 "미리 굽는가": 전달 함수 $T_p$의 내용

전처리(오프라인) 단계에서 각 정점 $p$마다 $\mathbf t_p$를 계산한다. 방법은 몬테카를로 광선 추적 — 구면 방향을 수천 개 샘플링해 각 방향 $\omega_j$에서

- **가림 여부** $V_p(\omega_j)$: $p$에서 $\omega_j$로 광선을 쏘아 물체 자신에 막히는지 검사 → **self-shadow**
- **코사인 항** $\max(0,\mathbf n_p\cdot\omega_j)$
- **상호반사(interreflection)**: 막힌 광선이 맞은 다른 표면 $q$가 받은 빛이 $p$로 다시 튕겨 오는 양. 이는 $q$의 전달 벡터 $\mathbf t_q$를 이용해 반복적으로(bounce 수만큼) 누적한다. 상호반사도 조명에 대해 **선형**이므로 결국 $\mathbf t_p$에 더해질 수 있다.

를 곱해 $Y_k(\omega_j)$와 함께 적분(사영)한다. 결과적으로 세 단계가 있다.

| 전달 종류 | $T_p(\omega)$ | 표현하는 효과 |
|---|---|---|
| unshadowed | $\max(0,\mathbf n_p\cdot\omega)$ | 코사인 항만. Ramamoorthi & Hanrahan(2001)의 SH 조사도(irradiance)와 동일 |
| shadowed | $V_p(\omega)\max(0,\mathbf n_p\cdot\omega)$ | 자기 그림자(soft self-shadow) |
| interreflected | 위 + 튕긴 빛 | 색 번짐(color bleeding), 틈새 밝아짐 |

핵심은 **가림·상호반사·코사인이 모두 조명에 대해 선형**이라는 점이다. 선형이니까 조명이 무엇이든 "조명 SH 기저 하나당 응답"을 저장해 두면($t_{p,k}$ = $k$번째 SH 조명에 대한 $p$의 응답) 임의의 조명은 그 응답들의 선형 결합으로 구해진다.

- **오프라인**: 정점당 $\mathbf t_p\in\mathbb R^{25}$ 계산 (분~시간).
- **런타임**: 환경맵 → $\mathbf l\in\mathbb R^{25}$ (프레임당 1회), 정점마다 $B(p)=\frac{\rho_p}{\pi}\,\mathbf l\cdot\mathbf t_p$ (정점 셰이더 수준의 비용).

---

## 4. 광택(glossy) 표면: 벡터 내적 → 행렬 곱(transfer matrix)

광택 재질은 출사 밝기가 **보는 방향** $\omega_o$에도 의존하므로 숫자 하나로 끝나지 않는다. Sloan 2002는 이를 두 단계로 나눈다.

1. **전달(transfer)**: 입사 조명 $L$(SH 벡터 $\mathbf l$)이 $p$에 도달하는 "전달된 조명" $L'_p$도 구면 함수이고 $L$에 대해 선형이므로, SH 계수 사이의 선형 사상 = **행렬**로 쓸 수 있다.
   $$
   \mathbf l'_p = M_p\,\mathbf l,\qquad M_p\in\mathbb R^{25\times 25}
   $$
   $M_p$의 $(k,k')$ 성분 = "$k'$번째 SH 조명이 들어왔을 때 $p$에 도달하는 조명의 $k$번째 SH 계수". 가림·상호반사는 여기 들어 있다.
2. **BRDF 컨볼루션 + 시점 평가**: 전달된 조명 $\mathbf l'_p$에 BRDF(Phong 로브 등, 원대칭이면 SH 대각 컨볼루션 $\alpha_\ell$)를 곱하고, 시점 방향 $\omega_o$(법선 기준 반사 방향)에서 SH를 **평가**한다.
   $$
   B(p,\omega_o)=\sum_k \alpha_{\ell(k)}\,(M_p\mathbf l)_k\;Y_k(\omega_o)
   $$

| | 확산 PRT | 광택 PRT |
|---|---|---|
| 정점당 저장 | 벡터 $\mathbf t_p$ (25개) | 행렬 $M_p$ (25×25 = 625개) |
| 런타임 연산 | 내적 1회 | 행렬-벡터 곱 + 시점 방향 SH 평가 |
| 출력 | 스칼라(시점 무관) | 시점 방향의 구면 함수 |

저장량 폭증이 광택 PRT의 약점이며, 후속 연구(Sloan 2003, Clustered PCA)가 행렬을 압축했다.

---

## 5. 회전하는 조명: SH 회전

조명(환경맵)이 회전하거나 물체가 회전하면 $L$을 물체 좌표계로 옮겨야 한다. SH의 좋은 성질은 **회전에 닫혀 있다**는 것 — 같은 차수 $\ell$ 안의 $2\ell+1$개 계수가 $(2\ell+1)\times(2\ell+1)$ 회전 행렬로 서로 섞이기만 하고 다른 차수로 새지 않는다.

$$
\mathbf l_{\text{obj}} = R_{\text{SH}}(R)\,\mathbf l_{\text{world}},\qquad R_{\text{SH}} = \mathrm{blockdiag}(R_0,R_1,R_2,\dots)
$$

따라서 환경맵을 다시 사영할 필요 없이, 프레임당 25차원 블록대각 회전 한 번으로 조명을 물체 프레임에 맞추고 그대로 $\mathbf l\cdot\mathbf t_p$를 계산한다. 그래서 논문 제목의 "**Dynamic** Lighting" — 조명은 매 프레임 바뀌어도 된다.

---

## 6. 한계

| 한계 | 이유 |
|---|---|
| **형상이 정적**이어야 함 | $\mathbf t_p$에 자기 가림·상호반사가 구워져 있으므로 물체가 변형되면 무효. 강체 이동·회전만 허용 (후속: Local Deformable PRT, 2005) |
| **저주파 조명**만 | 25개 SH는 부드러운 환경광만 표현. 태양 같은 점광원은 링잉(ringing)이 생기고 그림자가 흐릿함 (후속: Ng et al. 2003 웨이블릿 all-frequency PRT) |
| 조명이 **멀리** 있어야 함 | $L(\omega)$가 위치 무관이라는 가정. 근접 광원은 근사 |
| 광택은 저장량 큼 | 정점당 25×25 행렬 |
| 장면 간 그림자 없음 | 한 물체의 $\mathbf t_p$는 그 물체 자신만 고려(neighborhood transfer로 일부 확장) |

---

## 7. 3DGS와의 관계

`sh_walkthrough.py`의 3DGS 색 식과 광택 PRT의 최종 식을 나란히 놓으면 구조가 같다.

$$
\text{3DGS:}\quad \mathbf c_i(\mathbf d)=\max\Big(0,\sum_{k=0}^{15}\mathbf c_{i,k}\,Y_k(\mathbf d)+0.5\Big)
\qquad\qquad
\text{PRT(광택):}\quad B(p,\omega_o)=\sum_k \alpha_{\ell}\,(M_p\mathbf l)_k\,Y_k(\omega_o)
$$

둘 다 "**시점 방향에서 SH를 평가한 값이 색**"이다. 차이는 계수가 어디서 오는가에 있다.

| | PRT | 3DGS |
|---|---|---|
| SH 계수의 뜻 | $M_p\mathbf l$: **조명 $\mathbf l$과 전달 $M_p$가 분리**되어 런타임에 곱해짐 | $\mathbf c_{i,k}$: 조명·전달·BRDF가 **이미 곱해진 결과**를 학습으로 얻음 |
| 조명 변경 | 가능 (새 $\mathbf l$ 넣으면 됨) | 불가 — 촬영 당시 조명이 계수에 구워짐 |
| 형상 변경 | 불가 (강체만) | 불가 (같은 이유: 가림·반사가 색에 구워짐) |
| 계수 획득 | 광선 추적으로 사영 적분 | 다시점 사진에 대한 역전파(4절 실습) |
| 표현 대상 | 입사 조명($\omega_i$)의 전달 → 출사 | 출사 복사휘도($\omega_o$) 자체 |

즉 **3DGS의 SH 색은 PRT의 "$M_p\mathbf l$ 이후" 단계, 곧 특정 조명 아래에서의 출사 복사휘도를 시점 방향의 구면 함수로 구워 놓은 것**이라고 볼 수 있다. PRT가 "조명 × 전달"을 두 인자로 나눠 저장해 조명을 바꿀 수 있게 한 반면, 3DGS는 그 곱을 통째로 저장해 학습·렌더링을 단순하게 했고 그 대가로 재조명(relighting)을 포기했다. 3DGS를 재조명 가능하게 만드는 후속 연구(예: Relightable 3DGS, GS-IR)는 사실상 이 곱을 다시 조명 항과 재질·가시성 항으로 **분해**하려는 시도다.

또 하나 공통점: PRT가 25개 SH로 저주파 조명만 다뤘듯, 3DGS의 16개 SH도 부드러운 광택·프레넬까지만 표현한다(`sh_walkthrough.py` 정리 절의 "한계와 확장"). 날카로운 반사를 위해 두 분야 모두 SH를 벗어나는 방향(웨이블릿 / 구면 가우시안 / MLP 디코더)으로 발전했다.

---

## 한 줄 요약

PRT = "조명 $L$과 물체의 전달 $T_p$가 모두 구면 함수이고, $\int L\,T_p\,d\omega$는 정규직교 SH 덕분에 $\sum_k l_k t_{p,k}$로 바뀐다"는 관찰. 가림·상호반사·코사인을 $\mathbf t_p$(확산) 또는 $M_p$(광택)에 미리 굽고, 런타임엔 조명 계수와의 내적(또는 행렬 곱)만 한다. 3DGS의 SH 색은 이 곱을 특정 조명에서 통째로 구워 놓은 시점 의존 색이다.
