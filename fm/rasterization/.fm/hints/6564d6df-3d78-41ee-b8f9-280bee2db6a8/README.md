# 왜 Σ를 직접 최적화하지 않고 quats + scales로 표현하는가

## 0. 한 줄 요약

3D Gaussian의 모양은 공분산 $\Sigma$(3×3 대칭 **양정치**)로 정해지지만, **경사 하강은 "양정치"라는 제약을 지켜 줄 방법이 없다.**
그래서 학습 파라미터를 $\Sigma$ 자체가 아니라 회전 `quats`와 축 길이 `scales`로 두고,
$$\Sigma = (R\,\mathrm{diag}(s))\,(R\,\mathrm{diag}(s))^\top = R\,S\,S^\top R^\top$$
로 **구성(construct)** 한다. 이 형태는 파라미터가 어떤 값이 되든 항상 유효한 공분산이다.

에셋의 해당 셀(`rasterization_walkthrough.py` §2)이 정확히 이 이야기다:

```python
def covar_from_quat_scale(q, s):
    R = _quat_to_rotmat(q)          # [N,3,3]  (내부에서 q를 정규화)
    M = R * s[..., None, :]         # R @ diag(s)  — 열 스케일링
    return M @ M.transpose(-1, -2)  # Σ = R S Sᵀ Rᵀ
```

`M = R * s[..., None, :]`가 `R @ diag(s)`와 같은 이유: `diag(s)`를 오른쪽에서 곱하는 것은 $R$의 **$j$번째 열에 $s_j$를 곱하는** 연산이고, 브로드캐스트 곱 `R * s[None, :]`이 바로 그것이다. (왼쪽 곱 `diag(s) @ R`은 행 스케일링이라 다르다.)

---

## 1. 원 논문의 논거 (Kerbl et al. 2023, "3D Gaussian Splatting", Sec. 4)

논문 4절의 핵심 문단을 풀어 쓰면:

> 공분산 행렬 $\Sigma$는 **양반정치(positive semi-definite)일 때만 물리적 의미**를 갖는다. 우리는 모든 파라미터를 경사 하강으로 최적화하는데, 경사 하강은 "유효한 행렬만 만들어라"라는 제약을 쉽게 걸 수 없다. 업데이트 스텝과 그래디언트는 **너무나 쉽게 유효하지 않은 공분산 행렬을 만들어 낸다.**

그래서 논문은 "더 표현력 있으면서도 최적화하기 좋은 표현"을 찾고, 공분산이 **타원체(ellipsoid)의 구성(configuration)** 을 기술한다는 점에 착안한다.

> 스케일링 행렬 $S$와 회전 행렬 $R$이 주어지면 대응하는 $\Sigma$를 찾을 수 있다: $\Sigma = R S S^\top R^\top$.

그리고 두 인자를 **독립적으로 저장**한다: 스케일은 3D 벡터 $s$, 회전은 쿼터니언 $q$. 논문은 자동미분의 오버헤드를 피하려고 이 파라미터들에 대한 그래디언트를 부록에서 해석적으로 유도한다(gsplat도 CUDA 커널에 이 도함수를 직접 짜 넣었다).

정리하면 논거는 세 겹이다.

1. **유효성(validity)**: $\Sigma$가 PSD가 아니면 Gaussian이 아니다 — 밀도 $\exp(-\tfrac12 x^\top\Sigma^{-1}x)$가 어떤 방향으로 발산하고, $\Sigma^{-1}$(conic)이나 $\sqrt{\det}$ 계산이 깨진다.
2. **제약 없는 최적화(unconstrained optimization)**: Adam 같은 옵티마이저에게 $(q, s)$는 아무 실수 값이나 가져도 되는 **자유 변수**다. 제약을 파라미터화 안에 흡수해 버렸으므로 투영(projection)이나 페널티가 필요 없다.
3. **해석 가능성**: $R$과 $s$는 각각 "타원체의 방향"과 "축 반지름"이라 그 자체로 의미가 있고, 밀도화(densification) 규칙이 이 값을 직접 쓴다(§7).

---

## 2. 경사 하강 한 스텝이 제약을 깨는 구체적인 예

### 2-1. 대칭성부터 깨진다

$\Sigma$의 9개 성분을 그냥 파라미터로 두면, 손실의 그래디언트 $\partial L/\partial \Sigma$는 **일반적으로 대칭이 아니다**. 한 스텝만 밟아도 $\Sigma_{01}\neq\Sigma_{10}$이 되어 애초에 "공분산 행렬"이 아니게 된다.

이건 그나마 고칠 수 있다 — 상삼각 6개 성분 $(\Sigma_{00},\Sigma_{01},\Sigma_{02},\Sigma_{11},\Sigma_{12},\Sigma_{22})$만 저장하면 대칭성은 구조적으로 보장된다. 실제로 gsplat의 `quat_scale_to_covar_preci(..., triu=True)`도 결과를 이 6개로 반환한다. **하지만 양정치성은 여전히 보장되지 않는다.**

### 2-2. 양정치성은 고칠 방법이 없다

**예 A — 대각 성분이 음수가 되는 경우.** 아주 얇은 Gaussian $\Sigma = \mathrm{diag}(0.01,\,1,\,1)$을 생각하자. 손실이 "더 얇아져라"라고 밀면 $\partial L/\partial\Sigma_{00} > 0$이고, 학습률 $\eta$에 대해
$$\Sigma_{00} \leftarrow 0.01 - \eta\,g,\qquad \eta g = 0.05 \;\Rightarrow\; \Sigma_{00} = -0.04.$$
**분산이 음수다.** 어떤 학습률도 이걸 원천 차단하지 못한다 — 얇은 Gaussian은 3DGS에서 오히려 흔하다(평면 표면을 표현하는 splat은 한 축이 거의 0이다).

**예 B — 대각은 전부 양수인데도 깨지는 경우 (더 위험).** 2×2로 보면
$$\Sigma = \begin{bmatrix}1 & 0.9\\ 0.9 & 1\end{bmatrix},\qquad \det = 1 - 0.81 = 0.19 > 0 \ \ (\text{양정치, 고유값} 1.9,\ 0.1)$$
여기서 상관 성분만 조금 키우는 스텝 $\Sigma_{01}\!\leftarrow\!\Sigma_{01}+0.15$를 밟으면
$$\Sigma' = \begin{bmatrix}1 & 1.05\\ 1.05 & 1\end{bmatrix},\qquad \det = 1 - 1.1025 = -0.1025 < 0,\qquad \lambda = \{2.05,\ -0.05\}.$$
**대각 성분은 둘 다 1로 멀쩡한데 행렬은 부정치(indefinite)가 되었다.** "각 성분을 양수로 클램프한다" 같은 값싼 수선책이 통하지 않는다는 걸 보여 주는 예다. 유효성을 지키려면 매 스텝 고유분해를 하고 음의 고유값을 잘라내야 하는데(수백만 개 Gaussian × 수만 스텝), 비용도 크고 미분도 불안정하다.

### 2-3. 깨진 뒤에 실제로 무슨 일이 일어나는가

에셋의 파이프라인 순서를 따라가면 피해가 어디로 번지는지 보인다.

- $\Sigma_{2D} = J\,\Sigma_c\,J^\top + \varepsilon I$ — $\Sigma$가 부정치면 $\Sigma_{2D}$도 부정치가 될 수 있다.
- `conics = inv(cov2d)` — 부정치면 $\det$이 0을 지나며 **역행렬이 폭발**한다(NaN).
- `radii = ceil(3.33 * sqrt(diag(cov2d)))` — 음수의 제곱근 → NaN → 타일 교차 계산 전체가 오염된다.
- 알파 $\alpha = o\exp(-\tfrac12 d^\top\Sigma_2^{-1}d)$ — 부정치면 지수가 **양수**가 되어 $\alpha$가 발산한다.

NaN 하나가 그래디언트를 타고 역전파되면 그 스텝의 학습 전체가 망가진다. 수백만 개 splat 중 하나만 그래도 그렇다. 즉 이건 "가끔 있는 수치 문제"가 아니라 **구조적으로 막아야 하는 문제**다.

---

## 3. $M = RS$ 인수분해가 항상 PSD인 이유

핵심 한 줄:

$$\mathbf{v}^\top (M M^\top)\,\mathbf{v} \;=\; (M^\top\mathbf{v})^\top (M^\top\mathbf{v}) \;=\; \lVert M^\top \mathbf{v}\rVert^2 \;\ge\; 0 \qquad \forall\,\mathbf{v}.$$

**$M$이 무엇이든 상관없다.** 어떤 실수 3×3 행렬 $M$에 대해서도 $MM^\top$은 대칭이고($({MM^\top})^\top = MM^\top$) 양반정치다. 그래서 옵티마이저가 $q$와 $s$를 어디로 옮겨 놓든 결과는 항상 유효한 공분산이다. 제약이 파라미터 공간에서 **사라진 게 아니라, 함수의 상(image)에 흡수**됐다.

**언제 "반(semi)"이 아니라 진짜 양정치(PD)인가?** $MM^\top$이 PD ⟺ $M$이 가역 ⟺ 모든 $s_j \neq 0$ (회전 $R$은 항상 가역이므로). 즉 **어떤 축의 스케일이 정확히 0이면** 그 방향의 분산이 0인 납작한(degenerate) Gaussian이 되고 $\Sigma^{-1}$이 존재하지 않는다. 이래서 §5의 `exp` 파라미터화가 두 번째로 중요해진다 — $\exp(\cdot) > 0$은 스케일이 **결코 0이 되지 않음**을 보장한다.

**고유구조 확인.** $R$이 직교($R^\top R = I$)이므로
$$\Sigma = R\,S S^\top R^\top = R\,\mathrm{diag}(s_1^2, s_2^2, s_3^2)\,R^\top$$
이것이 정확히 $\Sigma$의 **고유분해**다: 고유값은 $s_j^2$, 고유벡터는 $R$의 열. 그래서 $\sqrt{\lambda_j} = |s_j|$이고, 에셋의 확인 코드가 이걸 그대로 보여 준다.

```python
print("sqrt(eigvals(Σ)) =", torch.linalg.eigvalsh(cov_manual[0]).sqrt(),
      " ← scales를 정렬한 것과 같다")
```

toy Gaussian 0 (z축 30° 회전, $s=(0.30,0.12,0.10)$)의 실제 값:

$$\Sigma = \begin{bmatrix} 0.071100 & 0.032736 & 0\\ 0.032736 & 0.033300 & 0\\ 0 & 0 & 0.010000\end{bmatrix},\qquad \lambda = (0.01,\ 0.0144,\ 0.09),\qquad \sqrt{\lambda} = (0.10,\ 0.12,\ 0.30).$$

$\sqrt{\lambda}$가 정렬된 `scales`와 정확히 일치한다. 대각 성분($0.0711$)이 $s_1^2 = 0.09$가 **아니라는** 점에 주목하라 — 회전이 축을 섞어 놓았기 때문이다. 이것이 §2-2 예 B와 같은 구조다: 비대각 성분은 "회전한 만큼" 자연스럽게 생기며, $RSS^\top R^\top$ 형태가 그 크기를 **자동으로 유효한 범위 안에** 묶어 준다.

**정밀도 행렬(precision)도 공짜로 얻는다.** gsplat의 `_quat_scale_to_preci_half`가 $M' = R\,\mathrm{diag}(1/s)$를 만들고 $\Sigma^{-1} = M'M'^\top$을 계산한다 — 역행렬을 **푸는(solve)** 게 아니라 **구성한다**. $\Sigma$를 직접 파라미터로 뒀다면 매번 3×3 역행렬을 풀어야 하고, 부정치일 때 그게 터진다.

---

## 4. 자유도 6 vs 파라미터 7

| | 개수 | 설명 |
|---|---|---|
| 대칭 3×3 행렬의 자유도 | **6** | 상삼각 성분 $\Sigma_{00},\Sigma_{01},\Sigma_{02},\Sigma_{11},\Sigma_{12},\Sigma_{22}$ ($n(n{+}1)/2 = 6$) |
| 기하학적 분해 | **6** | 3D 회전 $SO(3)$의 자유도 3 + 축 길이 3 |
| gsplat이 저장하는 수 | **7** | `quats` 4개 + `scales` 3개 |

**왜 7개인가.** 3D 회전 자체는 자유도가 3이지만, 이를 **특이점 없이(singularity-free), 부드럽게** 저장할 방법이 마땅치 않다.

- 오일러 각(3개)은 짐벌락(gimbal lock)이 있어 특정 자세에서 그래디언트가 병적으로 행동한다.
- 회전 행렬(9개)은 $R^\top R = I$라는 제약을 다시 경사 하강으로 못 지킨다 — **$\Sigma$와 똑같은 문제의 반복**이다.
- 축-각(axis-angle, 3개)은 $2\pi$ 근처에서 불연속이다.

쿼터니언 4개는 이 문제들이 전부 없고, 대신 **$\lVert q\rVert = 1$이라는 제약 1개**를 진다. $4 - 1 = 3$으로 자유도는 맞다.

**결정적인 차이**: 이 제약은 $\Sigma$의 양정치성과 달리 **정규화 한 줄로 공짜로 해결된다.**

```python
def _quat_to_rotmat(quats):
    quats = F.normalize(quats, p=2, dim=-1)   # ← 여기
    w, x, y, z = torch.unbind(quats, dim=-1)
    ...
```

즉 `quats`는 최적화 중에 정규화되지 않은 아무 4벡터여도 되고, **사용 직전에 나누기 한 번**으로 단위 쿼터니언이 된다. $\lVert q \rVert$ 방향은 $R$에 아무 영향을 주지 않는 잉여 자유도이므로 그 방향의 그래디언트는 0이 된다(정규화가 미분 가능하므로 자동으로 그렇다). 결국 **7개 파라미터가 6개 자유도를 덮되, 제약이 미분 가능한 정규화로 흡수된다.**

참고로 $q$와 $-q$는 같은 회전을 준다(이중 덮개, double cover). 최적화에는 문제가 되지 않는다 — 두 점 다 같은 손실을 주는 동등한 최소점일 뿐이다.

---

## 5. 학습 코드의 관행: 로그 공간 scales, 사용 직전 정규화 quats

`examples/simple_trainer.py`의 forward:

```python
quats = splats["quats"]                 # [N,4]  정규화 안 함 — rasterization이 내부에서 함
scales = torch.exp(splats["scales"])    # [N,3]  ← 저장은 log 공간
opacities = torch.sigmoid(splats["opacities"])
```

### scales는 log 공간에 저장한다

`splats["scales"]`는 **실제 축 길이가 아니라 그 로그**다. 세 가지 이득이 있다.

1. **양수 보장.** $\exp(\cdot) > 0$이므로 어떤 스텝을 밟아도 스케일이 0이나 음수가 되지 않는다. §3에서 봤듯 $s_j = 0$은 $\Sigma$를 특이(singular)하게 만들고 $\Sigma^{-1}$이 터진다. `exp`가 이 마지막 구멍을 막는다.
2. **곱셈적 업데이트.** 로그 공간의 덧셈 스텝은 실공간의 곱셈이다: $\log s \leftarrow \log s - \eta g \Rightarrow s \leftarrow s\cdot e^{-\eta g}$. 크기가 $10^{-3}$인 splat과 $10^{1}$인 splat이 **같은 학습률로 같은 상대 변화**를 받는다. 3DGS 씬은 스케일이 수십~수백 배 차이 나므로 이게 중요하다.
3. **초기화가 자연스럽다.** 초기화도 로그 공간에서 이뤄진다:
   ```python
   scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)
   ```
   최근접 이웃까지의 평균 거리를 스케일로 삼되, 저장은 로그로 한다.

같은 발상이 opacity에도 적용된다 — `sigmoid`로 $(0,1)$에 가둔다. **"제약을 옵티마이저에게 부탁하지 말고 파라미터화에 넣어라"** 라는 하나의 원칙이 3DGS 파라미터 전반을 관통한다.

### quats는 사용 직전에 정규화

`simple_trainer.py`에는 주석 처리된 줄이 남아 있다:

```python
# quats = F.normalize(splats["quats"], dim=-1)  # [N, 4]
# rasterization does normalization internally
quats = splats["quats"]  # [N, 4]
```

정규화를 **렌더링 시점(CUDA 커널 / `_quat_to_rotmat`)에 딱 한 번** 하는 편이 낫다. 파라미터를 매 스텝 in-place로 정규화(re-projection)하면 Adam의 모멘텀/2차 모멘트 상태와 파라미터가 어긋나 최적화가 어색해진다. 반면 forward에 정규화를 포함하면 그것도 계산 그래프의 일부라 그래디언트가 알아서 접선 방향으로만 흐른다.

단, **파라미터를 직접 만지는 코드에서는 명시적으로 정규화한다.** `gsplat/strategy/ops.py`의 split:

```python
scales = torch.exp(params["scales"][sel])      # log → 실제 길이
quats  = F.normalize(params["quats"][sel], dim=-1)
rotmats = normalized_quat_to_rotmat(quats)
samples = torch.einsum("nij,nj,bnj->bni", rotmats, scales,
                       torch.randn(2, len(scales), 3, device=device))
```

여기서 `einsum`이 하는 일이 정확히 $R\,\mathrm{diag}(s)\,\boldsymbol\varepsilon$, 즉 **$\mathcal{N}(0,\Sigma)$에서 샘플링**이다 (표준정규 $\boldsymbol\varepsilon$에 $M = RS$를 곱하면 공분산이 $M M^\top = \Sigma$). 인수분해 표현이 "유효성 보장"만이 아니라 **샘플링도 공짜로 준다**는 실전 이득이다 — $\Sigma$만 있었다면 Cholesky를 따로 돌려야 했다.

---

## 6. 대안들과의 비교

| 표현 | 파라미터 수 | PD 보장 | 문제점 |
|---|---|---|---|
| $\Sigma$ 6개 성분 직접 | 6 | ✗ | §2의 예처럼 한 스텝에 깨진다. 매 스텝 사영이 필요 |
| **Cholesky $\Sigma = LL^\top$** ($L$ 하삼각) | 6 | PSD ✓ / PD는 조건부 | $\ell_{jj} > 0$을 별도로 강제해야 PD. 축 방향·길이가 성분에 섞여 있어 **해석 불가** |
| **로그-Cholesky** ($\ell_{jj} = e^{d_j}$) | 6 | PD ✓ | 자유도 최소·PD 보장까지 완벽하지만 여전히 해석 불가. 회전과 스케일을 따로 못 만짐 |
| **행렬 지수 $\Sigma = \exp(A)$** ($A$ 대칭) | 6 | PD ✓ | 3×3 행렬 지수/그 도함수를 수백만 개에 대해 계산 — **비용이 비싸다** |
| **$\Sigma = R S S^\top R^\top$ (3DGS)** | 7 | PD ✓ (with `exp`) | 파라미터 1개 잉여. 대신 나머지가 전부 좋다 |

**Cholesky가 자유도(6)로는 더 "깔끔"한데 왜 안 쓰나?** 두 가지다.

1. **PD가 자동이 아니다.** $LL^\top$은 PSD일 뿐이고, PD가 되려면 대각 $\ell_{jj}\neq 0$을 따로 챙겨야 한다 — 결국 로그-Cholesky 같은 추가 장치가 붙는다. (반면 3DGS는 `exp(scales)`라는, opacity의 `sigmoid`와 같은 결의 장치 하나로 끝난다.)
2. **의미가 사라진다.** $L$의 성분 $\ell_{21}$은 기하학적으로 아무것도 뜻하지 않는다. 반면 $s = (0.30, 0.12, 0.10)$은 "이 splat은 30cm × 12cm × 10cm의 납작한 타원체"라는 문장으로 바로 읽힌다. 다음 절이 이 차이가 왜 실제로 돈이 되는지다.

로그-Cholesky는 이론적으로는 3DGS 파라미터화의 가장 강력한 경쟁자다. 실제로 후속 연구들이 시도했지만, **밀도화 휴리스틱을 다시 설계해야 한다**는 비용이 크다.

---

## 7. 독립된 축 스케일이 주는 해석 가능성 — 밀도화가 이걸 쓴다

$s$가 "각 축의 길이"라는 명시적 의미를 갖는 덕분에, **밀도화(densification) 규칙을 스케일로 직접 쓸 수 있다.** `gsplat/strategy/default.py`:

```python
is_grad_high = grads > self.grow_grad2d
is_small = (
    torch.exp(params["scales"]).max(dim=-1).values
    <= self.grow_scale3d * state["scene_scale"]
)
is_dupli = is_grad_high & is_small      # 작다 → 복제(clone): "부족 재구성"
is_large = ~is_small
is_split = is_grad_high & is_large      # 크다 → 분할(split): "과잉 재구성"
```

읽는 그대로다: **"화면 그래디언트가 큰데 splat이 작으면 복제하고, 크면 쪼갠다."** 여기서 "크다/작다"의 판정이 `exp(scales).max(dim=-1)` — **가장 긴 축의 길이**를 씬 스케일과 비교하는 것이다. $\Sigma$의 성분만 있었다면 이 판정을 하기 위해 매번 고유값 분해를 돌려야 한다.

split 연산도 마찬가지로 $s$를 직접 조작한다:

```python
p_split = torch.log(scales / 1.6).repeat(2, 1)   # 새 자식들의 스케일
```

**축 길이를 1.6으로 나눈다** — 원 논문의 $\phi = 1.6$이다. 부모 타원체 안에 자식 둘이 들어가도록 크기를 줄인다는 뜻이며, 로그 공간이므로 `log(scales / 1.6)`으로 쓴다. 그리고 자식들의 위치는 §5에서 본 대로 부모 분포 $\mathcal{N}(0,\Sigma)$에서 샘플링한다.

$\Sigma$나 Cholesky $L$을 파라미터로 뒀다면 이 규칙들을 전부 "고유값을 뽑아서, 스케일을 나누고, 다시 조립"하는 형태로 써야 했을 것이다. **파라미터화가 알고리즘의 어휘를 결정한다** — 3DGS가 $(q, s)$를 고른 진짜 이유의 절반은 여기에 있다.

---

## 8. 파이프라인 안에서의 위치

에셋의 전체 흐름에서 이 절은 ①번, 맨 앞이다.

```
① quats, scales ──→ Σ = R S Sᵀ Rᵀ   (3D 공분산, 항상 PD)
② Σ_c = R_view Σ R_viewᵀ            (카메라 좌표 — 직교 변환이라 PD 보존)
③ Σ_2D = J Σ_c Jᵀ + εI              (EWA 투영 — J가 랭크 2라도 εI가 PD를 되살림)
④ conics = Σ_2D⁻¹                   (여기서 역행렬을 취하므로 PD가 필수)
⑤ α = o·exp(-½ dᵀ conic d)          (PD여야 지수가 음수 → α ≤ o)
```

체인 전체가 PD 보존 연산으로 이뤄져 있음을 눈여겨보라. $R_{\text{view}}$는 직교이므로 ②는 합동변환(congruence)이라 PD를 보존한다. ③의 $J$는 2×3이라 $J\Sigma_c J^\top$이 특이해질 수 있는데(정확히 시선 방향으로 납작한 Gaussian), `eps2d=0.3` 항이 그것까지 막는다 — 이 역시 "제약을 구조에 넣는다"의 또 다른 사례다.

**출발점 ①에서 PD가 보장되면 끝까지 보장된다.** 그래서 이 한 줄의 파라미터화 선택이 파이프라인 전체의 수치 안정성을 떠받친다.

---

## 9. 요약

- **문제**: $\Sigma$는 대칭 PD여야 의미가 있는데, 경사 하강에는 그 제약을 지킬 방법이 없다. 한 스텝이면 $\det < 0$이 되고(예: $\begin{smallmatrix}1 & 1.05\\ 1.05 & 1\end{smallmatrix}$, 대각은 멀쩡한데 고유값 $-0.05$), 그 뒤 conic 역행렬과 반경 제곱근이 NaN을 뿜는다.
- **해법**: $\Sigma = M M^\top$, $M = R\,\mathrm{diag}(s)$. $\mathbf v^\top MM^\top\mathbf v = \lVert M^\top\mathbf v\rVert^2 \ge 0$이므로 **$M$이 무엇이든 유효**하다. 제약이 파라미터화 속으로 사라졌다.
- **7 = 4 + 3, 자유도는 6**: 쿼터니언 4개가 회전 자유도 3을 특이점 없이 덮고, 잉여 1개는 `F.normalize` 한 줄로 소거된다.
- **`exp`로 완성**: `scales`를 로그로 저장해 $s_j > 0$을 보장 → PSD가 아니라 진짜 PD. 덤으로 곱셈적 스텝을 얻는다.
- **보너스**: $R$의 열이 고유벡터, $s_j^2$이 고유값이므로 고유분해가 공짜. $M$을 표준정규에 곱하면 샘플링도 공짜. 그리고 $s$가 읽히는 값이라 밀도화 규칙(`exp(scales).max() <= grow_scale3d * scene_scale`, `log(scales/1.6)`)을 그 위에 바로 세울 수 있다.
