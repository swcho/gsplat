# 3D 얼룩을 화면에 눌러 찍기 — 투영 단계가 만드는 4가지

> 전제: 정규분포, 타원의 방정식, 벡터, 함수의 미분(접선 근사). 행렬은 회전변환 정도만 알아도 된다.

## 0. 목표

3D Gaussian Splatting은 공간에 **뿌옇게 번진 타원체 얼룩**(Gaussian) 수백만 개를 띄워놓고 사진을 만든다. 그런데 화면은 2차원이다. 그러니 3차원 얼룩을 화면에 **눌러 찍어**(splat) 2차원 타원 얼룩으로 바꿔야 한다.

`fully_fused_projection()`이 그 "눌러 찍기"를 하는 함수이고, 각 얼룩마다 4가지를 계산해서 내놓는다.

1. `means2d` — 얼룩 중심이 화면 몇 번째 픽셀에 오는가
2. `depths` — 카메라로부터 얼마나 멀리 있는가
3. `conics` — 화면에 찍힌 타원의 **모양**
4. `radii` — 그 타원이 대략 몇 픽셀 반경까지 퍼지는가

아래에서 하나씩 쌓아 올린다.

---

## 1. 1차원 정규분포에서 출발

고등학교에서 배운 정규분포의 모양은 이랬다.

$$f(x)\propto \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

$\mu$는 중심, $\sigma$는 퍼진 정도. 지수 안쪽을 이렇게 다시 쓰면 앞으로 나올 이야기와 모양이 맞는다.

$$f(x)\propto\exp\!\left(-\tfrac12\,\Delta\cdot\frac{1}{\sigma^2}\cdot\Delta\right),\qquad \Delta = x-\mu$$

**"편차 $\Delta$를 두 번 곱하고, 분산의 역수를 끼워 넣는다."** 이 구조를 기억해 두자. 3차원으로 가도 이 형태가 그대로 유지된다.

## 2. 2차원·3차원으로 늘리기: 공분산 행렬

이제 $x$가 하나가 아니라 좌표 $\Delta=(\Delta x,\Delta y)$ 라고 하자. $\frac{1}{\sigma^2}$ 자리에 들어갈 것은 이제 수 하나가 아니라 **2×2 행렬**이다.

$$f(\Delta)\propto\exp\!\left(-\tfrac12\,\Delta^\top A\,\Delta\right)$$

$\Delta^\top A\Delta$를 풀어 쓰면

$$\Delta^\top A\Delta = a\,\Delta x^2 + 2b\,\Delta x\Delta y + c\,\Delta y^2 \qquad\left(A=\begin{pmatrix}a&b\\b&c\end{pmatrix}\right)$$

즉 $\Delta x,\Delta y$의 **2차식**이다. 그리고 이 값이 상수인 곳, $a\Delta x^2+2b\Delta x\Delta y+c\Delta y^2 = k$ 는 기하에서 배운 **타원의 방정식**이다(회전된 타원). $b$가 0이 아니면 축이 기울고, $a,c$가 각 축 방향의 뾰족함을 정한다.

$$\text{2차원 Gaussian의 등고선} = \text{타원}$$

이게 "얼룩이 타원으로 보인다"는 말의 정체다. 원뿔곡선(conic section)의 방정식이라서 코드에서 이 행렬을 **`conics`** 라고 부른다.

3차원도 똑같다. $\Delta$가 3성분이 되고 행렬이 3×3이 되며, 등고선은 **타원체**가 된다. 이때 퍼짐을 나타내는 3×3 행렬 $\Sigma$를 **공분산 행렬**이라 하고, $A = \Sigma^{-1}$ 관계다(1차원에서 $\frac{1}{\sigma^2}$가 분산의 역수였던 것과 같다).

## 3. 얼룩 모양은 어떻게 저장하나: $\Sigma = RSS^\top R^\top$

3×3 공분산 $\Sigma$에는 자유도가 6개(대칭이므로) 있지만, 아무 6개 숫자나 넣으면 안 된다. "퍼짐"을 나타내야 하니 어떤 방향으로 재도 분산이 양수여야 한다. 학습 도중 숫자가 조금 잘못 움직여 이 조건이 깨지면 얼룩이 뒤집혀 계산이 터진다.

그래서 gsplat은 $\Sigma$를 직접 저장하지 않고 **타원체를 만드는 레시피**를 저장한다.

- 단위 구를 각 축으로 $s_x,s_y,s_z$ 배 늘린다 → 대각행렬 $S=\mathrm{diag}(s_x,s_y,s_z)$
- 그걸 회전시킨다 → 회전행렬 $R$

$$\Sigma = R\,S\,S^\top R^\top = (RS)(RS)^\top$$

이 형태면 항상 안전하다. 임의의 방향 벡터 $v\ne 0$에 대해

$$v^\top \Sigma v = v^\top (RS)(RS)^\top v = \lVert (RS)^\top v\rVert^2 > 0$$

**제곱의 합이니 반드시 양수.** 이게 asset이 "항상 양의 정부호가 보장된다"고 한 이유다. $s_i$는 log 값으로 저장해 `exp`를 씌우므로 자동으로 양수, $R$은 사원수(quaternion) 4개 숫자로 저장한다.

## 4. 카메라 좌표로 옮기기 — 근사 없음

카메라 기준으로 좌표를 다시 쓴다. 회전 $R_{cw}$와 이동 $t$를 쓰면

$$\mu_c = R_{cw}\mu + t,\qquad \Sigma_c = R_{cw}\,\Sigma\,R_{cw}^\top$$

중심 $\mu$는 회전하고 옮기지만, 공분산에는 $t$가 없다. **퍼짐은 평행이동해도 안 변하니까.** 그리고 회전은 길이를 보존하니 얼룩의 모양·크기도 그대로다. 여기까지는 완전히 정확하다.

## 5. 여기가 핵심: 원근투영은 비선형이다

카메라에서 화면으로 가는 규칙은 원근투영이다. 거리 $z$가 멀면 작게 보인다.

$$\pi(x,y,z) = \left(f_x\frac{x}{z}+c_x,\ f_y\frac{y}{z}+c_y\right)$$

$z$로 **나누기** 때문에 이건 비선형 함수다. 그리고 비선형 함수로 Gaussian을 밀면 결과는 Gaussian이 아니다 — 찌그러진 이상한 모양이 된다. 그러면 3절에서 쌓아 온 "타원" 구조를 못 쓴다.

### 해결책: 접선 근사

고등학교 미적분의 그 아이디어를 그대로 쓴다. 곡선 $y=f(x)$를 점 $x_0$ 근처에서

$$f(x) \approx f(x_0) + f'(x_0)(x-x_0)$$

로 **직선으로 바꿔** 다뤘다. 얼룩은 어차피 좁은 영역에만 퍼져 있으니, 그 좁은 영역에서만 투영을 직선(선형)으로 근사하면 충분하다.

다만 지금은 입력이 3개($x,y,z$)이고 출력이 2개다. 그래서 "미분계수 하나"가 아니라 **각 출력을 각 입력으로 편미분한 표**가 필요하다. 이걸 **야코비안 행렬** $J$라 한다. 그냥 $f'(x_0)$의 다변수 버전이다.

첫 번째 출력 $u = f_x\dfrac{x}{z}+c_x$ 를 각각 미분해 보자 (다른 변수는 상수 취급).

$$\frac{\partial u}{\partial x} = \frac{f_x}{z},\qquad \frac{\partial u}{\partial y} = 0,\qquad \frac{\partial u}{\partial z} = -\frac{f_x x}{z^2}$$

마지막 것은 $\frac{d}{dz}z^{-1} = -z^{-2}$ 를 쓴 것뿐이다. 두 번째 출력 $v$도 똑같이 해서 2×3 표로 모으면

$$J = \begin{pmatrix} \dfrac{f_x}{z} & 0 & -\dfrac{f_x x}{z^2}\\[8pt] 0 & \dfrac{f_y}{z} & -\dfrac{f_y y}{z^2}\end{pmatrix}$$

코드(`gsplat/cuda/_torch_impl.py:53`)에 이 6개 성분이 그대로 있다.

### 선형 근사 아래에서 공분산은 이렇게 변한다

1차원에서, $X$의 표준편차가 $\sigma$일 때 $Y = aX$의 표준편차는 $|a|\sigma$, 즉 **분산은 $a^2\sigma^2$** 였다. 계수가 두 번 곱해진다. 다차원에서 이 "두 번"이 앞뒤로 붙는 형태가 된다.

$$\boxed{\ \Sigma' = J\,\Sigma_c\,J^\top\ }$$

$J$가 2×3, $\Sigma_c$가 3×3, $J^\top$가 3×2이므로 결과는 **2×2**. 3D 타원체가 화면 위의 2D 타원으로 눌려 찍혔다. 이 방식에 붙은 이름이 **EWA splatting**(Elliptical Weighted Average, Zwicker 등 2001)이고, 3DGS가 이 논문에서 가져온 부분이다.

> 근사이므로 오차도 있다. 특히 화면 가장자리나 아주 가까운 얼룩에서 어긋난다. 코드가 $J$를 계산할 위치를 시야각 근처로 `clamp`하는 이유가 이것 — 시야 밖 멀리서 $x/z$가 커지면 $-f_x x/z^2$ 항이 폭주해 터무니없는 타원이 나온다.

## 6. 왜 역행렬(`conics`)로 넘겨주나

이제 픽셀 색을 칠할 차례다. 픽셀 하나에서 이 얼룩의 진하기는 (1절의 구조 그대로)

$$\alpha = o\cdot\exp\!\left(-\tfrac12\,\Delta^\top \Sigma'^{-1}\Delta\right),\qquad \Delta = (\text{픽셀 위치}) - \texttt{means2d}$$

여기 필요한 건 $\Sigma'$가 아니라 **$\Sigma'^{-1}$** 이다. 얼룩 하나가 수십~수백 픽셀에 걸치므로, 픽셀마다 역행렬을 구하면 같은 계산을 수백 번 반복하는 셈이다. 그래서 투영 단계에서 **딱 한 번** 역행렬을 구해 넘긴다.

2×2 역행렬은 공식이 짧다. $\Sigma'=\begin{pmatrix}\sigma_{11}&\sigma_{12}\\\sigma_{12}&\sigma_{22}\end{pmatrix}$, $\det = \sigma_{11}\sigma_{22}-\sigma_{12}^2$ 일 때

$$\Sigma'^{-1} = \frac{1}{\det}\begin{pmatrix}\sigma_{22} & -\sigma_{12}\\ -\sigma_{12} & \sigma_{11}\end{pmatrix}$$

대칭이라 서로 다른 값이 3개뿐이다. 그래서 `conics`는 얼룩당 숫자 **3개**로 저장된다.

$$\texttt{conics} = \left[\frac{\sigma_{22}}{\det},\ \frac{-\sigma_{12}}{\det},\ \frac{\sigma_{11}}{\det}\right] = [a,\,b,\,c]$$

이걸 받으면 픽셀 계산은 나눗셈 없는 2차식 하나로 끝난다.

$$\tfrac12\Delta^\top\Sigma'^{-1}\Delta = \tfrac12\left(a\,\Delta x^2 + c\,\Delta y^2\right) + b\,\Delta x\,\Delta y$$

카드 답의 "2D 공분산(conic)"은 이 뜻이다 — **엄밀히는 2D 공분산의 역행렬**이 넘어간다.

### $\det$가 0이 되면?

카메라를 정면으로 향한 얇은 원판 같은 얼룩은 화면에서 거의 **선분**으로 눌린다. 타원의 한 축이 0에 가까워져 $\det\to 0$, 역행렬이 발산한다. 게다가 픽셀보다 작은 얼룩은 화면 격자와 간섭해 반짝인다(에일리어싱).

두 문제를 한 줄로 막는다.

$$\Sigma' \leftarrow \Sigma' + 0.3\,I \qquad (\texttt{eps2d}=0.3)$$

두 축의 분산에 0.3씩 더하는 것 = 표준편차 $\sqrt{0.3}\approx 0.55$픽셀만큼 **일부러 살짝 번지게** 하기. 이러면 어떤 얼룩도 최소 한 픽셀 크기는 갖게 되고 $\det$도 0이 아니다. 대신 번진 만큼 옅어지므로, 안티에일리어싱 모드에서는 보정 계수

$$\texttt{compensations} = \sqrt{\frac{\det\Sigma'_{\text{원래}}}{\det(\Sigma'+0.3I)}}\ \le 1$$

를 함께 계산해 불투명도에 곱해 준다.

## 7. `radii`: 어디까지가 "보이는" 범위인가

Gaussian은 수학적으로는 무한히 뻗지만 실제로는 금방 0에 가까워진다. 그러니 어딘가에서 잘라야 하고, 자를 지점을 정하는 게 `radii`다.

$$\texttt{radius\_x} = \lceil 3.33\sqrt{\sigma_{11}}\,\rceil,\qquad \texttt{radius\_y} = \lceil 3.33\sqrt{\sigma_{22}}\,\rceil$$

$\sqrt{\sigma_{11}}$은 x축 방향 화면상 표준편차이므로, 이건 "$3.33\sigma$까지만 그린다"는 뜻이다. 왜 3.33인가? 그 지점의 진하기를 계산해 보면

$$\exp\!\left(-\tfrac12\times 3.33^2\right) = e^{-5.544} \approx 0.0039 \approx \frac{1}{256}$$

색을 0~255로 표현하니 $\tfrac{1}{256}$ 아래는 **한 단계도 못 바꾼다**. 눈에 보이지 않는 지점에서 딱 자른 것이다. (정규분포에서 $3\sigma$ 안에 99.7%가 들어간다는 그 감각과 같은 종류의 컷오프다.)

### `radii = 0`은 "이 얼룩은 없는 셈"

`radii`는 진짜 반경 말고 **유효성 표시**로도 쓰인다. 다음 중 하나에 걸리면 0으로 덮어쓴다.

- 카메라 뒤에 있거나 너무 가깝다/멀다 → `depths`가 near/far 범위 밖
- 화면 바깥에 있다 → `means2d ± radii` 박스가 화면과 안 겹침

그래서 반환된 텐서들은 크기는 전체 얼룩 수만큼이지만 **대부분의 칸이 무효**다. `radii > 0`으로 걸러야 한다. asset이 정확히 그렇게 센다.

```python
(info['radii'] > 0).all(-1).sum()   # 화면에 실제로 보이는 Gaussian 수
```

`.all(-1)`은 x, y 두 반경이 **둘 다** 0보다 커야 유효하다는 뜻이다.

`radii`의 두 번째 쓰임은 다음 단계다. 화면을 16×16 픽셀 타일로 자른 뒤, 각 얼룩이 `means2d ± radii` 박스로 어느 타일들에 걸치는지 센다. 반경을 넉넉히 잡으면 쓸데없는 타일 계산이 늘어 느려지고, 너무 좁게 잡으면 얼룩 꼬리가 잘려 타일 경계에 이음선이 보인다. 이 값의 정확도가 곧 속도와 화질의 균형점이다.

## 8. `depths`: 순서를 정하는 숫자

$$\texttt{depths} = z_c\ (\text{카메라 좌표계의 } z)$$

투명한 얼룩들을 겹쳐 색을 합칠 때는 순서가 중요하다. 앞의 것이 뒤의 것을 가리기 때문이다. 최종 색은

$$C = \sum_i c_i\,\alpha_i\prod_{j<i}(1-\alpha_j)$$

로 계산되는데, $\prod_{j<i}(1-\alpha_j)$가 "$i$번째 앞에 있는 것들을 통과해 남은 빛의 비율"이다. 그러니 $i$의 순서, 즉 **가까운 것부터의 정렬**이 필요하고, 그 정렬 키가 `depths`다.

## 9. 정리: 왜 이 4개인가

투영 단계는 **다음 두 단계가 바로 쓸 수 있는 형태**로 미리 가공해 넘기는 일을 한다.

| 산출값 | 계산식 | 누가 쓰나 |
|---|---|---|
| `means2d` | $\left(f_x\frac{x_c}{z_c}+c_x,\ f_y\frac{y_c}{z_c}+c_y\right)$ | 타일 박스 + 픽셀별 $\Delta$ / 학습에서는 이것의 기울기가 얼룩을 쪼갤 신호 |
| `depths` | $z_c$ | 앞→뒤 정렬 키 |
| `conics` | $(J\Sigma_c J^\top + 0.3I)^{-1}$ 의 3원소 | 픽셀별 진하기 $\alpha$ |
| `radii` | $\lceil 3.33\sqrt{\text{대각}}\,\rceil$, 컬링 시 0 | 걸치는 타일 계산 + 유효성 마스크 |

그리고 이 세 변환

$$\Sigma \xrightarrow[\text{회전}]{R_{cw}\,\cdot\,R_{cw}^\top} \Sigma_c \xrightarrow[\text{접선 근사}]{J\,\cdot\,J^\top} \Sigma' \xrightarrow[\text{역행렬}]{} \texttt{conics}$$

를 커널 하나에서 끝내는 것이 함수 이름의 **"fully fused"** 다. 중간 결과를 메모리에 쓰고 다시 읽는 왕복을 없애야, 얼룩 수백만 개를 실시간으로 처리할 수 있다.
