# 고교 수준 설명: gradient를 왜 `[-1,1]` 기준으로 바꿔서 모으나

## 0. 오늘 쓸 고교 지식만 정리

- **함수의 미분**: $f'(a)$는 "$a$에서 입력을 조금 늘렸을 때 출력이 몇 배로 변하나"라는 변화율.
- **연쇄법칙(합성함수 미분)**: $\dfrac{dL}{dx} = \dfrac{dL}{du}\cdot\dfrac{du}{dx}$.
- **벡터의 크기**: $\lVert(a,b)\rVert = \sqrt{a^2+b^2}$.
- **일차함수 그래프의 기울기**: $y = mx + k$ 에서 기울기는 $m$.

이 네 개면 충분하다. 편미분($\partial$)이 나오지만, "변수가 여러 개일 때 그중 하나만
움직여 보는 보통의 미분"이라고 읽으면 된다.

## 1. 무엇을 하려는 상황인가

3D Gaussian Splatting은 반투명한 타원 얼룩(Gaussian) 수십만 개를 화면에 겹쳐 그려서
사진을 재현한다. 학습 중에 "여기는 얼룩 하나로는 부족하니 둘로 늘리자"를 판단해야 하는데,
그 판단 기준이 **"이 얼룩을 화면에서 살짝 옮기면 오차가 얼마나 줄어드는가"** 이다.

손실(오차)을 $\mathcal{L}$, 어떤 얼룩의 화면상 중심 위치를 $(x, y)$라 하면 판단에 쓰는 값은

$$g = \left\lVert \left(\frac{\partial \mathcal{L}}{\partial x},\ \frac{\partial \mathcal{L}}{\partial y}\right) \right\rVert
= \sqrt{\left(\frac{\partial \mathcal{L}}{\partial x}\right)^{2} + \left(\frac{\partial \mathcal{L}}{\partial y}\right)^{2}}$$

이고, 이 $g$가 정해진 문턱값보다 크면 얼룩을 복제하거나 쪼갠다.

## 2. 여기서 생기는 "단위" 문제

문제는 $x$를 무슨 단위로 재느냐다. 코드가 실제로 계산해 주는 값은 **픽셀 단위**의 미분

$$\frac{\partial \mathcal{L}}{\partial x_{\text{pix}}} \quad \left[\frac{\text{오차}}{\text{픽셀}}\right]$$

이다. 그런데 미분값은 단위를 바꾸면 숫자가 바뀐다. 고교 물리의 익숙한 상황과 똑같다.
같은 경사로의 기울기를 "1 m 갈 때 오르는 높이(m)"로 재면 $0.1$, "1 cm 갈 때 오르는 높이(m)"로
재면 $0.001$이다. 경사로는 그대로인데 숫자가 100배 다르다.

렌더 해상도도 이와 같다. 같은 씬을 폭 $400$px로 그릴 때의 1픽셀과 폭 $1600$px로 그릴 때의
1픽셀은 화면에서 차지하는 비중이 4배 다르다. 그래서 "픽셀당 오차 변화율"을 그대로
문턱값 $2\times10^{-4}$와 비교하면, **해상도를 바꿀 때마다 문턱값을 다시 맞춰야 한다.**

## 3. 해법: 해상도에 안 흔들리는 자를 하나 정한다

그래서 gsplat은 화면 좌표를 항상 $[-1, 1]$로 놓는 자(NDC, normalized device coordinates)를
쓴다. 화면 왼쪽 끝이 $-1$, 오른쪽 끝이 $+1$, 위아래도 $-1 \sim +1$이다. 폭이 $W$ 픽셀이든
$4W$ 픽셀이든 이 자의 눈금은 변하지 않는다.

두 자 사이의 관계는 일차함수다. NDC 좌표 $x_n$과 픽셀 좌표 $x_{\text{pix}}$는

$$x_{\text{pix}} = \frac{x_n + 1}{2}\,W$$

로 대응된다($x_n=-1 \Rightarrow 0$px, $x_n=+1 \Rightarrow W$px 확인해 보라).
이 일차함수의 기울기가 곧 환산 계수다.

$$\frac{d\,x_{\text{pix}}}{d\,x_n} = \frac{W}{2}$$

이제 연쇄법칙을 그대로 적용하면 끝난다.

$$\frac{\partial \mathcal{L}}{\partial x_n}
= \frac{\partial \mathcal{L}}{\partial x_{\text{pix}}}\cdot\frac{d\,x_{\text{pix}}}{d\,x_n}
= \frac{\partial \mathcal{L}}{\partial x_{\text{pix}}}\cdot\frac{W}{2}$$

y축도 똑같이 $\dfrac{\partial \mathcal{L}}{\partial y_n} = \dfrac{\partial \mathcal{L}}{\partial y_{\text{pix}}}\cdot\dfrac{H}{2}$.

그리고 이것이 정답에 나오는 `default.py:248`의 코드다.

```python
# normalize grads to [-1, 1] screen space
grads[..., 0] *= info["width"]  / 2.0 * info["n_cameras"]
grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
```

`* width / 2.0`은 새로운 근사나 추정이 아니라, **이미 구한 미분값에 자 바꾸기 계수를 곱한
정확한 등식**이다. 이렇게 바꾼 뒤의 값은 "얼룩을 화면 절반 폭만큼 옮길 때의 오차 변화율"이라는,
해상도와 무관한 의미를 갖는다.

### 확인: 정말 해상도가 사라지는가

가로 폭만 2배로 늘려 렌더한다고 하자. 화면에서 같은 만큼(=같은 NDC 거리) 얼룩을 옮기려면
픽셀로는 2배를 움직여야 하므로, 픽셀당 변화율은 대략 절반이 된다.

$$\frac{\partial \mathcal{L}}{\partial x_{\text{pix}}} \ \longrightarrow\ \frac{1}{2}\cdot\frac{\partial \mathcal{L}}{\partial x_{\text{pix}}},
\qquad W \ \longrightarrow\ 2W$$

곱하면 $\dfrac{1}{2}\cdot\dfrac{2W}{2} = \dfrac{W}{2}$ 로 원래와 같다. 두 변화가 정확히
서로를 지운다. 그래서 문턱값 하나를 고정해 둘 수 있다.

## 4. 같이 곱해지는 `n_cameras`는 확률과 통계의 "평균 되돌리기"

같은 줄에 카메라 대수 $C$도 곱한다. 이유는 손실을 여러 장의 사진에 대해 **평균**으로
정의했기 때문이다.

$$\mathcal{L} = \frac{1}{C}\sum_{c=1}^{C} \mathcal{L}_c
\quad\Longrightarrow\quad
\frac{\partial \mathcal{L}}{\partial x} = \frac{1}{C}\sum_{c=1}^{C}\frac{\partial \mathcal{L}_c}{\partial x}$$

평균을 쓰면 사진을 많이 묶을수록 각 미분값이 $1/C$배로 작아진다. 여기에 $C$를 곱하면
$1/C$가 사라져서 **"사진 1장 기준"의 크기**로 돌아온다. 결국 line 248 한 줄이
해상도 의존성과 배치(묶음) 크기 의존성을 동시에 지운다.

## 5. 어떻게 "누적"하나

정규화까지 끝난 값을 매 스텝 바로 쓰지 않고 100스텝 정도 모아서 평균을 낸다. 어떤 얼룩이
$k$번 화면에 보였고 그때의 값들이 $g_1, \dots, g_k$였다면, 코드는

$$S = \sum_{i=1}^{k} g_i, \qquad n = k \qquad (\text{각각 } \texttt{grad2d},\ \texttt{count})$$

두 개만 들고 있다가, 판정할 때 산술평균

$$\bar{g} = \frac{S}{n}$$

을 만들어 $\bar{g} > 2\times 10^{-4}$인지 본다. 여기서 두 가지가 중요하다.

1. **각 $g_i$는 이미 벡터의 크기**다. $\left(\frac{\partial\mathcal{L}}{\partial x_n}, \frac{\partial\mathcal{L}}{\partial y_n}\right)$를
   그냥 더하면 방향이 반대인 것끼리 상쇄돼 0에 가까워질 수 있으므로, 방향을 버리고
   크기 $\sqrt{\cdot^2+\cdot^2}$만 더한다.
2. **분모 $n$은 "보인 횟수"**다. 카메라 뒤에 있거나 화면 밖이라 아예 안 보인 경우
   (코드에서는 화면상 반경 `radii`가 $0$인 경우)는 세지 않는다. 안 보인 뷰를 0으로 세면
   평균이 부당하게 낮아져서, 정작 필요한 얼룩이 안 쪼개진다. 이건 확률과 통계에서
   표본에 넣을 것과 뺄 것을 구분하는 일과 같다.

## 6. 정리

$$\underbrace{\frac{\partial \mathcal{L}}{\partial x_{\text{pix}}}}_{\text{코드가 주는 값(픽셀 단위)}}
\ \times\ \underbrace{\frac{W}{2}}_{\text{픽셀}\to[-1,1]\text{ NDC}}
\ \times\ \underbrace{C}_{\text{배치 평균 되돌리기}}
\ \xrightarrow[\ \lVert\cdot\rVert\ ]{}\ g_i
\ \xrightarrow[\ \text{보인 횟수로 평균}\ ]{}\ \bar{g}
\ \gtrless\ 2\times10^{-4}$$

- 정규화 기준은 화면을 항상 $[-1,1]$로 보는 **NDC**.
- 곱해지는 수는 연쇄법칙에서 나온 **자 바꾸기 계수** $W/2$, $H/2$.
- 그 덕분에 문턱값 $2\times10^{-4}$가 해상도가 400px이든 1600px이든 그대로 통한다.
- 주의: "gradient를 $[-1,1]$ 안으로 자른다(클리핑)"는 뜻이 **아니다**. 값의 범위를 자르는 게
  아니라 값을 재는 **좌표계(단위)**를 바꾸는 것이다.
