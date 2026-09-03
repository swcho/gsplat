# PSNR은 MSE로부터 어떻게 계산하는가?

`-10.0 * math.log10(mse)`로 계산한다(값 범위가 `[0,1]`로 정규화된 경우).
노트북의 `eval_psnr()`은 렌더를 `clamp(0,1)`한 뒤 `F.mse_loss`로 MSE를 구한다.

## 시각화

![expy 시각화](expy.png)
