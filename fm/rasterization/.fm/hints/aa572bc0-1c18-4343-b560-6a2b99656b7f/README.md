# `rasterize_mode="antialiased"`는 무엇을 하는가?

eps2d 블러 전후 행렬식의 비 $\sqrt{\det_0/\det}$ 를 불투명도에 곱한다.
화면 크기가 작아질 때 밝기를 보존하는 Mip-Splatting 기법이다.

## 시각화

![expy 시각화](expy.png)
