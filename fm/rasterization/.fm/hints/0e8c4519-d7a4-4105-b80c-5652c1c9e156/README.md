# `meta["isect_offsets"]`의 모양과 역할은?

`[C, tile_h, tile_w]` 모양으로, 각 타일이 `flatten_ids`에서 시작하는 위치를 담는다.
타일 `t`의 Gaussian 목록은 `flatten_ids[offsets[t]:offsets[t+1]]`이다.

## 시각화

![expy 시각화](expy.png)
