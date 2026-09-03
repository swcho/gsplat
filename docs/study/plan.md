시작점으로는 좋지만, 그것만으로는 부족합니다. "CUDA by Example"(Sanders & Kandrot, 2010)은 커널 실행 모델, 스레드/블록 인덱싱, 공유 메모리, 원자 연산, 스트림까지의 기초를 가장 부드럽게 알려 주는 책입니다. 다만 CUDA 3.x 시절 책이라, 이 저장소의 커널이 의존하는 다음 요소들은 다루지 않습니다.

이 코드가 쓰는 것	파일 수	책의 커버 여부
C++ 템플릿 + 컴파일 타임 디스패치	53	없음
PyTorch C++ 확장 API (ATen 텐서, autograd)	58	없음
cooperative_groups (블록/워프 동기화)	25	없음 (2017년 도입)
glm 벡터/행렬 수학	25	없음
__launch_bounds__, 동적 공유 메모리	10 / 17	부분적
CUB (라딕스 정렬, 스캔)	6	없음
libcu++ (cuda::std::optional, cuda::atomic)	4	없음
현실적인 학습 경로

CUDA by Example 1~9장을 일주일 안에 빠르게 훑어 실행 모델과 공유 메모리 감각만 얻습니다. 후반부(텍스처, 그래픽 상호운용)는 건너뛰어도 됩니다.
"Programming Massively Parallel Processors" 4판(Hwu, Kirk, El Hajj)에서 리덕션, 프리픽스 스캔, 정렬, 타일링 장을 읽습니다. 이 저장소의 타일 기반 래스터라이저가 정확히 이 패턴들의 조합입니다.
CUDA C++ Programming Guide 중 cooperative groups와 warp-level primitives 절만 참고서처럼 봅니다.
3DGS 논문의 래스터라이저 절과 원본 diff-gaussian-rasterization 구조를 이해합니다. gsplat 커널은 그 알고리즘의 정제판이라 알고리즘을 알면 코드가 훨씬 빨리 읽힙니다.
코드 읽는 순서 (짧고 단순한 것부터)

AdamCUDA.cu (123줄), MCMCPerturbCUDA.cu: 원소별 커널. 스레드 인덱싱과 ATen 텐서 접근 방식만 익히면 됩니다.
QuatScaleToCovarCUDA.cu: glm 수학과 forward/backward 쌍의 구조.
ProjectionEWA3DGSFused.cu: 가우시안별 투영. 3DGS의 핵심 수학이 들어 있습니다.
IntersectTile.cu: 타일 비닝과 CUB 정렬. 여기서 래스터라이저의 입력이 만들어집니다.
RasterizeToPixels3DGSSerialBatchFwd.cu: 타일별 공유 메모리 배치 로딩과 알파 블렌딩. 이 저장소의 대표 커널이고, 이해하면 Bwd와 2DGS 변형은 같은 골격입니다.
3DGUT의 RasterizeToPixelsFromWorld*는 마지막에 보세요. 위 다섯 개를 이해한 뒤에도 가장 어렵습니다.
Python 쪽 _torch_impl.py에 같은 연산의 순수 PyTorch 참조 구현이 있습니다. CUDA 커널을 읽다 막히면 이 파일의 대응 함수와 나란히 두고 보는 것이 가장 효과적입니다.