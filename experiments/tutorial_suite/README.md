# 튜토리얼 통합 모델

[통합 실행과 과거 실험 보고](../../docs/rl-tutorials.md)

1·3·5번 모델은 기존 `experiments/tutorial_1_5/`에 있는 파일을 그대로 사용한다.
여기에는 통합 실행에 추가로 필요한 2·4번 체크포인트와 통합 후 검증 결과를 보관한다.

두 파일은 Git에 이미 있던 체크포인트를 바이트 그대로 가져왔다.
기준 커밋은 [33a2373dd2e9b6de962acc7f3c6b1a0997a6e74a](https://github.com/Songjeongkeun/Pygame-Pingu-Ball-/commit/33a2373dd2e9b6de962acc7f3c6b1a0997a6e74a)다.

| 파일 | 원본 경로 | SHA-256 |
| --- | --- | --- |
| `models/stage_02.zip` | `tutorial_2/results/reward_transfer_2_6/stage_02/personal/from_repeat_12/seed_42/best.zip` | `2357750478d0433fa903cfc76387481eb0f69460d42e46a083cf4c71f1a5283a` |
| `models/stage_04.pt` | `tutorial_2_4/tutorial4/result_tutorial4/last.pt` | `87e5fc82772acd185d7428e1fcabed7aa0c22aa8499f626abcb22b1d68f825d0` |

2번 점프 보상과 4번 리프트 환경은 같은 커밋에서 가져와 공통 실행 코드에 연결했다.
이전 학습 스크립트와 중복 모델·로그는 새 통합 폴더로 옮기지 않았다.

- [통합 평가: 단계당 100판](results/evaluation-seed120000.json)
- [원래 게임 엔진과 물리 대조: 단계당 한 판](results/verification.json)
