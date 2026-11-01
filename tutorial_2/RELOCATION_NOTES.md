# 파일 이동 및 확인 기록

- 프로젝트 내부 설치 위치: `Pygame-Pingu-Ball-/tutorial_2/`
- 실행 위치: `Pygame-Pingu-Ball-` 프로젝트 최상위 (`python -m tutorial_2.stage2_common_bridge.run ...`)
- Python import: `tutorial_2.reward_repeat_2_4`, `tutorial_2.reward_transfer_2_6`, `tutorial_2.stage2_common_bridge`, `tutorial_2.tutorial_test`
- 최종 모델: `tutorial_2/results/reward_transfer_2_6/stage_02/personal/from_repeat_12/seed_42/best.zip`
- 12프레임 원본: `tutorial_2/results/reward_repeat_2_4/stage_02/personal/repeat_12/seed_42/best.zip`
- 팀 폴더 위치: `rl`, `pingu_ball`, `experiments`, `tools`, `tutorial_2_4`는 프로젝트 루트 그대로
- 경로 변경 때문에 비교 실패하지 않도록 6프레임 PPO 설정 검증에서 경로 문자열만 제외하고 모델 SHA256 등 조건은 검증
- 과거 평가 JSON의 절대 Windows 경로는 `tutorial_2/results/...` 형식으로 정리함; 수치·성공률을 변경하지 않음
- 원본과 새 저장본의 PPO 최종·출발 `best.zip` 데이터가 같음을 SHA256 비교 검증
- 정적·회귀 테스트: 103 passed, 6 skipped (실제 게임 의존성 부족으로 건너뜀)

## 주의

- **압축 해제만으로 실제 게임 96%를 다시 실행한 것은 아닙니다.** 전체 프로젝트의 게임/강화학습 런타임은 사용자 PC에서 검증해야 합니다.
- 팀 PPO 1·3·5 모델은 기존 `tools/tutorial_rl.py`와 `experiments/...`를 그대로 사용합니다.
- `tutorial_test`는 구형 실험 코드입니다. Git 경량판에 원래의 큰 결과 체크포인트는 제외되므로 일부 구형 결과 조회는 원본 전체 보존판을 사용하세요.
- stage 2 최종 6프레임 PPO만 96% 독립 평가가 검증됐으며, 단일 5개맵 정책을 만든 것은 아닙니다.
