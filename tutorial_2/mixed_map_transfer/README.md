# 튜토리얼 2단계 PPO → 새 혼합 강화학습 맵 1~10 전이 평가

이 폴더는 **기존 PPO를 다시 학습시키지 않고** 새로 추가된 10개 혼합 맵에서 평가하고 화면에 재생하기 위한 독립 추가 코드입니다.

- 원본 체크포인트: `tutorial_2/results/reward_transfer_2_6/stage_02/personal/from_repeat_12/seed_42/best.zip`
- 맵 데이터: `assets/rl_maps/layouts.json`, 로더: `pingu_ball/rl_maps.py`
- 기존 PPO 모델/코드 (`rl/`, `tutorial_2/stage2_common_bridge/`) **수정 없음**
- 3행동(왼쪽/대기/오른쪽), 1행동당 6틱, 원래 충돌 엔진·맵 물리 그대로 사용
- 새로운 맵에 대한 **학습/파인튜닝/BC/DAgger 없음**, 원본 가중치 SHA-256을 결과에 기록

## 설치

ZIP 안 `tutorial_2/mixed_map_transfer/` 폴더를 게임 저장소의 기존 `tutorial_2/` 아래에 복사합니다. 프로젝트 **최상위** (즉 `main.py`와 `rl/`가 있는 위치)에서, 기존 `venv` 활성화 후 실행합니다.

## 1. 맵 10개와 관측 형태 검사 (모델 실행 전)

```cmd
python -m tutorial_2.mixed_map_transfer.run inspect
```

각 맵 로드, 객체 수, 모델에 전달하는 관측 길이를 확인합니다. **검사 결과로 성공률이 측정되는 것은 아닙니다.**

## 2. 맵 1번 우선 시험

```cmd
python -m tutorial_2.mixed_map_transfer.run evaluate --maps 1 --episodes 5 --limit 60 --seed 120000
```

이 명령은 **60초 제한의 짧은 사전 평가**입니다. 180초 공식 평가와 비교하면 안 됩니다.

## 3. 혼합 맵 1~10 평가

```cmd
python -m tutorial_2.mixed_map_transfer.run evaluate --maps 1-10 --episodes 20 --limit 180 --seed 120000
```

각 맵마다 20판, 각 판 180초 게임 시간 상한. 기본은 확률적으로 정책 행동을 **샘플링**하며, `--greedy`를 추가하면 각 단계에서 가장 확률이 높은 행동만 선택합니다. **샘플링과 greedy 결과는 따로 적으세요.**

맵 지정 예시: `--maps 1,2,5`, `--maps 3-7`.

콘솔에는 각 판의 성공·실패·종료 시간·최소 도착 거리와 맵별 승률이 찍힙니다. 결과 파일은 `tutorial_2/results/mixed_map_transfer/` 아래 **JSON과 CSV**로 저장합니다. 실패 위치는 CSV의 `end_x`, `end_y`, `max_x`를 참고하세요. 왕복 맵이라 `max_x`가 곧 진행률을 뜻하지는 않습니다.

## 4. **PPO가 직접 움직이는 화면**

```cmd
python -m tutorial_2.mixed_map_transfer.run play --map 1 --seed 120000
```

Pygame 실제 게임 렌더러에 기존 PPO 행동을 주입합니다. 자동 플레이이며 좌우 키를 누를 필요는 없습니다. **Esc 또는 창 X**로 종료합니다. 다른 맵은 `--map 2`~`10`으로 바꿉니다.

## 중요한 실험 해석 주의점

**튜토리얼 2단계 96% 성능과 동일한 환경에서 평가하는 것이 아닙니다.**

- 튜토리얼의 공통 PPO는 입력 크기가 고정돼 있으나, 새 혼합 맵들은 객체 수가 더 많을 수 있습니다. 모든 객체를 매번 읽은 뒤, 공과 가까운 `CAPACITY`개를 전체 객체 입력에, 가까운 `LOCAL_COUNT`개를 지역 입력에 넣습니다. 가까운 순으로 잘리는 객체가 생길 수 있습니다.
- 훈련된 PPO는 튜토리얼 1~5번에 대한 stage one-hot만 받습니다. 새로운 맵 ID 슬롯을 추가하지 않고, **기존의 stage 2 슬롯만 1로 유지**합니다.
- 튜토리얼 2단계에만 존재하는 `spring-1`→`middle` 등 3개 구간 정보를 혼합 맵에 임의로 심지 않습니다. **모든 튜토리얼 구간(4×11 필드) 입력은 0**입니다. 이는 학습 분포와 다릅니다.
- 따라서 실패 시 실제 PPO 실력 외에도 입력 어댑터의 정보 손실·학습 분포 차이 때문일 수 있습니다. **0%여도 학습 96%가 거짓인 것은 아닙니다.**
- 전이 평가는 원본 정책 행동을 측정하는 목적이어서, 에피소드 반환 보상은 성공 +100점 / 실패·시간 초과 -25점을 `/100` 한 것만 계산합니다. 튜토리얼 2 전용 점프 중간 구간 보상은 새 맵에서는 정의하지 않습니다. **정책이 고정이므로 평가 보상이 행동 선택에 영향을 주지 않습니다.**
- 새 혼합 맵은 왕복 경로이므로 최댓값 `max_x`만으로 진행도를 판단하면 안 됩니다. `min_goal_distance`는 분석 보조 지표이며 경유 블록 달성을 의미하지 않습니다.

## 검증

```cmd
python -m unittest discover -s tutorial_2/mixed_map_transfer/tests -v
```

소스에서 의존성 없이 실행 가능한 정렬·맵 번호 테스트가 포함됩니다. `inspect`가 실제 GitHub 맵/관측 호환성 검사이며, 위 `evaluate` 또는 `play`가 실제 PPO 게임 실행 검사입니다.

## 새로 생기는 파일

이 ZIP은 `tutorial_2/mixed_map_transfer/`만 추가합니다. `main.py`, `rl/`, `pingu_ball/`, PPO 모델 `.zip`, 기존 실험 결과는 덮어쓰지 않습니다. 평가 출력은 무시/삭제해도 원본 학습 결과에는 영향이 없습니다.
