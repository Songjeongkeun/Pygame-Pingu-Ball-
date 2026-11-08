# 튜토리얼 1~5 강화학습 통합

튜토리얼 1~5의 새 학습, 평가, 화면 재생은 `tools/tutorial_rl.py` 한 곳에서 실행한다.
기본 시연은 스테이지에 맞는 기존 PPO 모델을 불러온다. 모델을 불러오는 방식, 관측, 행동, 보상만 단계별로 구분한다.
스테이지 4의 체크포인트는 다른 네 모델과 형식이 달라 원래 네트워크 구조를 사용한다.
이 묶음은 다섯 모델을 공통 코드로 실행한다. 다섯 스테이지를 학습한 단일 모델의 실험은 아니다.

| 단계 | 통합 실행에 쓰는 모델 | 행동과 보상 |
| --- | --- | --- |
| 1 파괴 블록 | 기본 비교에서 선택한 PPO | 좌·대기·우, 행동당 6틱. 지정된 다음 발판에 착지하면 구간 보상. |
| 2 점프 블록 | 개인 보상의 12→6틱 전이 PPO | 좌·대기·우, 행동당 6틱. 목표 발판 착지·높이·가로 범위 보상. |
| 3 좌우 이동 | 기본 비교에서 선택한 PPO | 좌·대기·우, 행동당 6틱. 지정 경로 달성과 이글루 도착을 구분. |
| 4 상하 이동 | 리프트 전용 PPO의 100만 스텝 최종본 | 방향 3개 × 12·30·72틱. 리프트 탑승·높은 탑승·상단 착지와 접근 보상. |
| 5 시간 블록 | 기본 비교에서 선택한 PPO | 좌·대기·우, 행동당 6틱. 시간 블록을 거친 착지에 구간 보상. |

1·3·5번은 [기존 보존본](../experiments/tutorial_1_5/README.md)의 해시와 학습 당시 소스·맵을 사용한다.
2·4번은 현재 게임의 튜토리얼 지형과 원래 관측·보상 규칙을 사용한다. [체크포인트 출처](../experiments/tutorial_suite/README.md)를 별도로 기록했다.
복합 맵 1은 [독립 실험](../experiments/rl_map_01/README.md)으로 유지한다.

## 실행

저장소 루트에서 `requirements-rl.txt`를 설치한 Python 환경으로 실행한다.

```powershell
python tools/tutorial_rl.py list --suite combined
python tools/tutorial_rl.py play --suite combined --stage 4 --greedy
python tools/tutorial_rl.py verify --suite combined
python tools/tutorial_rl.py evaluate --suite combined --stage all --episodes 100 --seed 120000 --output output/tutorial-suite/new-evaluation.json
```

`play`는 창을 띄우고, `--headless`를 추가하면 창 없이 한 판을 실행한다.
`verify`는 체크포인트 해시와 로드 여부를 확인하고, 한 판의 각 물리 틱을 게임 엔진과 대조한다.
`evaluate`는 원래 출발점에서 정책 확률을 표본 추출한다. 각 판의 시드는 120000부터 순서대로 증가한다.
각 모델은 원래 행동·보상 규격을 유지하므로 원점수를 단계 간 성능 비교에 사용하지 않는다.
원본 모델의 학습을 반복하거나 가중치를 변경하지 않는다.

새 학습은 같은 단계별 환경을 사용하되, 결과를 별도 파일에 저장한다. 기존 실험의 PPO 학습 설정을 재현하는 명령은 아니다.

```powershell
python tools/tutorial_rl.py train --stage 4 --steps 100000 --output output/tutorial-suite/new-stage04.zip
python tools/tutorial_rl.py play --suite combined --stage 4 --checkpoint output/tutorial-suite/new-stage04.zip
```

## 통합 후 검증 결과

평가 시드 120000~120099, 각 단계 100판, 모든 행동을 정책 확률에서 표본 추출했다.

| 단계 | 완주 | 평균 원점수 |
| --- | ---: | ---: |
| 1 파괴 블록 | 100/100 | 140.00 |
| 2 점프 블록 | 99/100 | 131.43 |
| 3 좌우 이동 | 100/100 | 100.10 |
| 4 상하 이동 | 100/100 | 164.96 |
| 5 시간 블록 | 100/100 | 130.00 |

2번의 한 판은 낙하로 끝났다. 이 표는 위 평가 방식과 각 단계의 기존 모델에 대한 결과다.
[500판 개별 기록](../experiments/tutorial_suite/results/evaluation-seed120000.json)을 보존했다.
별도 [물리 대조](../experiments/tutorial_suite/results/verification.json)에서는 최대 확률로 실행한 한 판을 단계마다 모두 완주했다.
모든 틱에서 공의 위치·속도·게임 시간과 충돌 이벤트가 원래 게임 엔진과 일치했다.
2·4번은 기존 구현과도 최대 확률·확률 표본 재생을 각각 한 판씩 대조해 관측, 보상, 종료 결과가 일치하는 것을 확인했다.

## 과거 실험 보고

| 실험 | 당시의 평가 | 결과와 해석 |
| --- | --- | --- |
| 튜토리얼 1~5 A2C/PPO 기본 비교 | 학습 시드 0, 검증 선택 모델, 맵별 독립 시험 100판 | A2C는 전 단계 0/100. PPO는 1·3·5번 각각 99/100, 2·4번 0/100. 3번은 성공한 99판 중 98판이 지정 핵심 구간 보상을 받지 않았다. |
| 초기 2·4번 PPO 재학습 | 각 조건의 학습 시드 0·1·2, 모델별 독립 시험 100판 | 0.1초 행동 유지 조건에서 2번 17%·60%·86%, 4번 0%·0%·0%. 세 시드의 차이가 크다. |
| 별도 점프·리프트 PPO | 2·4번 각각 100만 스텝, 학습 시드 42. 최종 모델별 500판, 확률 표본 95%와 최대 확률 5%를 혼합 | 2번 100%, 4번 99.8%. 현재 통합에서는 이 실험의 4번 최종본을 사용한다. |
| 2번 전용 12→6틱 전이 | 공통 관측과 개인 점프 보상. 이동 전 별도 시험 200판, 평가 시드 120000 | 192/200. 현재 통합의 2번 모델이지만, 이 수치를 새 통합 평가에 옮겨 적지는 않는다. |

당시 설정과 결과는 [기본·재학습 보고서](../experiments/tutorial_1_5/README.md),
[별도 2·4번 보고서](https://github.com/Songjeongkeun/Pygame-Pingu-Ball-/tree/33a2373dd2e9b6de962acc7f3c6b1a0997a6e74a/tutorial_2_4),
[2번 전이 보고서](https://github.com/Songjeongkeun/Pygame-Pingu-Ball-/tree/33a2373dd2e9b6de962acc7f3c6b1a0997a6e74a/tutorial_2)에 있다.
위 표는 역사적 결과다. 평가 조건과 모델 선택 기준이 다른 실험을 통합 성공률로 평균하지 않는다.
