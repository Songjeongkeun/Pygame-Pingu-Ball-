> **이동 후 실행:** 이 문서는 과거 실험 기록이야. 아래 명령은 프로젝트 루트에서 `tutorial_2.tutorial_test` 패키지로 실행하도록 경로만 맞췄어. 최종 96% 모델은 `tutorial_2/README.md`의 명령을 사용해.

# PinguBall PPO v4 — 팀 공통 1~5단계 엔진 기반 2·4단계 학습

## 핵심

이 ZIP은 팀 GitHub `Songjeongkeun/Pygame-Pingu-Ball-`의 `main` 브랜치(2026-11-01 확인)를 **전제**로 합니다. `rl/`, `pingu_ball/`, `tutorial_2_4/`의 어떤 파일도 덮어쓰지 않고 `tutorial_2/tutorial_test/`에 v4 전용 파일만 추가합니다.

- **원본 유지**: 팀의 `rl/milestone_env.py`에 구현된 1~5단계 공통 관측, 공통 3행동(-1/0/+1), 실제 물리 120Hz, 결정당 물리 6틱, 클리어 +100/실패 -25/구간 착지 +10, 보상 /100 스케일.
- **common**: 공통 보상 그대로. 독립 대조군.
- **combined**: 기존 v3처럼 점프 후 목표와의 높이/가로 정렬 +0.5점씩, 리프트 최초 착지/충분한 상승 +12점씩. 별도의 보상 중복 없음.
- **guided (실험군)**: combined + **오른쪽 새 최대 진척 x**에 대한 실험용 탐색 보상(픽셀당 0.045, 에피소드 최대 36점). 실패/타임아웃 시 진척 보상은 회수하므로 단순 전진 후 추락하는 전략을 보상하지 않음. 성공 시에는 유지. 진척 보상 활성화는 `stage==2/4`를 하드코딩하지 않고 **점프 발판 또는 508 상하 이동 발판**의 존재 여부로 결정함. 1·3·5단계에는 적용되지 않음.
- 기존 9행동/단계별 정책 `last.pt`가 아닌 **팀 공통 관측·3행동 PPO**를 새로 학습합니다. 두 모델은 구조가 달라 바로 혼용할 수 없습니다.

## 1. 설치 (VS Code 터미널에서 프로젝트 최상위 폴더)

이 ZIP을 프로젝트 루트에 압축 해제 후 **사용자 PC**의 활성화된 가상환경에서:

```bat
python -m pip install -r requirements-rl.txt
```

`gymnasium`, `stable-baselines3`, `pygame`, `torch` 등을 설치합니다. 이미 venv가 설정돼 있다면 같은 환경에서 진행하세요.

## 2. 게임 실행 검증 (학습 전 필수)

가장 간단한 방법은 `tutorial_test/01_check_and_install.cmd`를 프로젝트에서 실행하는 것입니다. 직접 실행할 경우:

```bat
python -m unittest discover -s tutorial_2\tutorial_test\tests -p "test_*_v4.py" -v
python -m tutorial_2.tutorial_test.train_ppo_v4 parity --stage 2 --actions 1000
python -m tutorial_2.tutorial_test.train_ppo_v4 parity --stage 4 --actions 1000
python -m tutorial_2.tutorial_test.train_ppo_v4 smoke --stage 2 --reward guided
python -m tutorial_2.tutorial_test.train_ppo_v4 smoke --stage 4 --reward guided
```

`01_check_and_install.cmd`는 1~5단계 모두 `parity` + `smoke` 검사를 실행합니다. **실제 게임 물리가 동일하다는 parity 결과가 확인되기 전에는 학습을 진행하지 마세요.**

## 3. 0부터 학습 (2·4단계)

`02_train_2_and_4_from_zero.cmd` 파일 실행 또는:

```bat
python -m tutorial_2.tutorial_test.train_ppo_v4 train --stage 2 --reward guided --steps 500000 --seed 42
python -m tutorial_2.tutorial_test.train_ppo_v4 train --stage 4 --reward guided --steps 500000 --seed 42
```

기존 가중치를 불러오는 `--init-model`을 쓰지 않으므로 **각각 무작위 정책에서 처음부터 학습**합니다. 2·4단계 PPO는 서로 다른 모델이며, 같은 1~5단계 공통 엔진과 동일한 신경망/하이퍼파라미터 설정을 사용합니다. 50만 스텝은 실험 시작값일 뿐 성공률을 보장하지 않습니다.

대조군도 같은 설정으로 돌리려면 `03_compare_common_from_zero.cmd`를 실행하거나:

```bat
python -m tutorial_2.tutorial_test.train_ppo_v4 train --stage 2 --reward common --steps 500000 --seed 42
python -m tutorial_2.tutorial_test.train_ppo_v4 train --stage 4 --reward common --steps 500000 --seed 42
```

## 4. 훈련 및 평가

- 이전 v3보다 긴 `n_steps=512`, PPO 학습률 `1e-4`, 3 epochs, clipping `0.1`, `target_kl=0.01`, `gae_lambda=0.995`.
- 엔트로피는 검증 성공률이 **3회 연속 60% 이상**이기 전에는 높은 값 `0.03`을 유지하고 이후 `0.01`로 감소시킵니다.
- 5만 스텝마다 평가(`--eval-every`), 성공률/착지 진행에 대한 2개 최고 체크포인트를 저장합니다.
- 별도의 100회 평가(`--selection-trials`)로 두 체크포인트 중 하나를 최종 선택한 뒤 **다른 시드**에서 독립 200회 평가(`--test-trials`). 성공률과 실패 원인을 JSON으로 기록합니다. 이들은 **내 환경에서 얻은 실제 PPO 결과가 아니며**, 사용자 컴퓨터에서 훈련 후 생성됩니다.
- 저장 경로: `tutorial_test/results_v4/stage_02/guided_scratch/seed_42/` 및 `stage_04/...`. 결과 폴더가 이미 있으면 자동 덮어쓰지 않고 중단합니다. 다시 0부터라면 `--root`에 새로운 이름을 지정하세요.

```bat
python -m tutorial_2.tutorial_test.result_summary_v4
python -m tutorial_2.tutorial_test.train_ppo_v4 evaluate --folder tutorial_2\tutorial_test\results_v4\stage_02\guided_scratch\seed_42 --trials 500
```

기대하는 결과가 나오지 않으면 `summary.json`, `selection.json`, `validations.csv`, `final_evaluation.json`을 보고 추가 실험을 결정해야 합니다. 미검증 상태에서 70~80% 성공을 보장하지 않습니다.

## 5. 제작 환경에서 실제 확인한 범위

- **코드 구문 검증 통과.**
- **순수 보상·보상 합산 테스트 29개 통과.** 1~5 실제 게임을 불러오는 4개 통합 시험은 패키지 부족으로 실행되지 않아 **skip** 처리됨. 총 33개 테스트 중 29 통과/4 skip.
- `gymnasium`, `stable-baselines3`, `pygame`의 설치를 시도했으나 현재 도구 실행 환경의 외부 패키지 서버 DNS 접속이 실패함.
- GitHub 저장소의 `rl/milestone_env.py`, `rl/recovery_train.py`, `pingu_ball/special_lab.py`, 단계별 기존 보상 로직을 **읽어 확인**했지만 이 제작 환경에서 원본 게임 전체와 PPO를 실행해 모델 성공률을 재현하지는 못했음.
- **실제 2·4단계 학습 성공률: 미측정**. 시험을 통과했다고 PPO 성능이 좋다는 뜻은 아닙니다.

## 6. 무결성

`config.json`에 원본 팀 코드 및 새 실험 코드 SHA256, 단계, 시드, 보상 종류를 저장합니다. 팀 게임 소스가 훈련 중 달라지면 `evaluate`가 이를 감지합니다. 1~5단계 공통 엔진을 직접 수정하지 않으므로 기존 팀원 학습 작업을 방해하지 않습니다.
