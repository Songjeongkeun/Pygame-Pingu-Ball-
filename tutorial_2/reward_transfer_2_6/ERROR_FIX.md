# 6프레임 PPO 전환 학습 오류 수정 (2026-11-01)

## 발생한 오류

이전 버전에서

```text
AttributeError: 'Namespace' object has no attribute 'stage'
```

가 발생했다. 원인은 `reward_transfer_2_6.train`의 인자에는 `stage`, `reward`, `repeat`가 없지만, 공통 학습기 `reward_repeat_2_4.train.PhysicsMonitor.checkpoint()`가 해당 값을 요구하기 때문이다. 이 오류가 검증 직전에 발생하여 새 `best.zip`이 생성되지 않았으므로, 뒤이어 실행한 평가의 `FileNotFoundError`는 파생 오류다.

## 수정 사항

1. 2단계 전용 PPO 전환용 검증 인자를 명시적으로 준비한다: `stage=2`, `reward='personal'`, `repeat=6`. `PhysicsMonitor`에는 이 인자와 기존 검증/물리예산 인자를 모두 전달한다.
2. 이전 실패로 **`config.json`만 남았을 때**, 기존 설정과 현재 설정이 완전히 일치하고 체크포인트·검증 이력이 전혀 없다면 자동으로 재시도한다. 이때 `[RECOVER]` 메시지를 출력한다. 설정이 다르거나 모델/이력 파일이 있으면 파일을 보호하기 위해 자동 복구를 거부한다.
3. 원본의 12프레임 `best.zip`에는 쓰기 작업을 하지 않는다. 팀 1·3·5단계 모델/코드를 수정하지 않는다. 행동 3개·공통 관측·PPO 학습은 그대로다. BC/DAgger는 사용하지 않는다.

## 설치

프로젝트 **루트**에서 이 ZIP의 `reward_transfer_2_6/` 폴더를 기존 폴더에 덮어쓴다. **`results/` 폴더는 삭제하거나 덮어쓰지 않는다.**

기존 명령을 **그대로 다시 실행**하면 된다. 실패한 첫 실행에는 저장된 모델이 없으므로 `--resume`을 붙이지 않는다.

```cmd
python -m tutorial_2.reward_transfer_2_6.train train --physics-budget 240000 --seed 42
```

`[RECOVER] ...`가 나오면 이전의 `config.json`만 남은 실행을 확인하고 다시 시작한 것이다. 이후 첫 `[CHECK]`가 정상적으로 출력되는지 확인한다.

학습이 정상 종료되고 새 `best.zip`이 생긴 뒤에만 다음 평가를 진행한다.

```cmd
python -m tutorial_2.reward_transfer_2_6.train evaluate --trials 200 --eval-seed 120000
```

### 예외 상황

- `Existing transfer data must be preserved`: 이전 결과(모델/검증 이력)가 있으므로 자동 덮어쓰지 않는다. 정상적인 이어 학습이라면 `--resume` 사용. 다른 실험이면 `--out`으로 별도 경로 지정.
- `Existing transfer config differs`: 이전 `config.json`의 실험 조건이 다르므로 별도 `--out` 경로 사용.
- 실제 물리 게임 환경 의존성(`pygame`, `gymnasium`, `stable-baselines3`)이 없는 작성 환경에서는 게임 학습을 직접 실행하지 못했다. 정적 검사와 원본 PhysicsMonitor 실행을 포함한 회귀 테스트 18개 통과.
