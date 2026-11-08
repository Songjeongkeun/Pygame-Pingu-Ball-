"""실행 진입점과 관계없이 저장한 정책과 수치 학습률을 복원한다."""
from .stabilized_models import StabilizedPPO


def load_policy(checkpoint, device='cpu', env=None):
    # CLI의 __main__에 있던 임시 스케줄 객체는 읽지 않는다.
    # PPO의 초기화가 저장된 수치 learning_rate로 같은 학습률 함수를 재구성한다.
    # 정책·가치 함수·최적화기 가중치는 원래 체크포인트에서 그대로 읽는다.
    return StabilizedPPO.load(checkpoint,device=device,env=env,
                              custom_objects={'lr_schedule':None})
