"""Keep the team's common observation, actions, physics and original stages untouched.

Stage 2 replaces *only* the milestone reward with the user's personal spring reward.
All other stages retain the team's MilestoneEnv; archived model playback for those
stages uses the team's own frozen-runtime tools/tutorial_rl.py rather than this env.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # team repository root
TUTORIAL_ROOT = Path(__file__).resolve().parents[1]  # tutorial_2/
STAGE2_MODEL = (TUTORIAL_ROOT / 'results' / 'reward_transfer_2_6' /
                'stage_02' / 'personal' / 'from_repeat_12' / 'seed_42' / 'best.zip')
RESULTS = TUTORIAL_ROOT / 'results' / 'stage2_common_bridge'


def make_env(stage: int):
    """For NEW common PPO experiences. Always 3 discrete actions, 6 physics ticks."""
    from rl.milestone_env import MilestoneEnv, OBS_SIZE
    if stage not in range(1, 6):
        raise ValueError('Only team tutorial stages 1..5')
    if stage == 2:
        from tutorial_2.reward_repeat_2_4.reward_env import RewardOnlyEnv
        env = RewardOnlyEnv(2, repeat=6)
        actual_repeat = env.team.unwrapped.repeat
    else:
        env = MilestoneEnv(stage=stage, limit=180., repeat=6)
        actual_repeat = env.unwrapped.repeat
    if env.action_space.n != 3 or env.observation_space.shape != (OBS_SIZE,) or actual_repeat != 6:
        env.close()
        raise AssertionError('Team common observation/action/repeat mismatch')
    return env


def load_stage2_model(path=STAGE2_MODEL):
    from rl.milestone_train import install
    from rl.milestone_models import MilestonePPO, MilestoneExtractor
    from rl.milestone_env import OBS_SIZE
    from stable_baselines3.common.vec_env import DummyVecEnv
    if not Path(path).is_file():
        raise FileNotFoundError(f'Stage 2 6-frame best model not found: {path}')
    install()
    vec = DummyVecEnv([lambda: make_env(2)])
    try:
        model = MilestonePPO.load(str(path), env=vec, device='cpu')
        if (model.action_space.n != 3 or model.observation_space.shape != (OBS_SIZE,)
            or not isinstance(model.policy.features_extractor, MilestoneExtractor)):
            raise ValueError('Stage 2 model is NOT the common 3-action MilestonePPO')
        model.policy.set_training_mode(False)
        return model, vec
    except BaseException:
        vec.close()
        raise


def archive_command(command: str, stage: int | None = None, episodes=100, seed=120000,
                    output: Path | None = None, greedy=False):
    """Team's original frozen-runtime runner; never load archives into current rl/."""
    import subprocess
    import sys
    team_runner = PROJECT_ROOT / 'tools' / 'tutorial_rl.py'
    if not team_runner.is_file():
        raise FileNotFoundError(f'Team archive launcher missing: {team_runner}')
    if command not in ('verify', 'evaluate', 'play'):
        raise ValueError(command)
    args = [sys.executable, '-B', str(team_runner), command]
    if command != 'verify':
        if stage not in range(1, 6) or stage == 2:
            raise ValueError('Archive dispatch is for 1, 3, 4, 5 only')
        args += ['--stage', str(stage), '--algorithm', 'ppo', '--experiment', 'baseline',
                 '--seed', str(seed)]
        if command == 'evaluate':
            args += ['--episodes', str(episodes)]
            if output is not None:
                args += ['--output', str(output.resolve())]
        if greedy:
            if command != 'play':
                raise ValueError('greedy applies to play only')
            args.append('--greedy')
    return subprocess.run(args, cwd=PROJECT_ROOT, check=True)
