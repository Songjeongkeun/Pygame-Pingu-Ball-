"""Pure helpers for safe 12-to-6 PPO transfer, no game dependencies."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

SOURCE_RELATIVE = Path('results/reward_repeat_2_4/stage_02/personal/repeat_12/seed_42/best.zip')
OUT_RELATIVE = Path('results/reward_transfer_2_6/stage_02/personal/from_repeat_12/seed_42')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for buf in iter(lambda: f.read(1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def read_source(path: Path, obs_size: int) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'12-frame PPO checkpoint not found: {path}')
    if path.suffix.lower() != '.zip':
        raise ValueError('Expected Stable-Baselines3 PPO .zip checkpoint')
    config_file = path.with_name('config.json')
    if not config_file.is_file():
        raise FileNotFoundError(f'Missing source PPO config: {config_file}')
    config = json.loads(config_file.read_text(encoding='utf-8'))
    required = dict(stage=2, reward='personal', repeat=12, action_count=3,
                    common_observation=int(obs_size))
    for k, expected in required.items():
        if config.get(k) != expected:
            raise ValueError(f'Incompatible SOURCE config: {k}={config.get(k)!r}; expected {expected!r}')
    if config.get('BC', True) or config.get('DAgger', True) or config.get('pretrained', True):
        raise ValueError('Source must be a scratch-trained pure PPO')
    return dict(checkpoint=str(path), sha256=sha256_file(path), source_config=config)


def make_config(source: dict, *, seed: int, lr: float, entropy: float,
                validation_trials: int, validation_seed: int,
                eval_every_frames: int, obs_size: int) -> dict:
    if seed < 0 or lr <= 0 or not (0 <= entropy <= .2) or validation_trials < 1 or eval_every_frames < 1:
        raise ValueError('Invalid transfer settings')
    return dict(stage=2, reward='personal', source_repeat=12, target_repeat=6,
                source_checkpoint=str(Path('tutorial_2') / SOURCE_RELATIVE), source_sha256=source['sha256'],
                source_training_seed=source['source_config'].get('seed'),
                adaptation_seed=seed, action_count=3, common_observation=int(obs_size),
                model='rl.milestone_models.MilestonePPO', from_scratch=False,
                PPO_checkpoint_warm_start=True, BC=False, DAgger=False, teacher=False,
                demonstrations=False, pretrained_external=False, physics_hz=120,
                ppo=dict(gamma=1., gae_lambda=.98, n_steps=128, batch_size=256,
                         n_epochs=5, clip_range=.2, target_kl=.02,
                         learning_rate=float(lr), ent_coef=float(entropy)),
                validation_trials=validation_trials, validation_seed=validation_seed,
                eval_every_frames=eval_every_frames)


def update_learning_schedule(model, lr: float):
    """SB3 normally keeps the source schedule; replace it explicitly for PPO adaptation."""
    if not (0 < lr <= 0.01):
        raise ValueError('Bad learning rate')
    model.learning_rate = float(lr)
    model._setup_lr_schedule()
    for group in model.policy.optimizer.param_groups:
        group['lr'] = float(lr)


def make_monitor_args(cli_args):
    """Adapt transfer CLI arguments to reward_repeat_2_4.PhysicsMonitor's API.

    The parent monitor needs stage/reward/repeat in addition to its normal
    validation and budget flags. These are *fixed* for this stage-2 transfer:
    stage=2, reward='personal', repeat=6. Never derive them from a checkpoint.
    """
    required = ('physics_budget', 'validation_trials', 'validation_seed',
                'eval_every_frames', 'no_early_stop', 'collapse_entropy')
    for attr in required:
        if not hasattr(cli_args, attr):
            raise ValueError(f'Missing monitor argument: {attr}')
    data = dict(vars(cli_args))
    data.update(stage=2, reward='personal', repeat=6)
    return SimpleNamespace(**data)


def prepare_fresh_output(folder: Path, config: dict) -> bool:
    """Prepare a *new* transfer run, optionally recovering a config-only crash.

    Returns True if a previous failed attempt left only config.json (and
    optionally before_transfer.json). NEVER removes checkpoints or training
    history. If any other file exists, require an explicit --resume or --out.
    """
    folder = Path(folder)
    settings = folder / 'config.json'
    if folder.exists():
        files = {f.name for f in folder.iterdir()}
        allowed = {'config.json', 'before_transfer.json'}
        unexpected = files - allowed
        if unexpected:
            raise FileExistsError(
                f'Existing transfer data must be preserved ({sorted(unexpected)}). '
                f'Use --resume or --out another path: {folder}')
        if settings.is_file():
            try:
                old_config = json.loads(settings.read_text(encoding='utf-8'))
            except (ValueError, OSError) as ex:
                raise ValueError(f'Unreadable prior config: {settings}') from ex
            if old_config != config:
                raise ValueError(
                    f'Existing transfer config differs from this run; '
                    f'use --out another path, do not overwrite: {settings}')
            return True
    else:
        folder.mkdir(parents=True, exist_ok=False)
    # No previous model/checkpoint/history to protect; record immutable setup.
    settings.write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str)+'\n', encoding='utf-8')
    return False


def equivalent_saved_config(saved: dict, current: dict) -> bool:
    """Compare old/new transfer config across a directory move.

    Historical source_checkpoint was a Windows absolute path. That field is
    provenance, not the checkpoint loaded at runtime; the SHA256 MUST agree.
    """
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return False
    a, b = dict(saved), dict(current)
    a.pop('source_checkpoint', None)
    b.pop('source_checkpoint', None)
    return a == b
