"""Tests which do not require pygame/SB3, plus interface/dispatch regression."""
import ast
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

from tutorial_2.stage2_common_bridge import core
from tutorial_2.stage2_common_bridge.run import args_parser


def test_default_stage2_checkpoint_is_exact_and_not_team_output():
    assert core.STAGE2_MODEL.parts[-5:] == ('stage_02','personal','from_repeat_12','seed_42','best.zip')
    assert 'stage2_common_bridge' not in str(core.STAGE2_MODEL)


def test_command_verify_defaults_to_stage2_no_archived_rewrite():
    args = args_parser().parse_args(['verify'])
    assert args.steps == 100 and args.archive is False


def test_eval_defaults_to_only_stage2():
    args = args_parser().parse_args(['evaluate'])
    assert args.stage == '2' and args.episodes == 200


def test_dispatch_uses_original_team_launcher_with_correct_inputs(tmp_path):
    runner = core.PROJECT_ROOT / 'tools' / 'tutorial_rl.py'
    with patch.object(Path, 'is_file', lambda self: str(self) == str(runner)):
        with patch.object(subprocess,'run', return_value=SimpleNamespace(returncode=0)) as called:
            core.archive_command('evaluate',stage=3,episodes=123,seed=72,output=tmp_path/'out.json')
            cmd=called.call_args[0][0]
            assert str(runner) in cmd and '--stage' in cmd and cmd[cmd.index('--stage')+1]=='3'
            assert cmd[cmd.index('--episodes')+1]=='123'
            assert cmd[cmd.index('--algorithm')+1]=='ppo'
            assert cmd[cmd.index('--experiment')+1]=='baseline'


def test_stage2_never_dispatched_as_archived_team_model():
    import pytest
    with pytest.raises((ValueError, FileNotFoundError)):
        core.archive_command('evaluate',stage=2)


def test_archive_dispatch_never_invokes_training():
    import pytest
    with pytest.raises((ValueError, FileNotFoundError)):
        core.archive_command('train',stage=1)


def test_runner_has_no_model_saving_or_training_calls():
    source=(Path(__file__).resolve().parents[1]/'run.py').read_text(encoding='utf-8')
    tree=ast.parse(source)
    attrs=[n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)]
    assert 'learn' not in attrs
    assert 'save' not in attrs


def test_verify_runs_original_archive_verification_without_training():
    runner = core.PROJECT_ROOT / 'tools' / 'tutorial_rl.py'
    with patch.object(Path, 'is_file', lambda self: str(self) == str(runner)):
        with patch.object(subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as called:
            core.archive_command('verify')
            cmd = called.call_args[0][0]
            assert 'verify' in cmd
            assert 'train' not in cmd
            assert '--stage' not in cmd


def test_original_factory_has_only_stage_two_personal_override():
    source = (Path(__file__).resolve().parents[1] / 'core.py').read_text(encoding='utf8')
    tree = ast.parse(source)
    maker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'make_env')
    assert any(isinstance(n, ast.Compare) and isinstance(n.left, ast.Name)
               and n.left.id == 'stage' for n in ast.walk(maker))
    assert 'if stage == 2:' in source
    assert 'RewardOnlyEnv(2, repeat=6)' in source
    assert 'MilestoneEnv(stage=stage, limit=180., repeat=6)' in source


def test_both_evaluation_and_play_route_stage_two_separately():
    source = (Path(__file__).resolve().parents[1] / 'run.py').read_text(encoding='utf8')
    assert 'if stage == 2:' in source
    assert 'evaluate_stage2(model, args.episodes, args.seed)' in source
    assert 'play_stage2(model, args.seed, args.greedy)' in source
