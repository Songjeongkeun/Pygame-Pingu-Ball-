"""Offline regression checks for the Git repository's tutorial_2/ layout."""
from pathlib import Path
import ast
import json
import zipfile

from tutorial_2.stage2_common_bridge import core
from tutorial_2.reward_transfer_2_6.config_logic import (
    SOURCE_RELATIVE, OUT_RELATIVE, read_source, make_config,
    equivalent_saved_config,
)


def test_paths_relocate_under_tutorial2():
    root = Path(__file__).resolve().parents[1]
    assert core.TUTORIAL_ROOT == root
    assert core.PROJECT_ROOT == root.parent
    assert core.STAGE2_MODEL == root / OUT_RELATIVE / 'best.zip'
    assert core.RESULTS == root / 'results' / 'stage2_common_bridge'


def test_final_and_source_models_exist_and_are_unchanged_archives():
    root = Path(__file__).resolve().parents[1]
    source = root / SOURCE_RELATIVE
    target = core.STAGE2_MODEL
    assert source.is_file() and target.is_file()
    for p in (source, target):
        with zipfile.ZipFile(p) as model:
            assert model.testzip() is None
            assert 'policy.pth' in model.namelist()
    original = read_source(source, obs_size=1664)
    target_config = json.loads((target.parent / 'config.json').read_text(encoding='utf8'))
    assert target_config['source_sha256'] == original['sha256']
    assert target_config['target_repeat'] == 6
    assert target_config['action_count'] == 3
    assert target_config['common_observation'] == 1664
    assert target_config['BC'] is False and target_config['DAgger'] is False


def test_transfer_resume_config_is_portable():
    root = Path(__file__).resolve().parents[1]
    source = read_source(root / SOURCE_RELATIVE, 1664)
    current = make_config(source, seed=42, lr=.0001, entropy=.01,
                          validation_trials=20, validation_seed=50000,
                          eval_every_frames=60000, obs_size=1664)
    saved = json.loads((root / OUT_RELATIVE / 'config.json').read_text(encoding='utf8'))
    assert equivalent_saved_config(saved, current)
    assert saved['source_checkpoint'] == 'tutorial_2/' + SOURCE_RELATIVE.as_posix()


def test_python_sources_compile():
    root = Path(__file__).resolve().parents[1]
    count = 0
    for p in root.rglob('*.py'):
        ast.parse(p.read_text(encoding='utf8'), filename=str(p))
        count += 1
    assert count >= 30
