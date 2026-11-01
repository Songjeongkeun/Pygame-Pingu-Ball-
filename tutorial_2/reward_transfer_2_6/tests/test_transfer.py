"""Tests that run without pygame/gymnasium/SB3; runtime physics smoke requires the game."""
from __future__ import annotations
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile
import pytest

from tutorial_2.reward_transfer_2_6.config_logic import (read_source, make_config,
                                               update_learning_schedule,
                                               sha256_file, make_monitor_args,
                                               prepare_fresh_output)

ROOT = Path(__file__).resolve().parents[1]


def checkpoint(tmp_path, **overrides):
    root = tmp_path/'stage_02'/'personal'/'repeat_12'/'seed_42'
    root.mkdir(parents=True,exist_ok=True)
    path = root/'best.zip'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('model','mock')
    cfg = dict(stage=2,reward='personal',repeat=12,action_count=3,
               common_observation=123,BC=False,DAgger=False,pretrained=False,seed=42)
    cfg.update(overrides)
    (root/'config.json').write_text(json.dumps(cfg),encoding='utf-8')
    return path


def test_source_contract_and_hash(tmp_path):
    path=checkpoint(tmp_path)
    r=read_source(path,123)
    assert r['sha256']==sha256_file(path)
    assert r['source_config']['repeat']==12
    assert r['checkpoint']==str(path.resolve())


@pytest.mark.parametrize('field,value',[('stage',4),('reward','original'),('repeat',6),
    ('action_count',9),('common_observation',40),('BC',True),('DAgger',True),('pretrained',True)])
def test_reject_wrong_source(tmp_path,field,value):
    path=checkpoint(tmp_path,**{field:value})
    with pytest.raises(ValueError):read_source(path,123)


def test_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):read_source(tmp_path/'best.zip',123)


def test_config_records_every_change_and_forbids_bc_dagger(tmp_path):
    src=read_source(checkpoint(tmp_path),123)
    cfg=make_config(src,seed=42,lr=.0001,entropy=.01,
                    validation_trials=20,validation_seed=50000,
                    eval_every_frames=60000,obs_size=123)
    assert cfg['target_repeat']==6 and cfg['source_repeat']==12
    assert cfg['PPO_checkpoint_warm_start'] is True and cfg['from_scratch'] is False
    assert cfg['BC'] is False and cfg['DAgger'] is False
    assert cfg['teacher'] is False and cfg['demonstrations'] is False
    assert cfg['ppo']['learning_rate']==.0001
    assert cfg['ppo']['ent_coef']==.01
    assert cfg['source_sha256']==src['sha256']


def test_lr_schedule_is_adapted_not_left_from_previous_training():
    class Model:
        def __init__(self):
            self.learning_rate=.0003
            self.policy=SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{'lr':.0003}]))
        def _setup_lr_schedule(self):
            self.lr_schedule=lambda progress:float(self.learning_rate)
    model=Model()
    update_learning_schedule(model,.0001)
    assert model.policy.optimizer.param_groups[0]['lr']==.0001
    assert model.lr_schedule(.5)==.0001


def test_only_changes_stage2_and_keeps_real_ppo():
    code=(ROOT/'train.py').read_text(encoding='utf-8')
    ast.parse(code)
    assert "base.make_vec(2, 'personal', 6)" in code
    assert "base.make_env(2, 'personal', 12)" in code
    assert "base.make_env(2, 'personal', 6)" in code
    assert 'MilestonePPO.load' in code
    assert "model.learn(" in code
    assert 'model.set_random_seed(args.seed)' in code
    assert 'update_learning_schedule(model,args.learning_rate)' in code
    assert 'model.ent_coef=float(args.entropy)' in code
    assert 'make_monitor_args(args)' in code
    assert 'prepare_fresh_output(folder, cfg)' in code
    assert 'model.policy.optimizer' not in code # managed in config_logic


def test_all_source_code_has_no_syntax_errors():
    for fn in (ROOT/'train.py', ROOT/'config_logic.py', ROOT/'__init__.py'):
        ast.parse(fn.read_text(encoding='utf-8'))


def transfer_cli_args(**overrides):
    data=dict(physics_budget=240000, validation_trials=20, validation_seed=50000,
              eval_every_frames=60000, no_early_stop=False, collapse_entropy=.1,
              seed=42, entropy=.01, learning_rate=.0001)
    data.update(overrides)
    return SimpleNamespace(**data)


def test_monitor_adapter_sends_stage_reward_and_repeat():
    input_args=transfer_cli_args()
    mon=make_monitor_args(input_args)
    assert mon.stage==2 and mon.reward=='personal' and mon.repeat==6
    assert mon.physics_budget==240000 and mon.validation_trials==20
    assert mon.eval_every_frames==60000 and mon.validation_seed==50000
    assert mon.no_early_stop is False and mon.collapse_entropy==.1
    assert not hasattr(input_args,'stage')   # the original CLI really lacks stage
    assert not hasattr(input_args,'reward')
    assert not hasattr(input_args,'repeat')


def test_recover_incomplete_config_only_run(tmp_path):
    folder=tmp_path/'output'
    cfg={'stage':2, 'target_repeat':6, 'seed':42}
    assert prepare_fresh_output(folder,cfg) is False
    assert (folder/'config.json').exists()
    assert prepare_fresh_output(folder,cfg) is True
    (folder/'before_transfer.json').write_text('{}',encoding='utf-8')
    assert prepare_fresh_output(folder,cfg) is True
    assert (folder/'before_transfer.json').exists()


def test_recover_does_not_overwrite_models_or_changed_experiment(tmp_path):
    folder=tmp_path/'output'
    cfg={'stage':2, 'target_repeat':6}
    prepare_fresh_output(folder,cfg)
    with pytest.raises(ValueError,match='differs'):
        prepare_fresh_output(folder,{'stage':2,'target_repeat':12})
    (folder/'last.zip').write_bytes(b'keep-this-checkpoint')
    with pytest.raises(FileExistsError,match='preserved'):
        prepare_fresh_output(folder,cfg)
    assert (folder/'last.zip').read_bytes()==b'keep-this-checkpoint'


def test_monitor_integration_with_real_parent_source(tmp_path):
    """Execute ACTUAL reward_repeat_2_4.PhysicsMonitor.checkpoint via AST.

    No SB3/Pygame needed for this callback test; the original class body is
    copied from the local project, not a handwritten fake of the callback.
    """
    import time
    parent_path=ROOT.parent/'reward_repeat_2_4'/'train.py'
    if not parent_path.exists():
        pytest.skip('Test needs reward_repeat_2_4 from the user project root')
    tree=ast.parse(parent_path.read_text(encoding='utf-8'),filename=str(parent_path))
    items=[node for node in tree.body if
           (isinstance(node,ast.ClassDef) and node.name=='PhysicsMonitor') or
           (isinstance(node,ast.FunctionDef) and node.name=='score')]
    assert len(items)==2
    class BaseCallback:
        def __init__(self,verbose=0):
            self.model=None
    def fake_evaluate(_model,stage,reward,repeat,trials,seed):
        assert (stage,reward,repeat,trials,seed)==(2,'personal',6,20,50000)
        return dict(stage=2,reward_mode='personal',success_rate=.1,wins=2,trials=20,
                    spring3_rate=.1, upper_rate=.2,spring2_rate=.3,
                    mean_target_landings=.4,first_spring_rate=.8,
                    middle_rate=.6,mean_max_x=700,mean_policy_entropy=.9,
                    mean_return=-5.)
    def save_json(path,value):
        path=Path(path)
        path.write_text(json.dumps(value),encoding='utf-8')
    ctx=dict(BaseCallback=BaseCallback,Path=Path,time=time,json=json,
             evaluate=fake_evaluate, save_json=save_json)
    exec(compile(ast.Module(body=items,type_ignores=[]),str(parent_path),'exec'),ctx)
    monitor=ctx['PhysicsMonitor'](tmp_path/'output',make_monitor_args(transfer_cli_args()))
    class Model:
        num_timesteps=5120
        ent_coef=.01
        def save(self,path):
            Path(path).write_bytes(b'test-weights')
    monitor.model=Model()
    monitor.physical_frames=60050
    rec=monitor.checkpoint()
    folder=tmp_path/'output'
    assert rec['validation']['wins']==2
    assert (folder/'best.zip').read_bytes()==b'test-weights'
    assert (folder/'last.zip').read_bytes()==b'test-weights'
    assert (folder/'latest.json').exists()
    assert (folder/'best_validation.json').exists()
