"""서로 다른 보상 실험의 실제 독립평가 결과만 표로 요약한다."""
from pathlib import Path
import argparse
import json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent/'results_v4')
    args=p.parse_args()
    rows=[]
    for file in sorted(args.root.glob('stage_*/**/summary.json')):
        data=json.loads(file.read_text(encoding='utf-8'))
        cfg=data['config']; ev=data['final']
        rows.append((cfg['stage'],cfg['reward'],cfg['seed'],cfg['additional_steps'],
                     ev['success_pct'],ev['wins'],ev['trials'],ev['mean_milestones'],
                     file.parent))
    if not rows:
        print('summary.json을 찾지 못했습니다. 아직 실제 학습을 완료하지 않은 상태입니다.')
        return
    print('스테이지   보상       seed    steps      독립평가        마일스톤   폴더')
    for stage,reward,seed,steps,pct,wins,trials,miles,folder in rows:
        print(f'{stage:>7}   {reward:<8}   {seed:>4}   {steps:>7}   '
              f'{wins:>4}/{trials:<4} {pct:>5.1f}%   {miles or 0:>7.2f}   {folder}')
    print('※ 검증 시드와 별도인 독립평가 결과이며, 이 표만으로 성능 개선을 확정하지는 않습니다.')


if __name__=='__main__':main()
