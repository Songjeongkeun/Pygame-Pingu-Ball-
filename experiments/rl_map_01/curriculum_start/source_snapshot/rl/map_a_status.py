"""실행 중인 맵1 학습의 저장 기록만 읽어 짧은 진행 상황을 출력한다."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args = parser.parse_args();root = args.root
    plan = json.loads((root/'plan.json').read_text(encoding='utf-8'))
    print(f'경과 {(time.time()-(root/"plan.json").stat().st_mtime)/60:.1f}분 · 예산 모델당 {plan["physics_budget"]:,} 물리 프레임')
    for seed in plan['seeds']:
        folder = root/f'seed_{seed}';path = folder/'episodes.jsonl'
        if not path.exists():continue
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines(keepends=True) if line.endswith('\n')]
        if not rows:continue
        live = json.loads((folder/'live.json').read_text(encoding='utf-8'))
        ticks = max(rows[-1]['physics_ticks'],live['physics_ticks'])
        print(f'시드{seed}: {ticks:,}틱 / 종료 {len(rows):,}판 / 성공 {sum(r["success"] for r in rows)}판 / '
              f'한 판 최대 구간 {max(len(r["milestones"]) for r in rows)}/16 / 최근 검증 {live["validation_success_rate"]*100:.0f}%' +
              (' / 완료' if (folder/'summary.json').exists() else ''))


if __name__=='__main__':main()
