"""Atomic local storage; only successful attempts update the best record."""
import json
import math
import os
from pathlib import Path
from .config import CAMPAIGN_ID, REGULAR_STAGE_COUNT


class Progress:
    def __init__(self, path):
        self.path = Path(path)
        self.records = {}
        self.tutorial_records = {}
        self.unlocked = 1
        self.error = ''
        self.load()

    def load(self):
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
            if not isinstance(data, dict) or data.get('version') not in (1, 2, 3, 4):
                raise ValueError('unsupported save format')
            unlocked = data.get('unlocked', 0)
            if type(unlocked) is not int or not 0 <= unlocked <= REGULAR_STAGE_COUNT:
                raise ValueError('invalid unlock state')
            records = data.get('records', {})
            if not isinstance(records, dict):
                raise ValueError('invalid records')
            valid = {}
            for key, record in records.items():
                if key not in {str(i) for i in range(REGULAR_STAGE_COUNT+1)} or not isinstance(record, dict):
                    continue
                if key == '0':
                    # Version 1 stored a tutorial time. Version 2 stores completion only.
                    if data['version'] == 1 or record.get('cleared') is True:
                        valid[key] = {'cleared': True}
                elif (type(record.get('score')) is int and 6 <= record['score'] <= 10
                      and type(record.get('time')) in (int, float)
                      and math.isfinite(record['time']) and record['time'] >= 0):
                    valid[key] = {'score': record['score'], 'time': record['time']}
            current_campaign = data.get('version') == 4 and data.get('campaign') == CAMPAIGN_ID
            # Retired regular stage numbers refer to different maps. Keep tutorial
            # completion and times, while starting the replacement campaign at 1.
            self.records = valid if current_campaign else {k:v for k,v in valid.items() if k == '0'}
            self.unlocked = max(1, unlocked) if current_campaign else 1
            tutorials = data.get('tutorial_records',{})
            if isinstance(tutorials,dict):
                self.tutorial_records = {key:float(elapsed) for key,elapsed in tutorials.items()
                                         if key in {str(i) for i in range(1,11)} and type(elapsed) in (int,float)
                                         and math.isfinite(elapsed) and elapsed>=0}
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError, AttributeError):
            self.error = '저장 파일을 읽지 못해 새 기록으로 시작합니다.'

    def complete(self, index, score, elapsed):
        if type(index) is not int or index not in range(REGULAR_STAGE_COUNT+1):
            raise ValueError('invalid stage')
        if index and (type(score) is not int or score not in range(6, 11)
                      or not math.isfinite(elapsed) or elapsed < 0):
            raise ValueError('invalid cleared attempt')
        old = self.records.get(str(index))
        improved = old is None
        if index:
            improved = improved or score > old['score'] or (score == old['score'] and elapsed < old['time'])
            if improved:
                self.records[str(index)] = {'score': score, 'time': round(elapsed, 6)}
        else:
            self.records['0'] = {'cleared': True}
        self.unlocked = max(self.unlocked, min(REGULAR_STAGE_COUNT, index+1))
        self.save()
        return improved

    def save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix('.tmp')
            temporary.write_text(json.dumps({'version': 4, 'campaign': CAMPAIGN_ID, 'unlocked': self.unlocked,
                                            'records': self.records, 'tutorial_records':self.tutorial_records}, ensure_ascii=False, indent=2,
                                           allow_nan=False), encoding='utf-8')
            os.replace(temporary, self.path)
            self.error = ''
        except OSError:
            self.error = '기록 저장에 실패했습니다. 폴더 쓰기 권한을 확인해 주세요.'

    def complete_tutorial(self, number, elapsed):
        if type(number) is not int or number not in range(1,11) or type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed<0:
            raise ValueError('invalid tutorial attempt')
        key=str(number)
        old=self.tutorial_records.get(key)
        improved=old is None or elapsed<old
        if improved:
            self.tutorial_records[key]=round(elapsed,6)
        self.unlocked=max(self.unlocked,1)
        self.save()
        return improved
