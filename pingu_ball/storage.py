"""Local, atomic best-record persistence; only cleared attempts are recorded."""
import json
from pathlib import Path
import os


class Progress:
    def __init__(self, path):
        self.path = Path(path)
        self.records = {}
        self.unlocked = 0
        self.error = ''
        self.load()

    def load(self):
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
            if data.get('version') != 1:
                return
            self.unlocked = max(0,min(7,int(data.get('unlocked',0))))
            self.records = {str(k):v for k,v in data.get('records',{}).items()
                            if str(k).isdigit() and 0<=int(k)<=7 and isinstance(v,dict)
                            and isinstance(v.get('time'),(float,int)) and v['time']>=0
                            and (int(k)==0 or isinstance(v.get('score'),int) and 6<=v['score']<=10)}
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError, AttributeError):
            self.error = '저장 파일을 읽지 못해 새 기록으로 시작합니다.'

    def complete(self, index, score, elapsed):
        old = self.records.get(str(index))
        improved = old is None or (index != 0 and score > old['score']) or (score==old.get('score') and elapsed < old['time'])
        if improved:
            self.records[str(index)]={'score':score,'time':round(elapsed,6)}
        self.unlocked=max(self.unlocked,min(7,index+1))
        self.save()
        return improved

    def save(self):
        try:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            tmp=self.path.with_suffix('.tmp')
            tmp.write_text(json.dumps({'version':1,'unlocked':self.unlocked,'records':self.records},ensure_ascii=False,indent=2),encoding='utf-8')
            os.replace(tmp,self.path)
            self.error=''
        except OSError:
            self.error='기록 저장에 실패했습니다. 폴더 쓰기 권한을 확인해 주세요.'
