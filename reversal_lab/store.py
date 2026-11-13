from __future__ import annotations
import json,sqlite3,time,uuid,hashlib
from contextlib import contextmanager
from pathlib import Path
import pandas as pd
from .data import fingerprint,finite_records

class Conflict(ValueError):pass

class Store:
    def __init__(self,root: str|Path):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.model_dir=self.root/'models';self.model_dir.mkdir(exist_ok=True);self.db=self.root/'labels.sqlite3'
        with self.connect() as c:
            c.executescript('''
            CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY,name TEXT NOT NULL,symbol TEXT NOT NULL,fingerprint TEXT NOT NULL UNIQUE,rows INTEGER,start INTEGER,end INTEGER,revision INTEGER DEFAULT 0,created REAL);
            CREATE TABLE IF NOT EXISTS candles(dataset TEXT NOT NULL,idx INTEGER NOT NULL,timestamp INTEGER NOT NULL,open REAL,high REAL,low REAL,close REAL,volume REAL,PRIMARY KEY(dataset,idx),UNIQUE(dataset,timestamp),FOREIGN KEY(dataset) REFERENCES datasets(id));
            CREATE TABLE IF NOT EXISTS labels(id TEXT PRIMARY KEY,dataset TEXT NOT NULL,timestamp INTEGER NOT NULL,kind TEXT CHECK(kind IN ('bottom','top')),radius INTEGER NOT NULL,notes TEXT NOT NULL,created REAL,updated REAL,UNIQUE(dataset,timestamp),FOREIGN KEY(dataset) REFERENCES datasets(id));
            CREATE TABLE IF NOT EXISTS reviews(id TEXT PRIMARY KEY,dataset TEXT NOT NULL,start INTEGER NOT NULL,end INTEGER NOT NULL,UNIQUE(dataset,start,end),FOREIGN KEY(dataset) REFERENCES datasets(id));
            CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,dataset TEXT,revision INTEGER,status TEXT,config TEXT,result TEXT,error TEXT,created REAL,updated REAL);
            CREATE TABLE IF NOT EXISTS models(id TEXT PRIMARY KEY,dataset TEXT,revision INTEGER,metadata TEXT,created REAL);
            ''')
            if 'parent' not in {x['name'] for x in c.execute('PRAGMA table_info(datasets)')}:c.execute('ALTER TABLE datasets ADD COLUMN parent TEXT REFERENCES datasets(id)')
    @contextmanager
    def connect(self):
        c=sqlite3.connect(self.db,timeout=30);c.row_factory=sqlite3.Row;c.execute('PRAGMA foreign_keys=ON');c.execute('PRAGMA journal_mode=WAL')
        try:
            yield c;c.commit()
        except Exception:
            c.rollback();raise
        finally:c.close()
    def dataset(self,id,c=None):
        if c is None:
            with self.connect() as conn:return self.dataset(id,conn)
        row=c.execute('SELECT * FROM datasets WHERE id=?',(id,)).fetchone()
        if row is None:raise KeyError('데이터셋을 찾을 수 없습니다.')
        return dict(row)
    def _insert_frame(self,c,frame,name,symbol,parent=None):
        fp=fingerprint(frame);id=uuid.uuid4().hex
        c.execute('INSERT INTO datasets(id,name,symbol,fingerprint,rows,start,end,revision,created,parent) VALUES(?,?,?,?,?,?,?,?,?,?)',(id,name,symbol,fp,len(frame),int(frame.timestamp.iloc[0]),int(frame.timestamp.iloc[-1]),0,time.time(),parent))
        c.executemany('INSERT INTO candles VALUES(?,?,?,?,?,?,?,?)',((id,i,int(r.timestamp),float(r.open),float(r.high),float(r.low),float(r.close),None if pd.isna(r.volume) else float(r.volume)) for i,r in enumerate(frame.itertuples())))
        return self.dataset(id,c)
    def import_frame(self,frame,name='BTCUSDT 15m',symbol='BTCUSDT'):
        fp=fingerprint(frame)
        with self.connect() as c:
            old=c.execute('SELECT id FROM datasets WHERE fingerprint=?',(fp,)).fetchone()
            if old:return self.dataset(old['id'],c)
            return self._insert_frame(c,frame,name,symbol)
    def extend_frame(self,id,frame):
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE');source=self.dataset(id,c)
            if len(frame)<=source['rows']:raise ValueError('기존 전체 이력에 새 봉을 추가한 CSV가 필요합니다.')
            if fingerprint(frame.iloc[:source['rows']])!=source['fingerprint']:raise ValueError('기존 OHLCV 이력이 달라 이어 붙일 수 없습니다. 별도 데이터셋으로 가져오세요.')
            old=c.execute('SELECT id,parent FROM datasets WHERE fingerprint=?',(fingerprint(frame),)).fetchone()
            if old:
                if old['parent']==id:return self.dataset(old['id'],c)
                raise ValueError('같은 데이터가 이미 별도로 존재합니다. 기존 데이터셋을 선택하세요.')
            child=self._insert_frame(c,frame,source['name']+' + 새 봉',source['symbol'],id);childid=child['id']
            for row in c.execute('SELECT * FROM labels WHERE dataset=?',(id,)).fetchall():c.execute('INSERT INTO labels VALUES(?,?,?,?,?,?,?,?)',(uuid.uuid4().hex,childid,row['timestamp'],row['kind'],row['radius'],row['notes'],row['created'],row['updated']))
            for row in c.execute('SELECT * FROM reviews WHERE dataset=?',(id,)).fetchall():c.execute('INSERT INTO reviews VALUES(?,?,?,?)',(uuid.uuid4().hex,childid,row['start'],row['end']))
            c.execute('UPDATE datasets SET revision=? WHERE id=?',(source['revision'],childid));return self.dataset(childid,c)
    def ancestors(self,id):
        ids=[]
        with self.connect() as c:
            while id:
                if id in ids:raise ValueError('데이터 이력 연결이 잘못됐습니다.')
                ids.append(id);id=self.dataset(id,c).get('parent')
        return ids
    @staticmethod
    def annotation_fingerprint(snapshot):
        semantic=dict(labels=sorted((x['timestamp'],x['kind'],x['radius'],x['notes']) for x in snapshot['labels']),reviews=sorted(set((x['start'],x['end']) for x in snapshot['reviews'])))
        return hashlib.sha256(json.dumps(semantic,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    def datasets(self):
        with self.connect() as c:return [dict(x) for x in c.execute('SELECT * FROM datasets ORDER BY created DESC')]
    def frame(self,id):
        with self.connect() as c:
            self.dataset(id,c);return pd.read_sql_query('SELECT idx,timestamp,open,high,low,close,volume FROM candles WHERE dataset=? ORDER BY idx',c,params=(id,))
    def candles(self,id,offset,limit,at=None):
        with self.connect() as c:
            meta=self.dataset(id,c)
            if at is not None:
                r=c.execute('SELECT idx FROM candles WHERE dataset=? ORDER BY abs(timestamp-?) LIMIT 1',(id,at)).fetchone();offset=max(0,int(r['idx'])-limit//2)
            offset=max(0,min(offset,max(0,meta['rows']-limit)))
            frame=pd.read_sql_query('SELECT idx,timestamp,open,high,low,close,volume FROM candles WHERE dataset=? AND idx>=? ORDER BY idx LIMIT ?',c,params=(id,offset,limit))
            return dict(dataset=meta,offset=offset,candles=finite_records(frame))
    def snapshot(self,id):
        with self.connect() as c:
            c.execute('BEGIN');meta=self.dataset(id,c)
            labels=[dict(x) for x in c.execute('SELECT * FROM labels WHERE dataset=? ORDER BY timestamp',(id,))]
            reviews=[dict(x) for x in c.execute('SELECT * FROM reviews WHERE dataset=? ORDER BY start',(id,))]
            return dict(schema_version=1,dataset=meta,labels=labels,reviews=reviews)
    def _revision(self,c,id,revision):
        c.execute('BEGIN IMMEDIATE');meta=self.dataset(id,c)
        if meta['revision']!=revision:raise Conflict('다른 창에서 라벨이 바뀌었습니다. 새로고침 후 다시 시도하세요.')
    def _timestamp(self,c,id,t):
        if c.execute('SELECT 1 FROM candles WHERE dataset=? AND timestamp=?',(id,t)).fetchone() is None:raise ValueError('해당 시각의 캔들이 없습니다.')
    def label(self,id,t,kind,radius,notes,revision):
        if kind not in ('bottom','top') or not 0<=radius<=5 or len(notes)>1000:raise ValueError('라벨 종류/구간/메모를 확인하세요.')
        with self.connect() as c:
            self._revision(c,id,revision);self._timestamp(c,id,t);now=time.time()
            c.execute('INSERT INTO labels VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(dataset,timestamp) DO UPDATE SET kind=excluded.kind,radius=excluded.radius,notes=excluded.notes,updated=excluded.updated',(uuid.uuid4().hex,id,t,kind,radius,notes,now,now));c.execute('UPDATE datasets SET revision=revision+1 WHERE id=?',(id,))
        return self.snapshot(id)
    def delete_label(self,id,labelid,revision):
        with self.connect() as c:
            self._revision(c,id,revision);cursor=c.execute('DELETE FROM labels WHERE dataset=? AND id=?',(id,labelid))
            if not cursor.rowcount:raise KeyError('라벨을 찾을 수 없습니다.')
            c.execute('UPDATE datasets SET revision=revision+1 WHERE id=?',(id,))
        return self.snapshot(id)
    def review(self,id,start,end,revision):
        if start>end:raise ValueError('검토 범위 시작이 끝보다 늦습니다.')
        with self.connect() as c:
            self._revision(c,id,revision);self._timestamp(c,id,start);self._timestamp(c,id,end)
            c.execute('INSERT OR IGNORE INTO reviews VALUES(?,?,?,?)',(uuid.uuid4().hex,id,start,end));c.execute('UPDATE datasets SET revision=revision+1 WHERE id=?',(id,))
        return self.snapshot(id)
    def delete_review(self,id,rid,revision):
        with self.connect() as c:
            self._revision(c,id,revision);cursor=c.execute('DELETE FROM reviews WHERE dataset=? AND id=?',(id,rid))
            if not cursor.rowcount:raise KeyError('검토 범위를 찾을 수 없습니다.')
            c.execute('UPDATE datasets SET revision=revision+1 WHERE id=?',(id,))
        return self.snapshot(id)
    def import_annotations(self,id,payload,revision):
        if payload.get('schema_version')!=1:raise ValueError('지원하지 않는 라벨 백업 버전입니다.')
        with self.connect() as c:
            self._revision(c,id,revision);meta=self.dataset(id,c)
            if payload.get('dataset',{}).get('fingerprint')!=meta['fingerprint']:raise ValueError('다른 캔들 데이터의 라벨입니다. 데이터 지문이 일치해야 합니다.')
            labels=payload.get('labels',[]);reviews=payload.get('reviews',[])
            if len(labels)>100_000 or len(reviews)>100_000:raise ValueError('라벨 백업이 너무 큽니다.')
            # Validate the ENTIRE payload before mutating. Invalid import is atomic.
            for x in labels:
                self._timestamp(c,id,int(x['timestamp']))
                if x['kind'] not in ('bottom','top') or not 0<=int(x['radius'])<=5 or len(x.get('notes',''))>1000:raise ValueError('잘못된 라벨 백업입니다.')
            for x in reviews:
                self._timestamp(c,id,int(x['start']));self._timestamp(c,id,int(x['end']))
                if x['start']>x['end']:raise ValueError('잘못된 검토 범위입니다.')
            now=time.time()
            for x in labels:c.execute('INSERT INTO labels VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(dataset,timestamp) DO UPDATE SET kind=excluded.kind,radius=excluded.radius,notes=excluded.notes,updated=excluded.updated',(uuid.uuid4().hex,id,int(x['timestamp']),x['kind'],int(x['radius']),x.get('notes',''),now,now))
            for x in reviews:c.execute('INSERT OR IGNORE INTO reviews VALUES(?,?,?,?)',(uuid.uuid4().hex,id,int(x['start']),int(x['end'])))
            c.execute('UPDATE datasets SET revision=revision+1 WHERE id=?',(id,))
        return self.snapshot(id)
    def job(self,id):
        with self.connect() as c:
            r=c.execute('SELECT * FROM jobs WHERE id=?',(id,)).fetchone()
            if not r:raise KeyError('작업을 찾을 수 없습니다.')
            d=dict(r);d['config']=json.loads(d['config']);d['result']=json.loads(d['result']) if d['result'] else None;return d
    def update_job(self,id,status,result=None,error=None):
        with self.connect() as c:c.execute('UPDATE jobs SET status=?,result=?,error=?,updated=? WHERE id=?',(status,json.dumps(result,ensure_ascii=False,allow_nan=False) if result is not None else None,error,time.time(),id))
    def models(self,id):
        lineage=self.ancestors(id);snapshot=self.snapshot(id);semantic=self.annotation_fingerprint(snapshot)
        with self.connect() as c:
            placeholders=','.join('?' for _ in lineage);rows=c.execute(f'SELECT * FROM models WHERE dataset IN ({placeholders}) ORDER BY created DESC',lineage).fetchall()
            results=[]
            for row in rows:
                meta=json.loads(row['metadata']);stale=meta.get('annotation_fingerprint')!=semantic if 'annotation_fingerprint' in meta else row['revision']!=snapshot['dataset']['revision']
                results.append(dict(id=row['id'],revision=row['revision'],stale=stale,created=row['created'],metadata=meta))
            return results
