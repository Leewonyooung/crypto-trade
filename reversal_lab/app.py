from __future__ import annotations
import json,os,secrets,time,uuid,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI,Request,Query
from fastapi.responses import FileResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
from typing import Literal
from .data import parse_csv
from .store import Store,Conflict
from . import learning

class Label(BaseModel):
    timestamp:int
    kind:Literal['bottom','top']
    radius:int=Field(default=1,ge=0,le=5)
    notes:str=Field(default='',max_length=1000)
    revision:int=Field(ge=0)
class Review(BaseModel):
    start:int
    end:int
    revision:int=Field(ge=0)
class Training(BaseModel):
    horizon:int=Field(default=3,ge=1,le=12)
class Prediction(BaseModel):
    model_id:str=Field(pattern=r'^[0-9a-f]{32}$')
    offset:int=Field(default=0,ge=0)
    limit:int=Field(default=1000,ge=1,le=3000)


class BackupDataset(BaseModel):
    fingerprint:str=Field(pattern=r'^[0-9a-f]{64}$')
class BackupLabel(BaseModel):
    timestamp:int
    kind:Literal['bottom','top']
    radius:int=Field(ge=0,le=5)
    notes:str=Field(default='',max_length=1000)
class BackupReview(BaseModel):
    start:int
    end:int
class Backup(BaseModel):
    schema_version:Literal[1]
    dataset:BackupDataset
    labels:list[BackupLabel]=Field(max_length=100000)
    reviews:list[BackupReview]=Field(max_length=100000)

def create_app(data_dir=None):
    store=Store(data_dir or os.environ.get('REVERSAL_DATA_DIR','.labeler-data'));csrf=secrets.token_urlsafe(32);pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='reversal-training');queue_lock=threading.Lock();pending=set()
    @asynccontextmanager
    async def lifespan(app):
        with store.connect() as c:c.execute("UPDATE jobs SET status='failed',error='서버가 재시작되어 작업이 중단됐습니다.' WHERE status IN ('queued','running')")
        yield
        pool.shutdown(wait=True,cancel_futures=False)
    app=FastAPI(title='Human Reversal Lab',lifespan=lifespan);app.state.store=store
    @app.middleware('http')
    async def guard(request,call_next):
        if request.method not in ('GET','HEAD','OPTIONS'):
            if not secrets.compare_digest(request.headers.get('x-reversal-csrf',''),csrf):return JSONResponse({'detail':'세션이 만료됐습니다. 페이지를 새로고침하세요.'},status_code=403)
            origin=request.headers.get('origin')
            if origin and origin.rstrip('/')!=str(request.base_url).rstrip('/'):
                # Forwarded deployments must preserve Host and scheme; never enable wildcard CORS.
                return JSONResponse({'detail':'다른 출처의 쓰기 요청은 허용하지 않습니다.'},status_code=403)
        response=await call_next(request);response.headers['X-Content-Type-Options']='nosniff';response.headers['Cache-Control']='no-store';return response
    @app.exception_handler(Conflict)
    async def conflict(_,exc):return JSONResponse({'detail':str(exc)},status_code=409)
    @app.exception_handler(ValueError)
    async def invalid(_,exc):return JSONResponse({'detail':str(exc)},status_code=400)
    @app.exception_handler(KeyError)
    async def missing(_,exc):return JSONResponse({'detail':str(exc).strip("'")},status_code=404)
    web=Path(__file__).parent/'web';app.mount('/assets',StaticFiles(directory=web),name='assets')
    @app.get('/')
    def home():return FileResponse(web/'index.html')
    @app.get('/api/session')
    def session():return {'csrf':csrf,'timezone':'Asia/Seoul','interval_ms':900000}
    @app.get('/api/datasets')
    def datasets():return store.datasets()
    @app.post('/api/datasets/import')
    async def import_csv(request:Request,name:str=Query(default='BTCUSDT 15m',max_length=120),symbol:str=Query(default='BTCUSDT',pattern=r'^[A-Za-z0-9_-]{1,30}$')):
        chunks=[];size=0
        async for chunk in request.stream():
            size+=len(chunk)
            if size>100*1024*1024:raise ValueError('CSV는100 MiB 이하로 올려주세요.')
            chunks.append(chunk)
        return store.import_frame(parse_csv(b''.join(chunks)),name,symbol)
    @app.post('/api/datasets/{dataset}/extend')
    async def extend_csv(dataset:str,request:Request):
        chunks=[];size=0
        async for chunk in request.stream():
            size+=len(chunk)
            if size>100*1024*1024:raise ValueError('CSV는100 MiB 이하로 올려주세요.')
            chunks.append(chunk)
        return store.extend_frame(dataset,parse_csv(b''.join(chunks)))
    @app.get('/api/datasets/{dataset}/candles')
    def candles(dataset:str,offset:int=Query(default=0,ge=0),limit:int=Query(default=300,ge=30,le=3000),at:int|None=None):return store.candles(dataset,offset,limit,at)
    @app.get('/api/datasets/{dataset}/annotations')
    def annotations(dataset:str):return store.snapshot(dataset)
    @app.post('/api/datasets/{dataset}/labels')
    def add_label(dataset:str,label:Label):return store.label(dataset,label.timestamp,label.kind,label.radius,label.notes,label.revision)
    @app.delete('/api/datasets/{dataset}/labels/{label_id}')
    def delete_label(dataset:str,label_id:str,revision:int=Query(ge=0)):return store.delete_label(dataset,label_id,revision)
    @app.post('/api/datasets/{dataset}/reviews')
    def review(dataset:str,review:Review):return store.review(dataset,review.start,review.end,review.revision)
    @app.delete('/api/datasets/{dataset}/reviews/{review_id}')
    def delete_review(dataset:str,review_id:str,revision:int=Query(ge=0)):return store.delete_review(dataset,review_id,revision)
    @app.get('/api/datasets/{dataset}/export')
    def export(dataset:str):return JSONResponse(store.snapshot(dataset),headers={'Content-Disposition':'attachment; filename="reversal-labels.json"'})
    @app.post('/api/datasets/{dataset}/annotations/import')
    def import_annotations(dataset:str,payload:Backup,revision:int=Query(ge=0)):
        return store.import_annotations(dataset,payload.model_dump(),revision)
    @app.get('/api/datasets/{dataset}/models')
    def models(dataset:str):return store.models(dataset)
    def run_job(job_id,dataset,snapshot,horizon):
        try:
            store.update_job(job_id,'running')
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=2):
                result=learning.train(store,dataset,snapshot,horizon)
            store.update_job(job_id,'succeeded',result=result)
        except ValueError as exc:store.update_job(job_id,'failed',error=str(exc))
        except Exception:
            import logging
            logging.getLogger(__name__).exception('Training job failed: %s',job_id);store.update_job(job_id,'failed',error='학습 중 오류가 발생했습니다. 서버 로그를 확인하세요.')
        finally:
            with queue_lock:pending.discard(job_id)
    @app.post('/api/datasets/{dataset}/train',status_code=202)
    def start_training(dataset:str,config:Training):
        snapshot=store.snapshot(dataset)
        if not snapshot['labels'] or not snapshot['reviews']:raise ValueError('저점·고점을 라벨링하고 검토 완료 구간을 먼저 저장해주세요.')
        with queue_lock:
            if len(pending)>=4:raise ValueError('학습 대기열이 가득 찼습니다. 기존 작업이 끝난 뒤 다시 시도하세요.')
            job_id=uuid.uuid4().hex;pending.add(job_id)
        with store.connect() as c:c.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?)',(job_id,dataset,snapshot['dataset']['revision'],'queued',json.dumps(config.model_dump()),None,None,time.time(),time.time()))
        pool.submit(run_job,job_id,dataset,snapshot,config.horizon);return store.job(job_id)
    @app.get('/api/jobs/{job_id}')
    def job(job_id:str):return store.job(job_id)
    @app.post('/api/datasets/{dataset}/predict')
    def predict(dataset:str,config:Prediction):return learning.predict(store,dataset,config.model_id,config.offset,config.limit)
    return app
