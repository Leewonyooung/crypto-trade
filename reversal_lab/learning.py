from __future__ import annotations
import hashlib,json,time,uuid
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report,confusion_matrix,f1_score
from .features import build,LOOKBACK,VERSION

KINDS={0:'neutral',1:'bottom',2:'top'}

def targets(frame,snapshot,horizon):
    """Only explicitly reviewed future windows become supervised samples."""
    n=len(frame);timestamps=frame.timestamp.to_numpy();reviewed=np.zeros(n,bool)
    for r in snapshot['reviews']:
        reviewed|=(timestamps>=r['start'])&(timestamps<=r['end'])
    by_time={int(t):i for i,t in enumerate(timestamps)};truth=[set() for _ in range(n)];event_groups=[set() for _ in range(n)]
    # Nearby/overlapping clicks are one independent event, not extra evidence.
    intervals=[]
    for label in snapshot['labels']:
        if label['timestamp'] not in by_time:raise ValueError('라벨 캔들이 데이터에 없습니다.')
        center=by_time[label['timestamp']];radius=label['radius']
        intervals.append((center-radius-horizon,center+radius,label['id']))
    group_for={};right=-1;group=None
    for left,end,label_id in sorted(intervals):
        if group is None or left>right:group=label_id
        right=max(right,end);group_for[label_id]=group
    for label in snapshot['labels']:
        if label['timestamp'] not in by_time:raise ValueError('라벨 캔들이 데이터에 없습니다.')
        center=by_time[label['timestamp']];radius=label['radius'];cls=1 if label['kind']=='bottom' else 2
        for pivot in range(max(0,center-radius),min(n,center+radius+1)):
            for decision in range(max(0,pivot-horizon),pivot):
                truth[decision].add(cls);event_groups[decision].add(group_for[label['id']])
    y=np.full(n,-1,int);groups=[set() for _ in range(n)]
    for i in range(LOOKBACK-1,n-horizon):
        # A gap in future or feature history invalidates the prediction sample.
        if not reviewed[i+1:i+horizon+1].all():continue
        if frame.timestamp.iloc[i+horizon]-frame.timestamp.iloc[i-LOOKBACK+1]!=(LOOKBACK-1+horizon)*900000:continue
        if len(truth[i])>1:continue # uncertain bottom/top within same horizon: do not overwrite.
        y[i]=next(iter(truth[i])) if truth[i] else 0;groups[i]=event_groups[i]
    return y,groups,reviewed

def split_samples(frame,y,groups,horizon,radius):
    ids=np.flatnonzero(y>=0)
    if len(ids)<300:raise ValueError('검토 완료 구간의 학습 샘플이 부족합니다. 더 넓은 구간을 검토해주세요 (최소300봉).')
    c1=int(ids[int(len(ids)*.6)]);c2=int(ids[int(len(ids)*.8)]);guard=horizon+2*radius
    train=ids[ids<c1-guard];val=ids[(ids>=c1+LOOKBACK)&(ids<c2-guard)];test=ids[ids>=c2+LOOKBACK]
    parts={'train':train,'validation':val,'test':test}
    group_sets={name:set().union(*(groups[i] for i in idx)) for name,idx in parts.items()}
    if any(group_sets[a]&group_sets[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):raise ValueError('동일 라벨 구간이 분할 경계를 넘습니다. 구간을 더 분리해주세요.')
    counts={}
    for name,idx in parts.items():
        counts[name]={KINDS[k]:int((y[idx]==k).sum()) for k in KINDS}
        if set(y[idx])!={0,1,2}:raise ValueError(f'{name} 구간에 저점·고점·중립이 모두 필요합니다. 시간순으로 더 넓게 라벨링/검토해주세요.')
        for cls in [1,2]:
            events=set().union(*(groups[i] for i in idx if y[i]==cls))
            if len(events)<(3 if name=='train' else 1):raise ValueError(f'{name} 구간의 독립 {KINDS[cls]} 라벨이 부족합니다 (학습 각3개, 검증/테스트 각1개 이상).')
    if not train[-1]+LOOKBACK<val[0]+1 or not val[-1]+LOOKBACK<test[0]+1:raise AssertionError('Lookback purge violated')
    return parts,counts

def decide(proba,threshold,classes):
    p=np.zeros((len(proba),3));p[:,np.asarray(classes,int)]=proba
    direction=np.where(p[:,1]>=p[:,2],1,2);confidence=np.maximum(p[:,1],p[:,2]);return np.where(confidence>=threshold,direction,0)

def metrics(y,pred):
    report=classification_report(y,pred,labels=[0,1,2],target_names=['neutral','bottom','top'],output_dict=True,zero_division=0)
    return dict(report=report,confusion_matrix=confusion_matrix(y,pred,labels=[0,1,2]).tolist(),positive_macro_f1=float(f1_score(y,pred,labels=[1,2],average='macro',zero_division=0)),signals=int((pred>0).sum()),samples=len(y))

def train(store,dataset,snapshot,horizon=3):
    if not 1<=horizon<=12:raise ValueError('예측 범위는1~12봉입니다.')
    frame=store.frame(dataset);X=build(frame);y,groups,_=targets(frame,snapshot,horizon);radius=max([x['radius'] for x in snapshot['labels']]+[0]);parts,counts=split_samples(frame,y,groups,horizon,radius)
    tr,va,te=parts['train'],parts['validation'],parts['test']
    learners={
        'logistic':make_pipeline(SimpleImputer(keep_empty_features=True,add_indicator=True),StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=1500,random_state=41)),
        'gradient_boosting':HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=7,min_samples_leaf=15,l2_regularization=10,early_stopping=False,random_state=41),
    }
    comparisons=[];best=None
    for name,model in learners.items():
        model.fit(X.iloc[tr],y[tr]);prob=model.predict_proba(X.iloc[va])
        for threshold in [.35,.45,.55,.65,.75]:
            pred=decide(prob,threshold,model.classes_);score=metrics(y[va],pred);row=dict(model=name,threshold=threshold,validation_f1=score['positive_macro_f1'],signals=score['signals']);comparisons.append(row)
            if best is None or row['validation_f1']>best[0]:best=(row['validation_f1'],name,threshold,model,score)
    _,name,threshold,model,val_metrics=best;test_pred=decide(model.predict_proba(X.iloc[te]),threshold,model.classes_)
    id=uuid.uuid4().hex;label_blob=json.dumps(snapshot,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    folds={name:dict(start_index=int(idx[0]),end_index=int(idx[-1]),start_time=int(frame.timestamp.iloc[idx[0]]),end_time=int(frame.timestamp.iloc[idx[-1]]),samples=len(idx),classes=counts[name]) for name,idx in parts.items()}
    meta=dict(id=id,dataset=dataset,dataset_fingerprint=snapshot['dataset']['fingerprint'],label_revision=snapshot['dataset']['revision'],label_snapshot_sha256=hashlib.sha256(label_blob).hexdigest(),created=time.time(),feature_version=VERSION,features=list(X.columns),lookback=LOOKBACK,horizon=horizon,label_radius_max=radius,model=name,threshold=threshold,folds=folds,validation=val_metrics,test=metrics(y[te],test_pred),comparisons=comparisons,test_prediction_start=int(frame.timestamp.iloc[te[0]]),training_events=len(set().union(*(groups[i] for i in tr))),interpretation='Human-label detection metrics, NOT trading win rate; test labels never used for fit/threshold selection.')
    import platform,sklearn
    meta['annotation_fingerprint']=store.annotation_fingerprint(snapshot)
    meta['libraries']={'python':platform.python_version(),'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__}
    meta['source_sha256']=hashlib.sha256((Path(__file__).read_bytes()+Path(__file__).with_name('features.py').read_bytes())).hexdigest()
    meta['patterns']={}
    for cls,kind in [(1,'bottom'),(2,'top')]:
        profile=[]
        for feature in X.columns:
            values=X.iloc[tr][feature];a=values[y[tr]==cls];b=values[y[tr]==0];sd=values.std()
            if not np.isfinite(sd) or sd==0 or not len(a.dropna()) or not len(b.dropna()):continue
            profile.append(dict(feature=feature,label_median=float(a.median()),neutral_median=float(b.median()),standardized_difference=float((a.mean()-b.mean())/sd)))
        meta['patterns'][kind]=sorted(profile,key=lambda p:abs(p['standardized_difference']),reverse=True)[:8]
    bundle=dict(model=model,metadata=meta)
    tmp=store.model_dir/f'{id}.tmp';joblib.dump(bundle,tmp);tmp.replace(store.model_dir/f'{id}.joblib');(store.model_dir/f'{id}.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,allow_nan=False));(store.model_dir/f'{id}.labels.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2))
    with store.connect() as c:c.execute('INSERT INTO models VALUES(?,?,?,?,?)',(id,dataset,snapshot['dataset']['revision'],json.dumps(meta,ensure_ascii=False,allow_nan=False),time.time()))
    return meta

def predict(store,dataset,model_id,offset=0,limit=1000):
    candidates=store.models(dataset);record=next((x for x in candidates if x['id']==model_id),None)
    if record is None:raise KeyError('이 데이터셋의 모델을 찾을 수 없습니다.')
    # Only internally generated artifacts are loaded. No model upload/deserialization endpoint.
    bundle=joblib.load(store.model_dir/f'{model_id}.joblib');meta=bundle['metadata'];ds=store.dataset(dataset)
    source=store.dataset(meta['dataset'])
    if meta['dataset'] not in store.ancestors(dataset) or source['fingerprint']!=meta['dataset_fingerprint'] or meta['feature_version']!=VERSION:raise ValueError('모델/데이터/특징 버전이 일치하지 않습니다.')
    frame=store.frame(dataset);end=min(len(frame),offset+limit);ids=np.arange(max(LOOKBACK-1,offset),end);ids=ids[frame.timestamp.iloc[ids].to_numpy()>=meta['test_prediction_start']]
    ids=ids[(frame.timestamp.iloc[ids].to_numpy()-frame.timestamp.iloc[ids-LOOKBACK+1].to_numpy())==(LOOKBACK-1)*900000]
    if not len(ids):return dict(predictions=[],stale=record['stale'],eligible_from=meta['test_prediction_start'],message='학습·검증에 사용하지 않은 테스트 시작 시점 이후만 예측을 표시합니다.')
    # Slice a finite100-bar context before each requested batch; no labels/future candles enter inference.
    begin=max(0,int(ids[0])-LOOKBACK+1);X=build(frame.iloc[begin:int(ids[-1])+1].reset_index(drop=True));model=bundle['model'];prob=model.predict_proba(X.iloc[ids-begin]);pred=decide(prob,meta['threshold'],model.classes_);mapped=np.zeros((len(ids),3));mapped[:,model.classes_.astype(int)]=prob
    rows=[dict(index=int(i),timestamp=int(frame.timestamp.iloc[i]),kind=KINDS[int(k)],bottom_score=float(p[1]),top_score=float(p[2]),horizon=meta['horizon']) for i,k,p in zip(ids,pred,mapped)]
    return dict(predictions=rows,stale=record['stale'],eligible_from=meta['test_prediction_start'],message='예측은 판단 시점의 봉에 표시합니다. 변곡 위치로 소급하지 않습니다.')
