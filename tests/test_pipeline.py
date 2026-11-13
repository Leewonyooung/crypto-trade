import tempfile,unittest,json,time
from pathlib import Path
import numpy as np,pandas as pd
from fastapi.testclient import TestClient
from reversal_lab.store import Store,Conflict
from reversal_lab.data import parse_csv
from reversal_lab.features import build,LOOKBACK
from reversal_lab.learning import targets,split_samples,train,predict
from reversal_lab.app import create_app
from .fixtures import candles,annotated

class DataTests(unittest.TestCase):
    def test_csv_normalization_and_validation(self):
        f=candles(150);parsed=parse_csv(f.rename(columns={'timestamp':'open_time'}).to_csv(index=False).encode());self.assertEqual(parsed.timestamp.iloc[0],f.timestamp.iloc[0])
        f.loc[2,'high']=1
        with self.assertRaises(ValueError):parse_csv(f.to_csv(index=False).encode())
    def test_duplicate_and_wrong_timeframe_rejected(self):
        f=candles(150);f.loc[1,'timestamp']=f.loc[0,'timestamp']
        with self.assertRaises(ValueError):parse_csv(f.to_csv(index=False).encode())
        f=candles(150);f.timestamp=np.arange(150)*300000+f.timestamp.iloc[0]
        with self.assertRaises(ValueError):parse_csv(f.to_csv(index=False).encode())
    def test_volume_optional(self):
        f=parse_csv(candles(150).drop(columns='volume').to_csv(index=False).encode());self.assertTrue(f.volume.isna().all())

class FeatureTests(unittest.TestCase):
    def test_prefix_and_future_invariance(self):
        f=candles(800);a=build(f);b=build(f.iloc[:500]);np.testing.assert_allclose(a.iloc[:500],b,atol=1e-11,rtol=1e-11,equal_nan=True)
        mutated=f.copy();mutated.loc[500:,['open','high','low','close']]*=10;mutated.loc[500:,'volume']*=20
        np.testing.assert_allclose(a.iloc[:500],build(mutated).iloc[:500],atol=1e-11,rtol=1e-11,equal_nan=True)
    def test_overlapping_clicks_are_one_event(self):
        f=candles(500);snap={'labels':[dict(id=str(i),timestamp=int(f.timestamp.iloc[i]),kind='bottom',radius=1) for i in [200,201,202]],'reviews':[dict(start=int(f.timestamp.iloc[0]),end=int(f.timestamp.iloc[-1]))]}
        y,g,_=targets(f,snap,3);events=set().union(*(g[i] for i in np.flatnonzero(y==1)));self.assertEqual(len(events),1)
    def test_exact_finite100_context(self):
        f=candles(800);a=build(f);b=build(f.iloc[401:501].reset_index(drop=True))
        np.testing.assert_allclose(a.iloc[500],b.iloc[-1],atol=1e-8,rtol=1e-8,equal_nan=True)
    def test_horizon_and_unreviewed_are_not_negative(self):
        f=candles(500);base={'labels':[dict(id='one',timestamp=int(f.timestamp.iloc[200]),kind='bottom',radius=0)],'reviews':[]}
        y,_,_=targets(f,base,3);self.assertTrue((y==-1).all())
        base['reviews']=[dict(start=int(f.timestamp.iloc[0]),end=int(f.timestamp.iloc[-1]))];y,_,_=targets(f,base,3)
        self.assertEqual(y[197:200].tolist(),[1,1,1]);self.assertEqual(y[200],0)
        base['labels'].append(dict(id='two',timestamp=int(f.timestamp.iloc[202]),kind='top',radius=0));y,_,_=targets(f,base,3);self.assertEqual(y[199],-1)

class StoreTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.store=Store(self.tmp.name);self.f=candles(500);self.ds=self.store.import_frame(self.f);self.id=self.ds['id']
    def tearDown(self):self.tmp.cleanup()
    def test_persistence_idempotent_import_and_revision(self):
        self.assertEqual(self.store.import_frame(self.f)['id'],self.id)
        s=self.store.label(self.id,int(self.f.timestamp.iloc[200]),'bottom',1,'메모',0)
        self.assertEqual(Store(self.tmp.name).snapshot(self.id)['labels'][0]['notes'],'메모')
        with self.assertRaises(Conflict):self.store.label(self.id,int(self.f.timestamp.iloc[201]),'top',1,'',0)
        s=self.store.label(self.id,int(self.f.timestamp.iloc[200]),'top',0,'수정',s['dataset']['revision']);self.assertEqual(len(s['labels']),1);self.assertEqual(s['labels'][0]['kind'],'top')
        s=self.store.delete_label(self.id,s['labels'][0]['id'],s['dataset']['revision']);self.assertEqual(s['labels'],[])
    def test_atomic_import_and_fingerprint(self):
        s=self.store.label(self.id,int(self.f.timestamp.iloc[200]),'bottom',1,'',0);backup=json.loads(json.dumps(s));backup['labels'].append(dict(timestamp=123,kind='top',radius=1,notes=''))
        with self.assertRaises(ValueError):self.store.import_annotations(self.id,backup,1)
        self.assertEqual(self.store.snapshot(self.id)['dataset']['revision'],1);self.assertEqual(len(self.store.snapshot(self.id)['labels']),1)
        backup['dataset']['fingerprint']='wrong'
        with self.assertRaises(ValueError):self.store.import_annotations(self.id,backup,1)
    def test_extension_preserves_prefix_labels_and_rejects_edited_history(self):
        self.store.label(self.id,int(self.f.timestamp.iloc[200]),'bottom',1,'keep',0)
        extra=candles(150);extra.timestamp=self.f.timestamp.iloc[-1]+900000+np.arange(len(extra))*900000
        extended=pd.concat([self.f,extra],ignore_index=True);child=self.store.extend_frame(self.id,extended)
        self.assertEqual(child['parent'],self.id);self.assertEqual(len(self.store.snapshot(child['id'])['labels']),1)
        self.assertEqual(self.store.extend_frame(self.id,extended)['id'],child['id'])
        changed=extended.copy();changed.loc[0,'close']+=.1
        with self.assertRaises(ValueError):self.store.extend_frame(self.id,changed)
    def test_review_delete(self):
        s=self.store.review(self.id,int(self.f.timestamp.iloc[0]),int(self.f.timestamp.iloc[-1]),0);s=self.store.delete_review(self.id,s['reviews'][0]['id'],s['dataset']['revision']);self.assertEqual(s['reviews'],[])

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.store=Store(cls.tmp.name);cls.f,cls.id,cls.snapshot=annotated(cls.store);cls.meta=train(cls.store,cls.id,cls.snapshot,3)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_purged_time_splits_and_independent_events(self):
        y,g,_=targets(self.f,self.snapshot,3);parts,counts=split_samples(self.f,y,g,3,1)
        self.assertGreater(parts['validation'][0]-parts['train'][-1],LOOKBACK-1);self.assertGreater(parts['test'][0]-parts['validation'][-1],LOOKBACK-1)
        events=[set().union(*(g[i] for i in parts[k])) for k in ['train','validation','test']]
        self.assertFalse(events[0]&events[1]);self.assertFalse(events[1]&events[2])
    def test_train_artifact_prediction_and_stale_detection(self):
        self.assertTrue((self.store.model_dir/f"{self.meta['id']}.labels.json").exists());self.assertEqual(self.meta['label_revision'],self.snapshot['dataset']['revision'])
        past=predict(self.store,self.id,self.meta['id'],0,300);self.assertEqual(past['predictions'],[])
        start=self.meta['folds']['test']['start_index'];pr=predict(self.store,self.id,self.meta['id'],start,50);self.assertEqual(len(pr['predictions']),50)
        self.assertTrue(all(x['timestamp']>=self.meta['test_prediction_start'] for x in pr['predictions']))
        self.assertTrue(all(x['horizon']==3 for x in pr['predictions']))
        rev=self.store.dataset(self.id)['revision'];self.store.label(self.id,int(self.f.timestamp.iloc[150]),'bottom',0,'added after train',rev)
        after=predict(self.store,self.id,self.meta['id'],start,50);self.assertTrue(after['stale']);self.assertEqual(pr['predictions'],after['predictions'])
    def test_extended_future_uses_existing_model_without_relabelling(self):
        extra=candles(200);extra.timestamp=self.f.timestamp.iloc[-1]+900000+np.arange(len(extra))*900000
        extended=pd.concat([self.f,extra],ignore_index=True);child=self.store.extend_frame(self.id,extended)
        self.assertIn(self.meta['id'],[x['id'] for x in self.store.models(child['id'])])
        result=predict(self.store,child['id'],self.meta['id'],len(self.f),100)
        self.assertEqual(len(result['predictions']),100);self.assertFalse(result['stale'])
        self.assertFalse(any(r['end']>int(self.f.timestamp.iloc[-1]) for r in self.store.snapshot(child['id'])['reviews']))
    def test_insufficient_labels_fail_honestly(self):
        bad=dict(self.snapshot,labels=self.snapshot['labels'][:1])
        with self.assertRaises(ValueError):train(self.store,self.id,bad,3)

class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.app=create_app(self.tmp.name);self.context=TestClient(self.app);self.client=self.context.__enter__();self.headers={'X-Reversal-CSRF':self.client.get('/api/session').json()['csrf']}
    def tearDown(self):self.context.__exit__(None,None,None);self.tmp.cleanup()
    def test_import_click_export_restore_and_csrf(self):
        content=candles(500).to_csv(index=False).encode();self.assertEqual(self.client.post('/api/datasets/import',content=content).status_code,403)
        r=self.client.post('/api/datasets/import',content=content,headers=self.headers);self.assertEqual(r.status_code,200,r.text);ds=r.json();id=ds['id'];t=ds['start']+200*900000
        r=self.client.post(f'/api/datasets/{id}/labels',json=dict(timestamp=t,kind='bottom',radius=1,notes='<script>not executable</script>',revision=0),headers=self.headers);self.assertEqual(r.status_code,200,r.text)
        conflict=self.client.post(f'/api/datasets/{id}/labels',json=dict(timestamp=t,kind='top',radius=0,notes='',revision=0),headers=self.headers);self.assertEqual(conflict.status_code,409)
        backup=self.client.get(f'/api/datasets/{id}/export').json();r=self.client.post(f'/api/datasets/{id}/annotations/import?revision=1',json=backup,headers=self.headers);self.assertEqual(r.status_code,200,r.text)
        self.assertEqual(self.client.get(f'/api/datasets/{id}/candles?at={t}&limit=100').status_code,200)
        self.assertEqual(self.client.post(f'/api/datasets/{id}/train',json={'horizon':3},headers=self.headers).status_code,400)
        self.assertEqual(self.client.post(f'/api/datasets/{id}/train',json={'horizon':50},headers=self.headers).status_code,422)
    def test_background_training_api(self):
        f,id,snapshot=annotated(self.app.state.store);r=self.client.post(f'/api/datasets/{id}/train',json={'horizon':3},headers=self.headers);self.assertEqual(r.status_code,202,r.text);job=r.json()
        for _ in range(200):
            job=self.client.get('/api/jobs/'+job['id']).json()
            if job['status'] in ('succeeded','failed'):break
            time.sleep(.05)
        self.assertEqual(job['status'],'succeeded',job);mid=job['result']['id'];start=job['result']['folds']['test']['start_index']
        result=self.client.post(f'/api/datasets/{id}/predict',json=dict(model_id=mid,offset=start,limit=50),headers=self.headers)
        self.assertEqual(result.status_code,200,result.text);self.assertEqual(len(result.json()['predictions']),50)
    def test_unknown_and_path_model_ids(self):
        self.assertEqual(self.client.get('/api/datasets/missing/annotations').status_code,404)
        self.assertEqual(self.client.post('/api/datasets/missing/predict',json={'model_id':'../../untrusted'},headers=self.headers).status_code,422)

if __name__=='__main__':unittest.main()
