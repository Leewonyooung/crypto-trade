"""Real browser smoke test with isolated SYNTHETIC data; no live labels are modified.
Run: python -m tests.browser_e2e (Playwright + Chromium required).
"""
import json,os,socket,subprocess,sys,tempfile,time,urllib.request,shutil
import numpy as np,pandas as pd
from pathlib import Path
from playwright.sync_api import sync_playwright
from reversal_lab.store import Store
from .fixtures import annotated,candles

ROOT=Path(__file__).resolve().parents[1]
def main():
    artifacts=ROOT/'.test-artifacts';artifacts.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        store=Store(tmp);frame,id,snapshot=annotated(store)
        sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close();url=f'http://127.0.0.1:{port}'
        env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');server=subprocess.Popen([sys.executable,'-m','reversal_lab','--data-dir',tmp,'serve','--port',str(port)],cwd=ROOT,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        def get(path):
            with urllib.request.urlopen(url+path,timeout=5) as r:return json.load(r)
        try:
            for _ in range(100):
                try:get('/api/session');break
                except Exception:time.sleep(.1)
            else:raise AssertionError('Test server did not start')
            with sync_playwright() as p:
                chromium=os.environ.get('CHROMIUM_PATH') or shutil.which('chromium') or p.chromium.executable_path
                browser=p.chromium.launch(executable_path=chromium,headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
                page=browser.new_page(viewport={'width':1440,'height':1150});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
                page.goto(url,wait_until='networkidle');page.locator('#viewport').wait_for();page.wait_for_function("document.querySelector('#viewport').textContent.includes('~')")
                window=get(f'/api/datasets/{id}/candles?offset=5700&limit=300');idx=100;stamp=window['candles'][idx]['timestamp']
                box=page.locator('#price').bounding_box();x=54+(idx+.5)/300*(box['width']-132);y=180
                page.get_by_role('button',name='저점 / 롱',exact=True).click();page.locator('#notes').fill('browser label <script>not executable</script>');page.locator('#price').click(position={'x':x,'y':y})
                page.wait_for_function("document.querySelector('#status').textContent.includes('저점 라벨을 저장')")
                labels=get(f'/api/datasets/{id}/annotations')['labels'];created=next(z for z in labels if z['timestamp']==stamp);assert created['kind']=='bottom'
                page.reload(wait_until='networkidle');page.wait_for_function("document.querySelector('#labels').textContent.includes('browser label')")
                page.get_by_role('button',name='고점 / 숏',exact=True).click();page.locator('#price').click(position={'x':x,'y':y})
                page.wait_for_function("document.querySelector('#status').textContent.includes('고점 라벨을 저장')");assert next(z for z in get(f'/api/datasets/{id}/annotations')['labels'] if z['timestamp']==stamp)['kind']=='top'
                page.get_by_role('button',name='직전 라벨 취소',exact=True).click();page.wait_for_function("document.querySelector('#status').textContent.includes('취소')");assert next(z for z in get(f'/api/datasets/{id}/annotations')['labels'] if z['timestamp']==stamp)['kind']=='bottom'
                page.get_by_role('button',name='라벨 지우기',exact=True).click();page.locator('#price').click(position={'x':x,'y':y});page.wait_for_function("document.querySelector('#status').textContent.includes('삭제')");assert not any(z['timestamp']==stamp for z in get(f'/api/datasets/{id}/annotations')['labels'])
                # Rapid successive clicks must be queued, never silently discarded.
                page.get_by_role('button',name='저점 / 롱',exact=True).click();page.locator('#notes').fill('rapid queue fixture')
                rapid_indices=[130,132,134]
                for ri in rapid_indices:
                    rx=54+(ri+.5)/300*(box['width']-132);page.locator('#price').click(position={'x':rx,'y':y})
                page.wait_for_function("Array.from(document.querySelectorAll('#labels .note')).filter(x=>x.textContent.includes('rapid queue fixture')).length===3")
                labels=get(f'/api/datasets/{id}/annotations')['labels']
                for ri in rapid_indices:assert any(z['timestamp']==window['candles'][ri]['timestamp'] for z in labels)
                page.get_by_role('button',name='현재 보이는 구간 검토 완료',exact=True).click();page.wait_for_function("document.querySelector('#status').textContent.includes('검토 완료로 저장')")
                with page.expect_download() as download:page.get_by_role('button',name='라벨 백업',exact=True).click()
                target=artifacts/'browser-label-backup.json';download.value.save_as(target);assert json.loads(target.read_text())['schema_version']==1
                page.get_by_role('button',name='새 모델 학습',exact=True).click();page.wait_for_function("document.querySelector('#job').textContent.includes('학습과 미사용 테스트 구간 평가가 완료')",timeout=60000)
                page.wait_for_function("document.querySelector('#model').options.length>0");page.get_by_role('button',name='테스트 구간 보기',exact=True).click();page.wait_for_timeout(200)
                page.get_by_role('button',name='현재 구간 예측',exact=True).click();page.wait_for_function("document.querySelector('#status').textContent.includes('판단 시점')",timeout=15000)
                # Append only unchanged history; existing labels/model remain available on unseen tail.
                extra=candles(600);extra.timestamp=frame.timestamp.iloc[-1]+900000+np.arange(len(extra))*900000
                extended=pd.concat([frame,extra],ignore_index=True);extension=Path(tmp)/'extended.csv';extended.to_csv(extension,index=False)
                page.locator('#extendCSV').set_input_files(str(extension))
                page.wait_for_function("document.querySelector('#status').textContent.includes('과거 이력이 같은 새 데이터')",timeout=20000)
                children=[d for d in get('/api/datasets') if d.get('parent')==id];assert len(children)==1
                page.wait_for_function("document.querySelector('#model').options.length>0")
                page.get_by_role('button',name='현재 구간 예측',exact=True).click();page.wait_for_function("document.querySelector('#status').textContent.includes('판단 시점')")
                assert not errors,errors
                page.screenshot(path=str(artifacts/'labeler-desktop.png'),full_page=True)
                # Mobile layout, no clipping/JS errors.
                page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(150);assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth+2');page.screenshot(path=str(artifacts/'labeler-mobile.png'),full_page=True)
                assert not errors,errors;browser.close()
            print('PASS Chromium: click/save/edit/undo/delete/reload/rapid-click queue, review, backup, train, held-out prediction, exact-prefix extension/model reuse, responsive layout; synthetic fixture only')
        finally:
            server.terminate()
            try:server.wait(timeout=10)
            except subprocess.TimeoutExpired:server.kill()
if __name__=='__main__':main()
