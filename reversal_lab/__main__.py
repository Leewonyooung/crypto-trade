import argparse,os
from .data import parse_csv
from .store import Store

def main():
    ap=argparse.ArgumentParser(description='클릭 라벨 → 시간순 검증 → 변곡 예측')
    ap.add_argument('--data-dir',default=os.environ.get('REVERSAL_DATA_DIR','.labeler-data'))
    commands=ap.add_subparsers(dest='command',required=True)
    imp=commands.add_parser('import');imp.add_argument('csv');imp.add_argument('--name',default='BTCUSDT 15m');imp.add_argument('--symbol',default='BTCUSDT')
    serve=commands.add_parser('serve');serve.add_argument('--host',default='127.0.0.1');serve.add_argument('--port',default=8765,type=int)
    args=ap.parse_args()
    if args.command=='import':
        from pathlib import Path
        ds=Store(args.data_dir).import_frame(parse_csv(Path(args.csv).read_bytes()),args.name,args.symbol);print(f"Imported {ds['rows']} candles: {ds['id']}")
    else:
        import uvicorn
        from .app import create_app
        uvicorn.run(create_app(args.data_dir),host=args.host,port=args.port)
if __name__=='__main__':main()
