"""Download official Mamba source only. No model execution or installation."""
import argparse,hashlib,json,re,traceback,urllib.request
from pathlib import Path

def get(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'T09-reference-audit'}),timeout=30) as r:return r.read()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    try:
        commit=json.loads(get('https://api.github.com/repos/state-spaces/mamba/commits/main'))['sha']
        assert re.fullmatch('[0-9a-f]{40}',commit)
        hashes={};urls={}
        for path in ['mamba_ssm/modules/mamba_simple.py','mamba_ssm/ops/selective_scan_interface.py','LICENSE']:
            url=f'https://raw.githubusercontent.com/state-spaces/mamba/{commit}/{path}'
            data=get(url);(out/Path(path).name).write_bytes(data)
            hashes[path]=hashlib.sha256(data).hexdigest();urls[path]=url
        (out/'reference.json').write_text(json.dumps(dict(commit=commit,hashes=hashes,urls=urls),indent=2),encoding='utf8')
        print('Reference downloaded and pinned:',commit,flush=True)
    except BaseException:
        (out/'download_error.txt').write_text(traceback.format_exc(),encoding='utf8');raise
if __name__=='__main__':main()
