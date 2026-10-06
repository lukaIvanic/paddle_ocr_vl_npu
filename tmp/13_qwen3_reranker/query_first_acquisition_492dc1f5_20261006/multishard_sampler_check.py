import gzip, io, json, pathlib, sys, tempfile
from unittest.mock import patch
import pyarrow as pa
import pyarrow.parquet as pq
sys.path.insert(0, '/home/luka/projects/paddle_ocr_vl_npu/13_qwen3_reranker')
import acquire_training_mixture as a
with tempfile.TemporaryDirectory() as directory:
    root=pathlib.Path(directory)
    config='nq_len-0-500'
    names=['train-00000-of-00002.parquet','train-00001-of-00002.parquet']
    for name, lo, hi in zip(names,[0,190],[190,200]):
        pq.write_table(pa.Table.from_pylist([{'query':f'query {i}', 'pos':[f'positive {i}'], 'neg':[f'negative {i}']} for i in range(lo,hi)]),root/name,row_group_size=17)
    metadata={'sha':a.REVISION,'siblings':[{'rfilename':config+'/'+n} for n in names], 'cardData':{'dataset_info':[{'config_name':config,'splits':[{'num_examples':200}]}]}}
    (root/'metadata.json').write_text(json.dumps(metadata))
    (root/'blocked.gz').write_bytes(gzip.compress(json.dumps({'query_hashes':[],'document_hashes':[]}).encode()))
    class OpenFile:
        def __init__(self,url): self.url=url
        def open(self): return (root/self.url.split('/')[-1].split('?')[0]).open('rb')
    with patch('fsspec.open',side_effect=lambda url,**kwargs:OpenFile(url)), patch('urllib.request.urlopen',return_value=io.BytesIO(json.dumps(metadata).encode())), patch.object(sys,'argv',['check','--metadata',str(root/'metadata.json'),'--blocked',str(root/'blocked.gz'),'--output',str(root/'out.gz'),'--cache',str(root/'cache'),'--train-queries','8','--val-queries','4','--prefer-parquet','--workers','4']):
        a.main()
    result=json.loads(gzip.decompress((root/'out.gz').read_bytes()))
    for row in result['train']+result['validation']:
        number=int(row['id'].rsplit('/',1)[1])
        assert row['query']==f'query {number}'
        assert row['documents']==[f'positive {number}',f'negative {number}']
    assert len(result['train'])==8 and len(result['validation'])==4
    print('MULTISHARD_INDEX_AND_SPLIT_CHECK_PASSED')
