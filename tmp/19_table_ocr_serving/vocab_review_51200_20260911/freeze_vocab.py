"""Freeze the reviewed native-ID/Unicode union; never retokenize reference text."""
import hashlib
import json
from pathlib import Path
import runpy

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
scope=runpy.run_path(str(HERE/'review_native_core.py'))
vocab=scope['vocab']
protected=scope['extended']
old=json.loads((ROOT/'19_table_ocr_serving/presets/table_compact_vocab/b1_verifier_topfreq_16384.json').read_text())
assert len(protected)==60352
tokenizer_sha=hashlib.sha256(Path('/tmp/paddle_tokenizer_vocab_audit.json').read_bytes()).hexdigest()
assert tokenizer_sha=='c8a215a59183d0d0781adc33bacd3ce6162716f7fd568fb30234a74d69803a7d'
# Preserve the old head's relative row order, including tie behavior among its
# existing IDs. New protected rows and the 64 remaining fillers are deterministic.
selected=list(old['token_ids'])
selected+=sorted(protected-set(selected))
fill=sorted(set(vocab)-protected)[:60416-len(selected)]
selected+=fill
assert len(selected)==len(set(selected))==60416
assert protected<=set(selected)
assert min(selected)>=0 and max(selected)<103424
sources=json.loads((HERE/'native_generation_counts.json').read_text())
for source in sources:
    assert set(source['all_token_ids'])<=set(selected)
digest=hashlib.sha256(json.dumps(selected,separators=(',',':')).encode()).hexdigest()
data=dict(format='paddleocr_vl_decode_vocab_v1',full_vocab_size=103424,
    selected_vocab_size=len(selected),token_ids_sha256=digest,token_ids=selected,
    selection='Existing 16384 row order; remaining reviewed native/Han/core IDs ascending; 64 remaining mapped IDs ascending.',
    protected_vocab_size=len(protected),filler_token_ids=fill,tokenizer_sha256=tokenizer_sha,
    source_native_traces=[dict(path=s['source'],sha256=s['source_sha256']) for s in sources],
    protections=['saved raw generation IDs','all original 16384 IDs','all Han-containing tokens',
        'all single-character/grapheme tokens','all whitespace-only tokens',
        'all Han Script_Extensions-containing tokens','all combining-mark-containing tokens',
        'all Unicode format-control-containing tokens','all tokenizer-declared special tokens'],
    original_vocab_token_ids_sha256=old['token_ids_sha256'])
destination=ROOT/'19_table_ocr_serving/presets/table_compact_vocab/native_han_core_60416.json'
destination.write_text(json.dumps(data,separators=(',',':'),ensure_ascii=False)+'\n')
print(json.dumps(dict(path=str(destination),size=len(selected),sha256=digest,
    fillers=[dict(id=i,token=vocab[i]) for i in fill]),ensure_ascii=False))
