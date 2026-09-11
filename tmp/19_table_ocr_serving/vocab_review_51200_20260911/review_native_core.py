"""Inventory check only: raw generated IDs plus Unicode protections; no encoding."""
import collections
import json
from pathlib import Path
import sys
import unicodedata
sys.path.insert(0, '/tmp/paddle-vocab-audit-deps')
import regex

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
raw=json.loads(Path('/tmp/paddle_tokenizer_vocab_audit.json').read_text())
vocab={i:t for t,i in raw['model']['vocab'].items()}
special=set()
for t in raw['added_tokens']:
    vocab[t['id']]=t['content']
    if t['special']: special.add(t['id'])
kept=set(json.loads((ROOT/'09_persistent_page_engine/presets/table_compact_vocab/b1_verifier_topfreq_16384.json').read_text())['token_ids'])
native=set().union(*(set(s['all_token_ids']) for s in json.loads((HERE/'native_generation_counts.json').read_text())))
han={i for i,t in vocab.items() if i not in special and regex.search(r'\p{Script=Han}',t)}
core={i for i,t in vocab.items() if i not in special and
      (len(t.replace('▁',' '))==1 or regex.fullmatch(r'\X',t.replace('▁',' ').lstrip(' ')))}
base=kept|native|han|core
assert len(base)==59537
omitted={i:t for i,t in vocab.items() if i not in base}
def entries(ids):
    return [dict(id=i,token=vocab[i]) for i in sorted(ids)]
groups={
    'han':han,
    'single_grapheme':core,
    'single_codepoint':{i for i,t in vocab.items() if len(t.replace('▁',' '))==1},
    'byte_fallback':{i for i,t in vocab.items() if regex.fullmatch(r'<0x[0-9A-Fa-f]{2}>',t)},
    'whitespace_only':{i for i,t in vocab.items() if t.replace('▁',' ').isspace()},
    'special':special,
    'han_script_extensions':{i for i,t in vocab.items() if i not in special and regex.search(r'\p{Script_Extensions=Han}',t)},
    'any_combining_mark':{i for i,t in vocab.items() if any(unicodedata.category(c).startswith('M') for c in t)},
    'variation_selector':{i for i,t in vocab.items() if any(0xFE00<=ord(c)<=0xFE0F or 0xE0100<=ord(c)<=0xE01EF for c in t)},
    'format_control':{i for i,t in vocab.items() if any(unicodedata.category(c)=='Cf' for c in t)},
}
probes={
    'Chinese punctuation':'，。！？；：、（）【】《》〈〉「」『』〔〕［］｛｝“”‘’…—·﹏～￥％＋－＝＜＞／＼｜',
    'Chinese numerals':'〇零一二三四五六七八九十百千万亿萬億壹贰貳叁參肆伍陆陸柒捌玖拾佰仟兩两',
    'Chinese characters':'汉漢语語国國龙龍发發髮后後台臺湾灣简簡体體繁龘麤𠮷𠀀',
    'Latin and Greek':'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzàáâäæçèéêëìíîïñòóôöœùúûüýÿßαβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ',
    'Mathematical notation':'±∓×÷−=≠≈≃≅≡≤≥≪≫∝∞∑∏∫∮√∂∇∈∉⊂⊆⊃⊇∪∩∅∀∃¬∧∨⇒⇔→←↔∠⊥∥°′″ℝℂℕℤℚ',
    'Invisible characters':'\n\r\t\u00a0\u202f\u3000\u200b\u200c\u200d\u2060\ufeff\ufe0f',
}
probe_result={}
for name,chars in probes.items():
    no_direct=[];excluded_direct=[]
    for c in dict.fromkeys(chars):
        ids={i for i,t in vocab.items() if t==c or t=='▁'+c}
        item=dict(character=c,codepoint=f'U+{ord(c):04X}',name=unicodedata.name(c,'UNNAMED'))
        if not ids:no_direct.append(item)
        elif not ids<=base:excluded_direct.append(dict(**item,ids=sorted(ids-base)))
    probe_result[name]=dict(no_direct_token_in_full_vocabulary=no_direct,excluded_direct_tokens=excluded_direct)
extended=set(base)
addition_counts=[]
for name in ('whitespace_only','han_script_extensions','any_combining_mark','format_control','special'):
    extra=groups[name]-extended
    extended.update(extra)
    addition_counts.append(dict(category=name,additional_ids=len(extra),cumulative_ids=len(extended)))
result=dict(base_ids=len(base),rounded_size=((len(base)+1023)//1024)*1024,
    proposed_extension=dict(additions=addition_counts,unique_ids=len(extended),
        unfilled_rows=60416-len(extended),added_ids=entries(extended-base),
        active_preset_changed=False),
    groups={k:dict(total=len(v),missing_count=len(v-base),missing=entries(v-base)) for k,v in groups.items()},
    probes=probe_result,omitted_examples=entries(set(omitted))[:80])
(HERE/'native_core_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(base_ids=len(base),proposed_ids=len(extended),
    additions=addition_counts,rounded_rows=60416,unfilled_rows=60416-len(extended)),indent=2))
