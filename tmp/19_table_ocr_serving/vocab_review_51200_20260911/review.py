"""Offline vocabulary audit only; never changes a serving preset."""
from __future__ import annotations
import argparse
import collections
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
sys.path.insert(0, '/tmp/paddle-vocab-audit-deps')
import regex
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', type=Path, default=Path('/tmp/paddle_tokenizer_vocab_audit.json'))
    parser.add_argument('--annotations', type=Path, default=Path('/tmp/paddle_vocab_audit_omnidocbench.json'))
    args = parser.parse_args()
    raw = json.loads(args.tokenizer.read_text())
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    selection = json.loads((ROOT/'19_table_ocr_serving/presets/table_compact_vocab/b1_verifier_topfreq_16384.json').read_text())
    kept = set(selection['token_ids'])
    vocab = {i:t for t,i in raw['model']['vocab'].items()}
    special = set()
    for t in raw['added_tokens']:
        vocab[t['id']] = t['content']
        if t['special']:
            special.add(t['id'])
    han_pattern = regex.compile(r'\p{Script=Han}')
    grapheme_pattern = regex.compile(r'\X')
    han = set()
    core = set()
    for i,t in vocab.items():
        if i in special:
            continue
        if han_pattern.search(t):
            han.add(i)
        literal = t.replace('▁',' ')
        normalized = literal.lstrip(' ')
        if len(normalized)==1 or len(literal)==1 or (normalized and grapheme_pattern.fullmatch(normalized)):
            core.add(i)
    protected = kept | han | core
    fill = [i for i in sorted(vocab) if i not in protected and i not in special][:51200-len(protected)]
    candidate = protected | set(fill)
    assert len(protected)==39556 and len(candidate)==51200
    all_missing = collections.Counter()
    reference_missing = collections.Counter()
    reference_ids = set()
    command_missing = collections.Counter()

    def examples(counts, limit=35):
        return [dict(id=i,token=vocab.get(i),count=n) for i,n in counts.most_common(limit)]

    def token_coverage(sequences):
        counts=collections.Counter();affected=0;records=0
        for ids in sequences:
            records+=1
            absent=[i for i in ids if i not in candidate]
            affected+=bool(absent);counts.update(absent)
        all_missing.update(counts)
        return dict(records=records,affected_records=affected,missing_unique=len(counts),
                    missing_occurrences=sum(counts.values()),missing=examples(counts,100))

    paths = [
        ROOT/'tmp/09_persistent_page_engine/table_vllm_poisson1000_seq64_14qps_npu4_retry_85a90862_20260908/budget16384/qps3/measured/results.jsonl',
        ROOT/'tmp/19_table_ocr_serving/lm_head_ab_20260911/b2_full_measured/b2/measured/results/results.jsonl',
        ROOT/'tmp/19_table_ocr_serving/lm_head_ab_20260911/b8_full_measured/b8/measured/results/results.jsonl',
    ]
    native = {}
    for path in paths:
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        unique={r['request_id']:r for r in rows}
        result=token_coverage(r['service_result']['response']['token_ids'] for r in unique.values())
        result['unique_tables']=len(unique)
        result['source_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        result['contexts'] = []
        for rid, row in unique.items():
            ids = row['service_result']['response']['token_ids']
            absent = [n for n,i in enumerate(ids) if i not in candidate]
            if absent:
                result['contexts'].append(dict(request_id=rid, snippets=[
                    dict(id=ids[n], token=vocab[ids[n]], text=tokenizer.decode(ids[max(0,n-8):n+9]))
                    for n in absent]))
        native[str(path.relative_to(ROOT))]=result

    annotations=json.loads(args.annotations.read_text())
    grouped=collections.defaultdict(list)
    commands=collections.Counter()
    class CellParser(HTMLParser):
        def __init__(self):
            super().__init__(); self.inside=False; self.parts=[]; self.cells=[]
        def handle_starttag(self, tag, attrs):
            if tag in ('td','th'):
                self.inside=True; self.parts=[]
        def handle_endtag(self, tag):
            if tag in ('td','th') and self.inside:
                self.cells.append(''.join(self.parts)); self.inside=False
        def handle_data(self, data):
            if self.inside:
                self.parts.append(data)
    for page in annotations:
        for item in page.get('layout_dets',[]):
            kind=item['category_type']
            # Canonical reference-string tokenization is NOT a saved generation.
            for field in ('text','latex','html'):
                value=item.get(field)
                if isinstance(value,str) and value:
                    grouped[kind+':'+field].append(value)
                    commands.update(re.findall(r'\\[A-Za-z]+',value))
                    if kind=='table' and field=='html':
                        cells=CellParser(); cells.feed(value)
                        grouped['table_cell_text:derived'].extend(c for c in cells.cells if c)
    reference={}
    reference_group_ids={}
    for kind,strings in sorted(grouped.items()):
        enc=tokenizer.encode_batch(strings,add_special_tokens=False)
        reference_ids.update(i for e in enc for i in e.ids)
        reference_group_ids[kind]={i for e in enc for i in e.ids}
        counts=collections.Counter(i for e in enc for i in e.ids if i not in candidate)
        reference_missing.update(counts)
        reference[kind]=dict(strings=len(strings),affected_strings=sum(any(i not in candidate for i in e.ids) for e in enc),
                            total_tokens=sum(len(e.ids) for e in enc),missing_occurrences=sum(counts.values()),
                            missing_unique=len(counts),missing=examples(counts))
    missing_commands=[]
    for command,n in commands.most_common():
        ids=tokenizer.encode(command,add_special_tokens=False).ids
        missing=[i for i in ids if i not in candidate]
        if missing:
            command_missing.update(missing)
            missing_commands.append(dict(command=command,occurrences=n,tokens=[vocab[i] for i in ids],missing=[vocab[i] for i in missing]))

    probes=[
        'MIN TEMP MAX AVG STD MEAN SD SE CI P95 P99',
        'Publisher Douglas brain temperature concentration coefficient uncertainty',
        'µg/mL μmol/L mmol/L mg/kg kg·m⁻³ m² s⁻¹ °C °F ± ≤ ≥ ≠ ≈ × ÷ ¾ ½ ‰',
        r'\frac{a}{b} \sqrt{x} \alpha \beta \gamma \Delta \sigma \epsilon \varepsilon',
        r'\sum_{i=1}^{n} \prod \int \infty \partial \nabla \forall \exists \notin',
        r'\operatorname{diag} \mathbf{x} \boldsymbol{\theta} \mathbb{R} \mathcal{L}',
        r'\begin{aligned} x &= y \\ z &= 1 \end{aligned}',
        'H₂O CO₂ Na⁺ Ca²⁺ Fe³⁺ SO₄²⁻ 10⁻⁶ 10⁹',
        'p<0.05 p≤0.01 95% CI 1.2e−3 1.2E-3 $ € £ ¥ ₹ ₽ ₩',
        'Ångström Müller François İstanbul Straße naïve café résumé',
        'Температура среднее стандартное отклонение',
        'درجة الحرارة متوسط القيمة',
        '平均 温度 標準偏差 カタカナ ひらがな',
        '평균 온도 표준편차',
        'तापमान औसत मान ค่าเฉลี่ย อุณหภูมิ',
        '<fcel>A<ecel><lcel><ucel><xcel><nl>',
    ]
    probe_results=[]
    for text in probes:
        ids=tokenizer.encode(text,add_special_tokens=False).ids
        probe_results.append(dict(text=text,missing=[dict(id=i,token=vocab[i]) for i in ids if i not in candidate]))
    omitted=[(i,t) for i,t in vocab.items() if i not in candidate and i not in special]
    result=dict(tokenizer_sha256=hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        annotations_sha256=hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
        selection_token_ids_sha256=selection['token_ids_sha256'],active_preset_changed=False,
        counts=dict(existing=len(kept),all_han_union=len(kept|han),protected=len(protected),fill=len(fill),
                    candidate=len(candidate),highest_fill_id=max(fill)),
        native_outputs=native,reference_strings=reference,reference_missing_overall=examples(reference_missing,100),
        formula_commands=dict(unique=len(commands),missing_commands=missing_commands),probes=probe_results,
        omitted_early_examples=omitted[:100],
        observed_native_missing=examples(all_missing,100),
        repair_counts=dict(add_all_observed_native=len(set(all_missing)-protected),
            protected_plus_all_reference_ids=len(protected|reference_ids),
            protected_plus_table_cell_formula_native=len(protected|reference_group_ids['table_cell_text:derived']|reference_group_ids['equation_isolated:latex']|set(all_missing)),
            missing_table_cell_formula_native=len((reference_group_ids['table_cell_text:derived']|reference_group_ids['equation_isolated:latex']|set(all_missing))-candidate),
            protect_reference_tokenizations_total=len(protected|set(reference_missing)),
            protect_reference_and_native_total=len(protected|set(reference_missing)|set(all_missing)),
            extra_missing_latex_command_pieces=len(set(command_missing)-protected)))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
