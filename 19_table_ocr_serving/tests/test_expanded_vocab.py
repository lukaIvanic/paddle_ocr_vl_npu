"""Frozen vocabulary integrity; no inference or reference retokenization."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]
PRESETS=ROOT/'19_table_ocr_serving/presets/table_compact_vocab'
SHA='c730b5388f9871ead92e2cb484f8df81ba69518f1e8baeccc5a37f44c1514637'

class ExpandedVocabulary(unittest.TestCase):
    def test_frozen_mapping_and_original_order(self):
        old=json.loads((PRESETS/'b1_verifier_topfreq_16384.json').read_text())
        new=json.loads((PRESETS/'native_han_core_60416.json').read_text())
        ids=new['token_ids']
        self.assertEqual(len(ids),60416)
        self.assertEqual(len(set(ids)),60416)
        self.assertEqual(ids[:16384],old['token_ids'])
        self.assertTrue(all(0<=i<103424 for i in ids))
        digest=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
        self.assertEqual(digest,SHA)
        self.assertEqual(new['token_ids_sha256'],SHA)
        self.assertEqual(new['protected_vocab_size'],60352)
        self.assertEqual(ids[-64:],new['filler_token_ids'])

    def test_all_audited_native_ids_are_preserved(self):
        source=ROOT/'tmp/19_table_ocr_serving/vocab_review_51200_20260911/native_generation_counts.json'
        selected=set(json.loads((PRESETS/'native_han_core_60416.json').read_text())['token_ids'])
        for run in json.loads(source.read_text()):
            self.assertTrue(set(run['all_token_ids'])<=selected,run['source'])

if __name__=='__main__':
    unittest.main()
