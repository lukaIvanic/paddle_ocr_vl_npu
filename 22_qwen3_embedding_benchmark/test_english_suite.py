"""CPU contract tests; real NPU validation is separate."""
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from suite_protocol import ENGLISH, aggregate, format_embedding, validate_tasks
from run_english_suite import dispatch, embedding_command, tokenize_rerank, save, validate_device_snapshot, reuse_embedding, digest


class EnglishTests(unittest.TestCase):
    def test_device_guard(self):
        snapshot='| 0     910B2 | OK | 111 |\n| No running processes found in NPU 0 |'
        validate_device_snapshot(snapshot,[0])
        for value in (snapshot.replace('OK','Alarm'),snapshot.replace('NPU 0','NPU 7')):
            with self.assertRaises(RuntimeError):
                validate_device_snapshot(value,[0])

    def test_explicit_sharing_keeps_health_and_other_device_guards(self):
        snapshot='| 0     910B2 | OK | 111 |'
        with self.assertRaises(RuntimeError):
            validate_device_snapshot(snapshot,[0])
        validate_device_snapshot(snapshot,[0],allow_occupied=[0])
        with self.assertRaises(RuntimeError):
            validate_device_snapshot(snapshot.replace('OK','Alarm'),[0],allow_occupied=[0])
        with self.assertRaises(RuntimeError):
            validate_device_snapshot(snapshot+'\n| 2     910B2 | OK | 111 |',[0,2],allow_occupied=[0])

    def test_symmetric_prompts(self):
        for task, (_, _, instruction, symmetric) in ENGLISH.items():
            self.assertEqual(format_embedding('text',task,'query'),f'Instruct: {instruction}\nQuery:text')
            self.assertEqual(format_embedding('text',task,'passage'),
                             f'Instruct: {instruction}\nQuery:text' if symmetric else 'text')
        self.assertEqual(sum(row[3] for row in ENGLISH.values()),3)

    def test_validation_and_no_partial_reference(self):
        tasks=[SimpleNamespace(metadata=SimpleNamespace(name=n,type='Retrieval',eval_splits=['test'],
               main_score='ndcg_at_10',dataset={'path':r[0],'revision':r[1]}),hf_subsets=['default'])
               for n,r in ENGLISH.items()]
        validate_tasks(tasks)
        with self.assertRaises(ValueError):
            validate_tasks(tasks[:-1])
        rows=[{'task':n,'ndcg_at_10':.5} for n in ENGLISH]
        self.assertIsNone(aggregate(rows[:-1],'embedding')['published_percent'])
        self.assertEqual(aggregate(rows,'embedding')['macro_ndcg_at_10_percent'],50)
        self.assertTrue(aggregate(rows,'reranker')['complete'])
        self.assertEqual(aggregate(rows,'reranker',65.80)['published_percent'],65.80)
        self.assertAlmostEqual(aggregate(rows,'reranker',65.80)['delta_pp'],-15.80)
        self.assertIsNone(aggregate(rows[:-1],'reranker',65.80)['published_percent'])

    def test_saved_candidates_are_verified_without_old_reranker_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            source, output = Path(d)/'source', Path(d)/'output'
            contract = dict(benchmark='test', mteb='1.38.9', tasks=list(ENGLISH),
                            embedding_revision='fixed', embedding_max_length=8192,
                            dtype='float16', top_k=100)
            save(source/'manifest.json', contract)
            save(source/'embedding/summary.json',
                 {'complete':True, 'completed_tasks':list(ENGLISH)})
            save(source/'embedding/results.json', [])
            save(source/'reranker/summary.json', {'old':True})
            for name in ENGLISH:
                path=source/'embedding'/name/'mteb'/f'{name}_default_predictions.json'
                save(path, {'q':{'d':.5}})
                save(source/'embedding'/name/'result.json', {'candidates_sha256':digest(path)})
            reuse_embedding(source, output, contract)
            self.assertFalse((output/'reranker').exists())
            self.assertTrue((output/'saved_embedding_provenance.json').exists())
            save(path, {'corrupted':True})
            with self.assertRaises(ValueError):
                reuse_embedding(source, output, contract)

    def test_dispatch_preserves_all_work_with_zero_and_tail(self):
        results=[]
        dispatch(range(37),['a','b','c'],lambda n:n,lambda e,n:(e,n*n),results.append)
        self.assertEqual(sorted(v for _,v in results),[n*n for n in range(37)])
        self.assertEqual(len(results),37)
        results=[]
        dispatch([],['a'],lambda n:n,lambda e,n:n,results.append)
        self.assertEqual(results,[])

    def test_embedding_configuration(self):
        cmd=embedding_command(18530)
        self.assertIn('--enforce-eager',cmd)
        self.assertIn('--async-scheduling',cmd)
        self.assertEqual(cmd[cmd.index('--max-num-seqs')+1],'128')
        self.assertEqual(cmd[cmd.index('--max-num-batched-tokens')+1],'32768')

    def test_atomic_json(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'sub'/'result.json'
            save(path,{'complete':False})
            save(path,{'complete':True})
            self.assertIn('true',path.read_text())
            self.assertFalse(path.with_suffix('.json.partial').exists())

    def test_reranker_preserves_suffix_and_string_instruction(self):
        class Tok:
            def encode(self,text,**kwargs):
                return [1,2]
            def __call__(self,texts,**kwargs):
                self.texts=texts
                return {'input_ids':[[3]*10000 for t in texts]}
        tok=Tok()
        ids,truncated=tokenize_rerank(tok,'ArguAna',[{'query':'q','document':'d'}])
        self.assertEqual(len(ids[0]),8192)
        self.assertEqual(ids[0][-2:],[1,2])
        self.assertEqual(truncated,1)
        self.assertEqual(tok.texts,['<Instruct>: Given a claim, find documents that refute the claim\n<Query>: q\n<Document>: d'])

    def test_document_first_preserves_field_meanings_and_original_default(self):
        from training_smoke_data import body
        class Tok:
            def encode(self,text,**kwargs): return [1,2]
            def __call__(self,texts,**kwargs):
                self.texts=texts
                return {'input_ids':[[3,4,5] for _ in texts]}
        tok=Tok();pairs=[{'query':'actual query','document':'actual document'}]
        original,_=tokenize_rerank(tok,'ArguAna',pairs)
        self.assertEqual(tok.texts,[body(ENGLISH['ArguAna'][2],'actual query','actual document','query_first')])
        reverse,_=tokenize_rerank(tok,'ArguAna',pairs,'document_first')
        self.assertEqual(tok.texts,['<Instruct>: Given a claim, find documents that refute the claim\n<Document>: actual document\n<Query>: actual query'])
        self.assertEqual(reverse,original)
        with self.assertRaises(ValueError):tokenize_rerank(tok,'ArguAna',pairs,'contents_swapped')


if __name__=='__main__':
    unittest.main()
