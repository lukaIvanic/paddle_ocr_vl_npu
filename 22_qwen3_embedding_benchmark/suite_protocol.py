"""Expansion protocol, separate from the completed Chinese experiment.

Sources: Qwen commit 44548aa5 (evaluation/task_prompts.json), MTEB 1.38.9.
Do not substitute present-day benchmark membership or full-corpus BEIR tasks.
"""
from protocol import QWEN_COMMIT, MODEL_REVISION

MTEB_VERSION = '1.38.9'
BENCHMARK = 'MTEB(eng, v2)'
REFERENCES = {'embedding': 61.82, 'reranker': 69.76}
# Pinned MTEB English v2 retrieval subset. All use test/default/nDCG@10.
ENGLISH = {
    'ArguAna': ('mteb/arguana', 'c22ab2a51041ffd869aaddef7af8d8215647e41a',
                'Given a claim, find documents that refute the claim', True),
    'CQADupstackGamingRetrieval': ('mteb/cqadupstack-gaming', '4885aa143210c98657558c04aaf3dc47cfb54340',
                'Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question', True),
    'CQADupstackUnixRetrieval': ('mteb/cqadupstack-unix', '6c6430d3a6d36f8d2a829195bc5dc94d7e063e53',
                'Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question', True),
    'ClimateFEVERHardNegatives': ('mteb/ClimateFEVER_test_top_250_only_w_correct-v2', '3a309e201f3c2c4b13bd4a367a8f37eee2ec1d21',
                'Given a claim about climate change, retrieve documents that support or refute the claim', False),
    'FEVERHardNegatives': ('mteb/FEVER_test_top_250_only_w_correct-v2', '080c9ed6267b65029207906e815d44a9240bafca',
                'Given a claim, retrieve documents that support or refute the claim', False),
    'FiQA2018': ('mteb/fiqa', '27a168819829fe9bcd655c2df245fb19452e8e06',
                'Given a financial question, retrieve user replies that best answer the question', False),
    'HotpotQAHardNegatives': ('mteb/HotpotQA_test_top_250_only_w_correct-v2', '617612fa63afcb60e3b134bed8b7216a99707c37',
                'Given a multi-hop question, retrieve documents that can help answer the question', False),
    'SCIDOCS': ('mteb/scidocs', 'f8c2fcf00f625baaa80f62ec5bd9e1fff3b8ae88',
                'Given a scientific paper title, retrieve paper abstracts that are cited by the given paper', False),
    'TRECCOVID': ('mteb/trec-covid', 'bb9466bac8153a0349341eb1b22e06409e78ef4e',
                'Given a query on COVID-19, retrieve documents that answer the query', False),
    'Touche2020Retrieval.v3': ('mteb/webis-touche2020-v3', '431886eaecc48f067a3975b70d0949ea2862463c',
                'Given a question, retrieve detailed and persuasive arguments that answer the question', False),
}


def format_embedding(text, task, role):
    if role not in ('query', 'passage') or not isinstance(text, str):
        raise ValueError('Unexpected role or non-text input')
    _, _, instruction, symmetric = ENGLISH[task]
    return f'Instruct: {instruction}\nQuery:{text}' if role == 'query' or symmetric else text


def validate_tasks(tasks):
    if {t.metadata.name for t in tasks} != set(ENGLISH) or len(tasks) != len(ENGLISH):
        raise ValueError('English retrieval membership changed')
    for t in tasks:
        meta = t.metadata
        path, revision, _, _ = ENGLISH[meta.name]
        if (meta.type != 'Retrieval' or meta.eval_splits != ['test'] or
                list(t.hf_subsets) != ['default'] or meta.main_score != 'ndcg_at_10' or
                meta.dataset['path'] != path or meta.dataset['revision'] != revision):
            raise ValueError(f'Task contract changed: {meta.name}')


def aggregate(rows, stage):
    by_task = {row['task']: row['ndcg_at_10'] for row in rows}
    if len(rows) != len(by_task) or not set(by_task).issubset(ENGLISH):
        raise ValueError('Duplicate or unexpected task')
    complete = set(by_task) == set(ENGLISH)
    mean = 100 * sum(by_task.values()) / len(by_task) if by_task else None
    return {'benchmark': BENCHMARK, 'stage': stage, 'complete': complete,
            'completed_tasks': list(by_task), 'missing_tasks': sorted(set(ENGLISH) - set(by_task)),
            'macro_ndcg_at_10_percent': mean,
            'published_percent': REFERENCES[stage] if complete else None,
            'delta_pp': mean - REFERENCES[stage] if complete else None,
            'aggregation': 'unweighted task mean; partial means are NOT compared with full published aggregate'}
