"""Explicit HF-model-card Qwen3 reranker inputs; no implicit chat template.

The full formatted input is capped at 8192, preserving prefix and suffix as in
the official Transformers example. This differs from truncating away the
assistant suffix, or silently supplying a generic web-search instruction.
"""
from protocol import TASKS

PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
MAX_LENGTH = 8192


def body(task, query, document, input_order='query_first'):
    fields = [f'<Query>: {query}', f'<Document>: {document}']
    if input_order == 'document_first':
        fields.reverse()
    elif input_order != 'query_first':
        raise ValueError(input_order)
    return '\n'.join([f'<Instruct>: {TASKS[task][2]}'] + fields)


def tokenize_pairs(tokenizer, task, pairs, input_order='query_first'):
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    limit = MAX_LENGTH - len(prefix) - len(suffix)
    encoded = tokenizer([body(task, p['query'], p['document'], input_order) for p in pairs],
                        add_special_tokens=False, padding=False, truncation=False)['input_ids']
    return [prefix + row[:limit] + suffix for row in encoded], sum(len(row) > limit for row in encoded)
