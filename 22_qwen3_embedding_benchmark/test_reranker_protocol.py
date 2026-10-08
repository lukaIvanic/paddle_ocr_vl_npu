import unittest
from reranker_protocol import PREFIX, SUFFIX, MAX_LENGTH, body, tokenize_pairs


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def __call__(self, texts, **kwargs):
        return {'input_ids': [self.encode(t) for t in texts]}


class ProtocolTests(unittest.TestCase):
    def test_document_first_preserves_field_meanings(self):
        ids, count = tokenize_pairs(CharacterTokenizer(), 'T2Retrieval',
            [{'query':'query text', 'document':'document text'}], 'document_first')
        text = ''.join(map(chr,ids[0]))
        self.assertEqual(text, PREFIX + body('T2Retrieval','query text','document text','document_first') + SUFFIX)
        self.assertIn('\n<Document>: document text\n<Query>: query text',text)
        self.assertEqual(count,0)
        with self.assertRaises(ValueError):
            body('T2Retrieval','query','document','unknown')

    def test_fields_and_suffix(self):
        pairs = [{'query': '商品查询', 'document': '商品描述'}]
        ids, count = tokenize_pairs(CharacterTokenizer(), 'EcomRetrieval', pairs)
        self.assertEqual(count, 0)
        text = ''.join(map(chr, ids[0]))
        self.assertEqual(text, PREFIX + body('EcomRetrieval', '商品查询', '商品描述') + SUFFIX)
        self.assertIn('e-commerce', text)

    def test_truncation_preserves_readout(self):
        ids, count = tokenize_pairs(CharacterTokenizer(), 'T2Retrieval', [{'query': 'Q', 'document': 'x'*10000}])
        self.assertEqual(len(ids[0]), MAX_LENGTH)
        self.assertEqual(count, 1)
        self.assertTrue(''.join(map(chr, ids[0])).endswith(SUFFIX))


if __name__ == '__main__':
    unittest.main()
