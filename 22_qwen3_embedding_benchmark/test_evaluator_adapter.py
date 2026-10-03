"""CPU-side pinned-evaluator interface checks; not inference validation."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from run_evaluation import EndpointEncoder, Observer


class AdapterTests(unittest.TestCase):
    def test_pinned_metadata_and_tokenizer_initialization(self):
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(model='unused-in-this-mocked-test', output=Path(folder))
            with patch('transformers.AutoTokenizer.from_pretrained') as tokenizer:
                encoder = EndpointEncoder(args, Observer())
            try:
                self.assertEqual(encoder.mteb_model_meta.embed_dim, 1024)
                self.assertEqual(encoder.mteb_model_meta.max_tokens, 8192)
                self.assertEqual(encoder.mteb_model_meta.similarity_fn_name, 'cosine')
                self.assertIsNone(encoder.mteb_model_meta.public_training_code)
                tokenizer.assert_called_once_with(args.model, local_files_only=True)
            finally:
                encoder.log.close()


if __name__ == '__main__':
    unittest.main()
