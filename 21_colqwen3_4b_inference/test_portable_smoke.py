"""Asset rejection tests; no accelerator or PyTorch dependency."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from run_portable_smoke import verify_files
from discover_310p import inventory


class AssetChecks(unittest.TestCase):
    def test_discovery_finds_actual_assets_without_assuming_directory_names(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            model=root/'custom_model_name'
            model.mkdir()
            (model/'config.json').write_text('{"model_type":"ops_colqwen3"}')
            dataset=root/'custom_dataset_name'/'english-corpus'
            dataset.mkdir(parents=True)
            (dataset/'test-00000-of-00001.parquet').touch()
            setup=root/'runtime'
            setup.mkdir()
            (setup/'set_env.sh').touch()
            found=inventory([root])
            self.assertEqual(found['model_candidates'],[str(model)])
            self.assertEqual(found['dataset_candidates'],[str(dataset.parent)])
            self.assertIn(str(setup/'set_env.sh'),found['setup_scripts'])
            self.assertFalse(found['truncated'])
            self.assertTrue(inventory([root],max_dirs=0)['truncated'])

    def test_identical_assets_pass_and_changed_or_missing_assets_fail(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            (root/'config.json').write_bytes(b'original')
            hashes={'config.json':hashlib.sha256(b'original').hexdigest()}
            self.assertEqual(verify_files(root,hashes),hashes)
            (root/'config.json').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                verify_files(root,hashes)
            (root/'config.json').unlink()
            with self.assertRaises(FileNotFoundError):
                verify_files(root,hashes)


if __name__=='__main__':
    unittest.main()
