"""Asset rejection tests; no accelerator or PyTorch dependency."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from run_portable_smoke import verify_files


class AssetChecks(unittest.TestCase):
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
