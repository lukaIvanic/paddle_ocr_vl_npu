import json
from pathlib import Path
import tempfile
import unittest

from run_pixel_cap_ablation import select_pages


class PixelCapSelectionTests(unittest.TestCase):
    def test_selection_uses_recognition_only_strict_threshold_and_dataset_order(self):
        dataset = [{'page_info': {'image_path': name}} for name in ['a.png', 'b.png', 'c.png']]
        rows = [{'request_id': name + ':layout', 'page': name, 'phase': 'layout',
                 'prompt_token_ids': [7] * 1500} for name in ['a.png', 'b.png', 'c.png']]
        for name, count in [('c.png', 1500), ('b.png', 1408), ('a.png', 1500)]:
            rows.append({'request_id': name + ':recognition:0', 'page': name,
                         'phase': 'recognition', 'prompt_token_ids': [7] * count,
                         'block_index': 0, 'block_type': 'table', 'bbox': [0, 0, 1, 1], 'angle': 0})
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'trace.jsonl'
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            subset, indices, crops = select_pages(dataset, path, 5632, 7)
            self.assertEqual(indices, [0, 2])
            self.assertEqual(subset, [dataset[0], dataset[2]])
            self.assertEqual(len(crops), 2)
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows + [rows[0]]))
            with self.assertRaises(ValueError):
                select_pages(dataset, path, 5632, 7)


if __name__ == '__main__':
    unittest.main()
