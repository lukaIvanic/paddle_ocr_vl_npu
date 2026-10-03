import unittest
from decision2_ascend.protocol import validate_metadata


class ProtocolTests(unittest.TestCase):
    def test_valid(self):
        validate_metadata([1,2,3,4,5], {"candidate_positions": [1,3], "query_position": 4, "token_count": 5})

    def test_invalid(self):
        base = {"candidate_positions": [1,3], "query_position": 4, "token_count": 5}
        for change in ({"candidate_positions": [3,1]}, {"candidate_positions": [1,True]},
                       {"candidate_positions": [1,7]}, {"candidate_positions": [1,1]},
                       {"query_position": 3}, {"token_count": 4}, {"extra": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_metadata([1,2,3,4,5], dict(base, **change))
        with self.assertRaises(ValueError):
            validate_metadata([1,2,3], None)


if __name__ == "__main__":
    unittest.main()
