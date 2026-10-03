"""Small pure-Python HTTP boundary, independently testable without an NPU."""


def validate_metadata(tokens, metadata):
    if not isinstance(tokens, list) or not tokens or any(type(x) is not int or x < 0 for x in tokens):
        raise ValueError("Decision2 expects exactly one pretokenized question per HTTP request")
    if not isinstance(metadata, dict) or set(metadata) != {"candidate_positions", "query_position", "token_count"}:
        raise ValueError("Invalid Decision2 metadata keys")
    positions = metadata["candidate_positions"]
    if not isinstance(positions, list) or not 2 <= len(positions) <= 255:
        raise ValueError("Expected 2..255 candidate positions")
    if any(type(p) is not int or not 0 <= p < len(tokens) - 1 for p in positions):
        raise ValueError("Candidate position out of range")
    if sorted(set(positions)) != positions:
        raise ValueError("Candidate positions must be strictly increasing")
    if type(metadata["query_position"]) is not int or metadata["query_position"] != len(tokens) - 1:
        raise ValueError("Decision readout must be the final token")
    if type(metadata["token_count"]) is not int or metadata["token_count"] != len(tokens):
        raise ValueError("Token count mismatch")
