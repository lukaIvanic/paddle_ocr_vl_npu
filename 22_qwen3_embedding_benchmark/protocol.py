"""Qwen's published CMTEB-R contract (MTEB 1.38.9), not current defaults."""

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MODEL_REVISION = "b22da495047858cce924d27d76261e96be6febc0"
QWEN_COMMIT = "44548aa5f0a0aed1c76d64e19afe47727a325b8f"
PROMPTS_SHA256 = "c9493103f173d53b1d0a40e42789863af2ce2d5de022a0e075c390229791bfd2"
MAX_LENGTH = 8192
DIMENSIONS = 1024
# name: (dataset revision, qrels revision, instruction, published nDCG@10)
TASKS = {
    "EcomRetrieval": ("687de13dc7294d6fd9be10c6945f9e8fec8166b9", "39c90699b034ec22ac45b3abf5b0bbb5ffd421f9", "Given a user query from an e-commerce website, retrieve description sentences of relevant products", 0.64316),
    "VideoRetrieval": ("58c2597a5943a2ba48f4668c3b90d796283c5639", "faa71382b6a29cf1778d1f436b963e75cb5b927c", "Given a video search query, retrieve the titles of relevant videos", 0.74363),
    "MedicalRetrieval": ("2039188fb5800a9803ba5048df7b76e6fb151fc6", "37b8efec53c54c3d9c6af212f6710b62ccdf895c", "Given a medical question, retrieve user replies that best answer the question", 0.56037),
    "MMarcoRetrieval": ("539bbde593d947e2a124ba72651aafc09eb33fc2", "bae08bb7bddbedb96c7e7db52018a55167b67f89", "Given a web search query, retrieve relevant passages that answer the query", 0.79855),
    "CmedqaRetrieval": ("cd540c506dae1cf9e9a59c3e06f42030d54e7301", "279d737f36c731c8ff6e2b055f31fe02216fa23d", "Given a Chinese community medical question, retrieve replies that best answer the question", 0.41095),
    "DuRetrieval": ("a1a333e290fe30b10f3f56498e3a0d911a693ced", "497b7bd1bbb25cb3757ff34d95a8be50a3de2279", "Given a Chinese search query, retrieve web passages that answer the question", 0.84098),
    "CovidRetrieval": ("1271c7809071a13532e05f25fb53511ffce77117", "a9f41b7cdf24785531d12417ce0d1157ed4b39ca", "Given a question on COVID-19, retrieve news articles that answer the question", 0.84761),
    "T2Retrieval": ("8731a845f1bf500a4f111cf1070785c793d10e64", "1c83b8d1544e529875e3f6930f3a1fcf749a8e97", "Given a Chinese search query, retrieve web passages that answer the question", 0.83677),
}


def format_text(text, task, role):
    if role == "query":
        # Exact official evaluation shell template: no space after Query:.
        return f"Instruct: {TASKS[task][2]}\nQuery:{text}"
    if role == "passage":
        return text
    raise ValueError(f"Unexpected prompt role: {role!r}")


def validate_task(task):
    meta = task.metadata
    expected = TASKS[meta.name]
    assert meta.type == "Retrieval"
    assert meta.eval_splits == ["dev"]
    assert meta.main_score == "ndcg_at_10"
    assert meta.dataset["revision"] == expected[0]
    assert meta.dataset["qrel_revision"] == expected[1]

