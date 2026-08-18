from pathlib import Path


def test_gitignore_blocks_generated_data_and_writing_outputs():
    text = Path(".gitignore").read_text(encoding="utf-8")
    required_patterns = [
        "data/",
        "outputs/",
        "*.parquet",
        "*.joblib",
        "manuscript/online_sync/",
        "example/",
        "examples/generated_writing/",
        "external/",
        "models/",
        "cache/",
        "*.zip",
    ]
    for pattern in required_patterns:
        assert pattern in text
