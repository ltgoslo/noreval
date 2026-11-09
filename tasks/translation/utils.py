import os
import sys

# noreval_corpus.py lives at the repository root, one level up
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from noreval_corpus import (  # noqa: F401
    bleu_stderr,
    chrf_stderr,
    mean_bleu,
    mean_chrf,
)
