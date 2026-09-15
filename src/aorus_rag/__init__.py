"""Hand-rolled RAG for the GIGABYTE AORUS MASTER 16 AM6H spec sheet.

No RAG framework is used anywhere in this package. Chunking, embedding,
indexing, retrieval, prompt assembly and streaming generation are all
implemented here directly on top of numpy and llama.cpp.
"""

# Module map. The system runs in two phases:
#   build (offline, once -- the outputs ship with the repo)
#     fetch.py      download the product pages into data/raw/
#     parse.py      HTML -> the 17-row spec table
#     normalize.py  spec rows -> atomic one-value facts
#     chunk.py      structure-aware chunks -> corpus.jsonl
#     embed.py      chunks -> vectors
#     index.py      vectors -> index.npz; BM25 and RRF also live here
#   ask (per question)
#     retrieve.py   dense + BM25 retrieval, fused with RRF
#     prompt.py     language detection and prompt assembly
#     llm.py        streaming generation on llama.cpp, with TTFT / TPS timing
# pipeline.py wires these together, cli.py exposes them as commands, bench.py evaluates.
__version__ = "0.1.0"
