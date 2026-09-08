"""Hand-rolled RAG for the GIGABYTE AORUS MASTER 16 AM6H spec sheet.

No RAG framework is used anywhere in this package. Chunking, embedding,
indexing, retrieval, prompt assembly and streaming generation are all
implemented here directly on top of numpy and llama.cpp.
"""

__version__ = "0.1.0"
