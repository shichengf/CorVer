# Third-party notices

`corver/rewards/parsing.py` retains the triplet parser from [QuCo-RAG](https://github.com/ZhishanQ/QuCo-RAG), `src/generate_quco.py`. The extraction prompt follows the same upstream method. The upstream MIT license is reproduced in [licenses/QuCo-RAG-MIT.txt](../licenses/QuCo-RAG-MIT.txt).

The frozen extractor, base models, Wikipedia index, and index tokenizer are external resources referenced by pinned Hugging Face IDs. Their upstream model/data licenses and access conditions apply independently of this repository's MIT code license. See [data notes](../data/README.md) for the selected question pool's sources. Dependencies such as Infini-gram, Unsloth, TRL, and Transformers retain their own licenses.
