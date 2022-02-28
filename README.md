# Anatomy of Language Representation: What Representation Contributes to Good Text Classification Performance?

Implementation of the paper submitted to repl4nlp 2022: "Anatomy of Language Representation: What Representation Contributes to Good Text Classification Performance?"

## Paper Abstract


## Prepare Dataset
Please visit the [website](https://ai.stanford.edu/~amaas/data/sentiment/)website to access the dataset

## Code Implementations

1. Establishing layer-wise similarity between different layers at the same language model

```   
cd src
sh cka.sh
```

2. Fine-tuning each langauge model to IMDb dataset

```   
cd src
python finetuning.py
```
