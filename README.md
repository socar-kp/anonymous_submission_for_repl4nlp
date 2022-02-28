# Anatomy of Language Representation: What Representation Contributes to Good Text Classification Performance?

Implementation of the paper submitted to repl4nlp 2022: "Anatomy of Language Representation: What Representation Contributes to Good Text Classification Performance?"

## Paper Abstract
With the development of effective language models, utilizing this neural networks architecture and the pre-trained weights has become a de facto method in various natural language processing tasks. However, while numerous studies focused on elevating target task performance based on these language models, there has been less spotlight on the contained knowledge of various pre-trained models and how the fine-tuning convey the representation power for the target task. In this study, we propose a proof-of-concept level study on the dynamics of various language models during fine-tuning. Upon a simple text classification task, our study discovered the following takeaways. First, the fine-tuning lets the model learn high-level representations, which implicit contextual understanding of the text data. Second, we discovered good fine-tuned models' high-level representations are particularly changed from the pre-trained ones. Third, we also analyzed good fine-tuned models' high-level representations are discretized from the low and middle-level representations. Fourth, we scrutinized that various pre-trained language models have representations similar to each other in a particular manner. Lastly, we examined that fine-tuned models do not share similar high-level representations although they accomplish similar target performance; thus, we hypothesized that each language model has its own way of understanding text data. Based on our study's takeaways and improvement avenues, we expect the dynamics of language models and fine-tuning mechanisms will be revealed shortly.


## Prepare Dataset

1. Please visit the [website](https://ai.stanford.edu/~amaas/data/sentiment/) to access the dataset

2. Make the dataset directory with the following procedures, and move the downloaded dataset into the dataset directory.

```   
mkdir dataset
cd dataset
```

## Codes

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
