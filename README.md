Name: Boris Zarubin
Email: b.zarubin@innopolis.university

# SpERT pretrained on a NEREL task
This is a [SpERT](https://github.com/lavis-nlp/spert) model, pretrained on a [RuNNE](https://huggingface.co/datasets/iluvvatar/RuNNE) for this [CodaLab competition](https://codalab.lisn.upsaclay.fr/competitions/18459).

All notebooks are available on kaggle and designed to run on kaggle.
## Kaggle links
### Notebooks
- [preprocess](https://www.kaggle.com/code/k1shin/spert-nerel-preprocess)
- [train](https://www.kaggle.com/code/k1shin/spert-nerel-train)
- [convert-safetensors](https://www.kaggle.com/code/k1shin/spert-nerel-convert-safetensors)
- [predict](https://www.kaggle.com/code/k1shin/spert-nerel-predict)
### Models
- [rubert-large](https://www.kaggle.com/models/k1shin/rubert-large-nerel)
- [rubert-tiny](https://www.kaggle.com/models/k1shin/rubert-tiny-nerel)
### Datasets
- [ner-datasets](https://www.kaggle.com/datasets/k1shin/ner-datasets)

## Introduction
Before applying SpERT model, different other approaches were researched and tested:
- [piqn](https://github.com/tricktreat/piqn) - too expensive in terms of computational resources
- [MRC](https://github.com/ShannonAI/mrc-for-flat-nested-ner) - outdated and abandoned due to compatibility problems
- [Biaffine NER](https://github.com/amir-zeldes/biaffine-ner) - outdated and abandoned due to compatibility problems

Eventually it was decided to use SpERT model. The main challenge here was to transform RuNNE dataset to SPERT format and vice versa, the details are described in the next section.

## Implementation
### Preprocess
Our input is a list of strings, where each string contains several sentences and char offsets with NER labels. Our task is to convert this data to a list of sentences, where each sentence is a list of tokens, where NER labels link these tokens using indices.
Further, `nltk.tokenize.sent_tokenize(text, language='russian')` and `nltk.tokenize.word_tokenize(sentence, language='russian')` will be used for tokenizing.
The main parts of the preprocess notebook are
- **Preprocess functions**
First task to solve is finding char offsets, having tokens list and original text. It is done by iterating through the original text and yielding char offsets if next token is met. If not all tokens are found a `ValueError` is raisen.
*Implemented in `tokens_to_indices(text, tokens)`*
Second task to solve is finding token indices, having char offsets for each token and some input offset. it is done by iterating through original char offsets and finding point of overlaps.
*Implemented in `index_to_tokens(token_indices, index)`*
- **Transformation functions**
Now, that we have some backend functions we could apply them to transform data.
Our first step is to parse the NEREL data. Here sentinfo (sentence information) format is introduced, which serves as a mediator between NEREL and SPERT. It looks as follows:
    ```
    List[{
        'sentence': sentence as a string,
        'sent_offset': sentence offset relative to original NEREL text (NEREL batch),
        'tokens': list of tokens,
        'tokens_offsets': tokens offsets relative to sentence,
        'batch_ix': index of NEREL batch,
        'sent_ix': sentence index in sentinfo data,
        'ners': NER labels in SPERT format
    }]
    ```
    Now, to convert NEREL to sentinfo, we do the following:
    ```
    sentinfo = []
    for nerel_batch in nerel_data:
        for sentence in sent_tokenize(nerel_batch):
            sent_offset = tokens_to_indices(nerel_batch, sentence)
            tokens = word_tokenize(sentence)
            tokens_offsets = tokens_to_indices(sentence, tokens)
            ners = []
            for nerel_ner in nerel_batch['ners']:
                if nerel_ner in sentence:
                    ners.append(index_to_tokens(tokens_offsets, nerel_ner))
            
            # append all this data as sentinfo item
            sentinfo.append(...)
    ```
    *Implemented in `nerel_to_sentinfo(nerel_data, train=True)`*
- **Convert NEREL dataset to SPERT format**
To convert NEREL to SPERT we firstly convert it to sentinfo and then return the following:
    ```
    return [{
        'tokens': sent['tokens'],
        'entities': sent['ners'],
        'relations': []
    } for sent in sentinfo]
    ```
    *Implemented in `sentinfo_to_spert(sentinfo)`*
- **Convert SPERT predictions to NEREL**
Here, `sentinfo['batch_ix']` and `sentinfo['sent_offset']` are used to convert SPERT predictions to NEREL format.
*Implemented in `pred_to_nerel(batches_count, sentinfo, pred_data)`*
### Training and Predicting
Training and Predicting are done as in [here](https://github.com/lavis-nlp/spert/#examples).
The only differences are:
- there are no relations in NEREL dataset
- different BERT is used (`cointegrated/rubert-tiny` and `ai-forever/ruBert-large`)
### Converting safetensors
In this notebook, `model.safetensors` if converted to `pytorch_model.bin` to avoid compatibility problems.

## Overall pipeline
Overall train-to-predict pipeline would look like that:
1. Convert NEREL dataset to SPERT format
2. Train SpERT model
3. Convert safetensors
4. Predict
5. Convert predictions to NEREL format

## Results
### Solution 1. SpERT with `rubert-large`
`rubert-large-15epochs`. Dev set:
```
Mention F1:         83.00%
Mention recall:     83.28%
Mention precision:  82.73%
Macro F1:           74.54%
Macro F1 few-shot:  0.00%
```
`rubert-large-20epochs`. Dev set:
```
Mention F1:         82.54%
Mention recall:     83.21%
Mention precision:  81.88%
Macro F1:           74.08%
Macro F1 few-shot:  0.00%
```
`rubert-large-30epochs`. Dev set:
```
Mention F1:         83.00%
Mention recall:     83.53%
Mention precision:  82.47%
Macro F1:           74.02%
Macro F1 few-shot:  0.00%
```
### Solution 2. SpERT with `rubert-tiny`
`rubert-tiny-20epochs`. Dev set:
```
Mention F1:         69.72%
Mention recall:     71.47%
Mention precision:  68.05%
Macro F1:           55.92%
Macro F1 few-shot:  0.00%
```
## Conculsion
`rubert-large` has shown to be the best in terms of `F1` score, with insignificant difference between models with different number of training epochs. On other hand, `rubert-tiny` was too modest to capture complex nature of NEREL dataset.
