# Multi2WOZ: A Robust Multilingual Dataset and Conversational Pretraining for Task-Oriented Dialog

Authors: Chia-Chien Hung, Anne Lauscher, Ivan Vulić, Simone Paolo Ponzetto, Goran Glavaš

NAACL 2022 Main Conference: Coming soon

## Introduction
Research on (multi-domain) task-oriented dialog (TOD) has predominantly focused on the English language, primarily due to the shortage of robust TOD datasets in other languages, preventing the systematic investigation of cross-lingual transfer for this crucial NLP application area. In this work, we introduce Multi2WOZ, a new multilingual multi-domain TOD dataset, derived from the well-established English dataset MultiWOZ, that spans four typologically diverse languages: Chinese, German, Arabic, and Russian. In contrast to concurrent efforts, Multi2WOZ contains gold-standard dialogs in target languages that are directly comparable with development and test portions of the English dataset, enabling reliable and comparative estimates of cross-lingual transfer performance for TOD. This enables us to detect and explore crucial challenges for TOD cross-lingually. We then introduce a new framework for multilingual conversational specialization of pretrained language models (PrLMs) that aims to facilitate cross-lingual transfer for arbitrary downstream TOD tasks. Using such conversational PrLMs specialized for concrete target languages, we systematically benchmark a number of zero-shot and few-shot cross-lingual transfer approaches on two standard TOD tasks: Dialog State Tracking and Response Retrieval. Our experiments show that, in most setups, the best performance entails the combination of (i) conversational specialization in the target language and (ii) few-shot transfer for the concrete TOD task. Most importantly, we show that our conversational specialization in the target language allows for a much more sample-efficient few-shot transfer for downstream TOD tasks.

## Citation
If you use any source codes, or datasets included in this repo in your work, please cite the following paper:
<pre>

</pre>

## Pretrained Models
The pre-trained models can be easily loaded using huggingface [Transformers](https://github.com/huggingface/transformers) or Adapter-Hub [adapter-transformers](https://github.com/Adapter-Hub/adapter-transformers) library using the **AutoModel** function. Following pre-trained versions are supported:
* `umanlp/TOD-XLMR`: XLMR pre-trained using both MLM and RCL objectives, which can be found [here](https://huggingface.co/umanlp/TOD-XLMR/tree/main)
* `xlm-roberta-base`

```
import torch
import transformers
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("umanlp/TOD-XLMR")
model = AutoModel.from_pretrained("umanlp/TOD-XLMR")
```
or
```
import torch
import transformers
from transformers import XLMRobertaModel, XLMRobertaTokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained("umanlp/TOD-XLMR")
model = XLMRobertaModel.from_pretrained("umanlp/TOD-XLMR")
```

For running pretraining for `TOD-XLMR`, please check `downstream` folder for more details.

The scripts for downstream tasks are mainly modified from [here](https://github.com/jasonwu0731/ToD-BERT), where there might be slight version differences of the packages, which are noted down in the `requirements.txt` file.

## Datasets

### Multi2WOZ
Multi2WOZ is a multilingual dataset translated from MultiWOZ-2.1. We focus on the development and test portions of the English MultiWOZ 2.1 (in total of 2,000 dialogs containing the total of 29.5K utterances) and translate the monolingual English-only MultiWOZ dataset to four linguistically diverse languages: Arabic (AR), Chinese (ZH), German (DE), and Russian (RU). 
The datasets can be downloaded from [Multi2WOZ](https://drive.google.com/drive/folders/18olGkUl8mVDpQ4WdNUOOT30y2roIE4O5?usp=sharing). 

### LangCC and LangOpenSubtitles
Two datasets **LangCC** and **LangOpenSubtitles** are created for intermediate training purpose, in order to encode knowledge via the language-specific corpus.
You can simply download the data from [LangCC](https://drive.google.com/drive/folders/18WSQQp6omwgR1KHzmyw13KQBBdd3voHB?usp=sharing) and [LangOpenSubtitles](https://drive.google.com/drive/folders/1SR8_37shLu8cRmb5WHWfLAHAeir1y9hS?usp=sharing). Or you can modify the scripts for your own usage.


## Structure
This repository is currently under the following structure:
```
.
└── LangCC
└── LangOpenSubtitles
└── downstream
    └── models
    └── utils
    └── dialog_datasets
└── specialization
└── README.md
```
