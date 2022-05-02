# Training and Evaluation

## TOD-XLMR pretraining
TOD-XLMR is trained on the concatenation of nine task-oriented dialog datasets, where the full dialog datasets can be downloaded from [here](https://drive.google.com/file/d/1EnGX0UF4KW6rVBKMF3fL-9Q2ZyFKNOIy/view?usp=sharing).

The scripts are mainly modified from [TOD-BERT](https://github.com/jasonwu0731/ToD-BERT), and can run the pretraining as follows:

```
./run_tod_lm_pretraining.sh 0 xlmr xlm-roberta-base ./save/pretrain/ToD-XLMR --only_last_turn --add_rs_loss -dpath /path_to_store_the_dialog_datasets_folder/dialog_datasets --fp16 --overwrite_output_dir --domain "all"
```

For simple usage, you can easily load the pre-trained model using huggingface Transformers library using the `AutoModel` function. The following pre-trained version is supported:

* `umanlp/TOD-XLMR`: XLMR pre-trained using both MLM and RCL objectives, which can be found [here](https://huggingface.co/umanlp/TOD-XLMR/tree/main)

## Evaluation
Datsets can be downloaded from [here](https://drive.google.com/file/d/1l5DYhPXk2uhdXd4EsfGMhk9v382uLh3f/view?usp=sharing) and run the evaluation:

- **Zero-shot**:
```
./evaluation_pipeline_cl.sh 0 xlmroberta "umanlp/TOD-XLMR" save/de/TOD-XLMR-zeroshot -dpath /path_to_store_the_dialog_datasets_folder/dialog_datasets --tgt_lang="de" --overwrite
```
- **Few-shot**:
```
./evaluation_pipeline_cl-continue.sh 0 xlmroberta "umanlp/TOD-XLMR" save/de/TOD-XLMR-fewshot-full "save/de/TOD-XLMR-zeroshot/RR/MWOZ/run0/pytorch_model.bin" -dpath /path_to_store_the_dialog_datasets_folder/dialog_datasets --tgt_lang="de" --overwrite
```