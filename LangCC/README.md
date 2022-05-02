# LangCC

This dataset is created for intermediate training purpose, in order to encode knowledge via the language-specific text corpus: [CC-100](https://data.statmt.org/cc-100/).
You can simply download the full data from [here](https://drive.google.com/drive/folders/18WSQQp6omwgR1KHzmyw13KQBBdd3voHB?usp=sharing) or you can modify the scripts for your own usage.


## Download files from OpenSubtitles
```
* Chinese
wget https://data.statmt.org/cc-100/zh-Hans.txt.xz -O zh-Hans.txt.xz

* Arabic
wget https://data.statmt.org/cc-100/ar.xz -O ar.txt.xz

* Russian
wget https://data.statmt.org/cc-100/ru.xz -O ru.txt.xz

* German
wget https://data.statmt.org/cc-100/de.xz -O de.txt.xz
```

## Prepare files
```
python langcc_extract.py --input_lang_file "./de.txt.xz" --save_file_name "./langcc/cc_de_500K.txt" --max_line 500000
python langcc_prep.py --input_lang_file="./langcc/cc_de_500K.txt" --save_train_file_name="./langcc/cc_de_train_200K.txt" --save_test_file_name="./langcc/cc_de_test_10K.txt"
```