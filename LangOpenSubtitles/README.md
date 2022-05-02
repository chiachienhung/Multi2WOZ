# LangOpenSubtitles

This dataset is created for intermediate training purpose, in order to encode knowledge via the language-specific dialogic corpus: [OpenSubtitles](https://opus.nlpl.eu/OpenSubtitles-v2018.php).
You can simply download the full data from [here](https://drive.google.com/drive/folders/1SR8_37shLu8cRmb5WHWfLAHAeir1y9hS?usp=sharing) or you can modify the scripts for your own usage. The scripts are modified from [xlift_dst](https://github.com/nikitacs16/xlift_dst/tree/main/intermediate_finetuning).


## Download files from OpenSubtitles
```
* en-zh
wget http://opus.nlpl.eu/download.php?f=OpenSubtitles2016%2Fen-zh.txt.zip -O en-zh.zip
unzip en-zh.zip -d ./en-zh

* en-ar
wget http://opus.nlpl.eu/download.php?f=OpenSubtitles2018%2Far-en.txt.zip -O en-ar.zip
unzip en-ar.zip -d ./en-ar

* en-ru
wget http://opus.nlpl.eu/download.php?f=OpenSubtitles2018%2Fen-ru.txt.zip -O en-ru.zip
unzip en-ru.zip -d ./en-ru

* en-de
wget http://opus.nlpl.eu/download.php?f=OpenSubtitles2018%2Fde-en.txt.zip -O en-de.zip
unzip en-de.zip -d ./en-de
```

## Prepare IMDBIds
```
python extract_imdb_ids.py --src_file "./data/en-de/OpenSubtitles.de-en.en" --tgt_file "./data/en-de/OpenSubtitles.de-en.de" --ids "./data/en-de/OpenSubtitles.de-en.ids" --save_file_name "./data/en-de_imdbs.json"
```

## Select Methods

**1. mlm-tlm**
```
python convert_mlm.py --file1 "./data/en-de/OpenSubtitles.de-en.en" --file2 "./data/en-de/OpenSubtitles.de-en.de" --fileids "./data/en-de/OpenSubtitles.de-en.ids" --ofile tlm_mlm_300K.en-de.txt --tlm --count 300000 --split
```

**2. rs-mono**
```
python convert_mlm.py --file1 "./data/en-de/OpenSubtitles.de-en.de" --file2 "./data/en-de/OpenSubtitles.de-en.de" --fileids "./data/en-de/OpenSubtitles.de-en.ids" --fileids_lines "./data/en-de_imdbs.json" --ofile rs_mono_dialogue.de-de_ --rs_dialogue --count 100000 --max_length 4

python concat_files.py --file1 "./data/en-de/OpenSubtitles.de-en.de" --file2 "./data/en-de/OpenSubtitles.de-en.de" --fileids "./data/en-de/OpenSubtitles.de-en.ids" --ofile "rs_mono_dialogue_300K.de-de.json" --lang "de"
```

**3. rs-x**
```
python convert_mlm.py --file1 "./data/en-de/OpenSubtitles.de-en.en" --file2 "./data/en-de/OpenSubtitles.de-en.de" --fileids "./data/en-de/OpenSubtitles.de-en.ids" --fileids_lines "./data/en-de_imdbs.json" --ofile rs_dialogue_300K.en-de.pkl --rs_dialogue --count 100000 --max_length 4

python concat_files.py --file1 "./data/en-de/OpenSubtitles.de-en.en" --file2 "./data/en-de/OpenSubtitles.de-en.de" --fileids "./data/en-de/OpenSubtitles.de-en.ids" --ofile "rs_dialogue_300K.en-de.json" --lang "de"
```