# Language Specialization

For language specialization, here are multiple methods you can implement for your own usage:

- **mlm (LangCC)**
```
./run_tod_intermediate_mlm.sh 0 "umanlp/TOD-XLMR" save/lang_eval/de/TOD-XLMR_MLM_prep_patience3_max256 ../LangCC/Mono-CC/train/cc_de_train_200K_final.txt ../LangCC/Mono-CC/test/cc_de_test_10K_final.txt --overwrite_output_dir --patience 3
```

- **mlm-tlm (LangOpenSubtitles)**
```
./run_tod_intermediate_mlm.sh 0 "umanlp/TOD-XLMR" save/lang_eval/de/TOD-XLMR_MLM_TLM_prep_patience3_max256 ../LangOpenSubtitles/mlm-tlm/tlm_mlm_200K.en-de.txt ../LangOpenSubtitles/mlm-tlm/tlm_mlm_10K.en-de.txt --overwrite_output_dir --patience 3
```

- **rs-mono (LangOpenSubtitles)**
```
./run_tod_intermediate_rs.sh 0 "umanlp/TOD-XLMR" save/lang_eval/de/TOD-XLMR_rsmono "de-de" --overwrite_output_dir --do_lower_case --patience 3
```

- **rs-x (LangOpenSubtitles)**
```
./run_tod_intermediate_rs.sh 0 "umanlp/TOD-XLMR" save/lang_eval/de/TOD-XLMR_rsx "de" --overwrite_output_dir --do_lower_case --patience 3
```


