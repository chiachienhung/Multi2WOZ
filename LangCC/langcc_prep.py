import random
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type = int, default = 10, help="set random seed")
    parser.add_argument('--train_size', type=int, default = 200000, help="size for training set")
    parser.add_argument('--test_size', type=int, default = 10000, help="size for testing set")
    parser.add_argument('--input_lang_file', type=str, help="input language file path") #"./langcc/cc_de_500K.txt"
    parser.add_argument('--save_train_file_name', type=str, default = "./train/train.txt", help = "file name for training data")
    parser.add_argument('--save_test_file_name', type=str, default = "./test/test.txt", help = "file name for testing data")
    return parser.parse_args()

def remove_puncts(text):
    return re.sub(r"\.+", ".", text)

def remove_email(text):
    text = re.sub(r"\[…\]", " ", text)
    text = re.sub(r"\S*@\S*\s?", "", text)
    return re.sub(r"\_+", " ", text)

def remove_quotes(text):
    text = text.replace("'", "").replace("…", "").replace(". .", ".")
    text = re.sub(r"[\x08|\x06|\x05|\x07|\xad|\u200b|\x96|\x97|█|\u200f|\u200c|\u200e|\u200d|\u061c]+", "", text)
    return re.sub('[`"“„»«<>↑~”•●]', " ", text) #ar, ru, de
    #return re.sub('[`"“„»«<>↑~”•●]', " ", text) #cn

def remove_url(text):
    text = re.sub(r"https?://\S+|www\.\S+|\w*\.[com|org|de|ru|ar|cn]+/*\w*|\w+//\w+:*/*|https?:\w*", " [URL] ", text, flags=re.IGNORECASE)
    text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " [URL] ", text, flags=re.IGNORECASE)
    text = re.sub(' +', ' ', text)
    text = text.replace("[ [URL]", " [URL]").replace("( [URL] )", " [URL]").replace("( [URL]", "[URL]").replace("[URL] )", "[URL]").replace("[URL] -", "[URL]").replace("[URL] [URL]", "[URL]")
    text = text.replace("[ [URL]", " [URL]").replace("（ [URL] ）", " [URL]").replace("[URL] [URL]", "[URL]").replace("（ [URL]", "[URL]").replace("[URL] ）", "[URL]")
    text = re.sub(' +', ' ', text)
    text = text.replace("[URL] [URL]", " [URL]")
    text = re.sub(r'[/\\]+', " ", text)
    return re.sub(' +', ' ', text) #de, ar, ru, en
    #return re.sub(' +', '', text) #cn

def save_file(file_name, corpus_list, max_len = 10000):
    c = 0
    with open(file_name, 'a') as s:
        for i, element in enumerate(corpus_list):
            element = remove_email(remove_puncts(remove_url(remove_quotes(element))))
            element = element.replace('[ url ]', '[URL]')
            element = element.strip()
            if element!="[URL]" and len(element)>30 and c<max_len:
                c+=1
                s.write("{}\n".format(element))
            if i%10000==0:
                print(i)            

if __name__ == '__main__':
    args = parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    with open(args.input_lang_file, 'r') as f:
        data = f.read().split('\n')
        data = data[0:500000]
    print("Original data size: {}".format(len(data)))
    random.shuffle(data)
    train, test = train_test_split(data, test_size=0.10, random_state=args.random_seed)
    train = train[0:args.train_size+10000]
    test = test[0:args.test_size+10000]
    print("Training data size: {}".format(len(train)))
    print("Testing data size: {}".format(len(test)))
    save_file(args.save_test_file_name, test, max_len=args.test_size)
    save_file(args.save_train_file_name, train, max_len=args.train_size)
    
    #cd ./XLM
    #from https://github.com/facebookresearch/XLM/tree/cd281d32612d145c6742b4d3f048f80df8669c30
    ###en, de, ar, ru###
    #cat ../langcc/cc_de_test_10K.txt  | ./tools/tokenize.sh de | python ./tools/lowercase_and_remove_accent.py > ../langcc/cc_de_test_10K_final.txt
    #cat ../langcc/cc_de_train_200K.txt  | ./tools/tokenize.sh de | python ./tools/lowercase_and_remove_accent.py > ../langcc/cc_de_train_200K_final.txt
    ###zh-cn###
    #cat ../langcc/cc_cn_test_10K.txt  | ./tools/tokenize.sh zh | python ./tools/lowercase_and_remove_accent.py > ../langcc/cc_cn_test_10K_final.txt
    #cat ../langcc/cc_cn_train_200K.txt  | ./tools/tokenize.sh zh | python ./tools/lowercase_and_remove_accent.py > ../langcc/cc_cn_train_200K_final.txt