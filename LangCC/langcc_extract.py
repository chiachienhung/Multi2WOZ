import os
import lzma
import re
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_lang_file', type=str, help="input language file path") #"./de.txt.xz
    parser.add_argument('--save_file_name', type=str, default = "./cc_de_500K.txt", help = "save file name for language data")
    parser.add_argument('--max_line', type=int, default=500000)
    return parser.parse_args()

def is_match(regex, text):
    pattern = re.compile(regex)
    return pattern.search(text, re.IGNORECASE) is not None
def match_num(regex, text):
    pattern = re.compile(regex, re.IGNORECASE)
    return pattern.findall(text)
    #return len(pattern.findall(text))

def store_line(filename):
    if os.path.exists(filename):
        mode = 'a'
    else:
        mode = 'w'
    return mode

if __name__ == '__main__':
    args = parse_args()
    count = 0
    extract_list = []
    num = []
    with lzma.open(args.input_lang_file, mode='rt') as file:
        for i, line in enumerate(file):
            if len(line.split())>30: ##ar, ru, de, en
            #if len(line)>30: ## cn
                extract_list.append(line)
                num.append(len(extract_list))
            if len(extract_list)>1000:
                mode = store_line(args.save_file_name)
                with open(args.save_file_name, mode) as s:
                    for element in extract_list:
                        s.write(element)
                count +=1000
                extract_list = []
            if i%100000==0 and i!=0:
                print("Load {}".format(i))
            if len(num)>args.max_line:
                print(i)
                break
    mode = store_line(args.save_file_name)
    if len(extract_list)!=0:
        with open(args.save_file_name, mode) as s:
            for element in extract_list:
                s.write(element)