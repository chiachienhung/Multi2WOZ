import json
import argparse
import re, os
import random
import glob
import numpy as np
import linecache
import unicodedata
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, help='First file to process')
    parser.add_argument('--file2', type=str, nargs="?")
    parser.add_argument('--fileids', type=str, help='File with ids to process')
    parser.add_argument('--ofile', type=str)
    parser.add_argument('--lang', type=str)
    return parser.parse_args()

def load_file(file):
    with open(file, 'r') as f:
        for index, _ in enumerate(f):
            a = index
    f.close()
    print("Number of lines: {}".format(a))
    return a

def load_dial_json(filename):
    with open(filename, "r") as f:
        dial_json = json.load(f)
    return dial_json

def save_dial_json(json_dial, filename):
    with open(filename, "w") as f:
        json.dump(json_dial, f, indent=2, separators=(",", ": "), sort_keys=False)

def prep_text(text):
    characters = ["", "{", "}", "", "", "", "", ""]
    if not any(j for j in text if unicodedata.category(j).startswith('C')) and not any(j in text for j in characters):
        return True
    else:
        return False
    
def get_data(args, data, total_lines):
    context = []
    response = []
    label = []
    c = 0
    for dial in data:
        if prep_text(dial['context']) and prep_text(dial['response']) and prep_text(dial['false_response']):
            if dial['curr_lang']==1:
                file = args.file2 #tgt file
            else:
                file = args.file1
            context.append(dial['context'])
            response.append(dial['response'])
            label.append(1)
            context.append(dial['context'])
            response.append(dial['false_response'])
            label.append(0)
            # sampel negative samples
            negative_sampling = random.randint(1,3)
            all_id = dial['context_ids']+[dial['response_id']]+[dial['false_response_id']]
            i=0
            ids = []
            while(i<negative_sampling):
                f_resp_id = random.randint(0,total_lines)
                if f_resp_id not in all_id:
                    f_resp = linecache.getline(file, f_resp_id+1).strip()
                    if prep_text(f_resp) and len(f_resp)>10:
                        context.append(dial['context'])
                        response.append(f_resp)
                        label.append(0)
                        all_id.append(f_resp_id)
                        i+=1
        c+=1
    return context, response, label

def convert_to_json(context, response, label):
    dialogues = []
    for c, r, l in zip(context, response, label):
        dials = {}
        dials['context'] = c
        dials['response'] = r
        dials['label'] = l
        dialogues.append(dials)
    return dialogues

if __name__ == '__main__':
    args = parse_args()
    #all_files = glob.glob("./" + args.lang + "/*K.pkl") #for rs-x
    all_files = glob.glob("./" + args.lang + "/*mono*K.pkl") #for rs-mono
    total_lines = load_file(args.fileids)
    dialogues = []
    for file in all_files:
        dialog = pickle.load(open(file, "rb"))
        print(len(dialog), file)
        context, response, label = get_data(args, dialog, total_lines)
        dialogues += convert_to_json(context, response, label)
        print(len(context), len(dialogues))
    save_dial_json(dialogues, "./" + args.lang + '/prep/' + args.ofile)
    print(len(dialogues))
    bs = 400000
    for i in range(0, len(dialogues), bs):
        pickle.dump(dialogues[i:i+bs], open("./" + args.lang + '/prep/' + args.ofile+str(i)+"K.pkl", "wb"), protocol=2) # converts array to binary and writes to output
        print(len(dialogues[i:i+bs]))