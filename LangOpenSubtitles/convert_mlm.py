import sys
import random
import argparse
from tqdm import *
import numpy as np
from sklearn.model_selection import train_test_split
import unicodedata
import json
import linecache
import pickle

# bi: 0
# mono: 1
random.seed(1)
np.random.seed(1)

def parse_args():
    parser = argparse.ArgumentParser(description='Process to chat-level files')
    parser.add_argument('--file1', type=str, help='First file to process')
    parser.add_argument('--file2', type=str, help="Second file to process")
    parser.add_argument('--fileids', type=str, help='File with ids to process')
    parser.add_argument('--fileids_lines', type=str, help='File with unique ids and start and end for each dialogue')
    parser.add_argument('--ofile', type=str)
    parser.add_argument('--xdm', action='store_true')
    parser.add_argument('--xdm_dialogue', action='store_true')
    parser.add_argument('--rs_dialogue', action='store_true')
    parser.add_argument('--single', action='store_true')
    parser.add_argument('--mixed', action='store_true')
    parser.add_argument('--tlm', action='store_true')
    parser.add_argument('--monodm', action='store_true') 
    parser.add_argument('--response', action='store_true')
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--count', type=int, default=200000)
    parser.add_argument('--max_length', type=int, default=15)
    return parser.parse_args()

def coin_toss():
    if random.random() < 0.5:
        return 1
    return 2

def load_dial_json(filename):
    with open(filename, "r") as f:
        dial_json = json.load(f)
    return dial_json

def get_hard_neg_response(args, all_ids, curr_id, curr_lang, orig_resp, context_ids): # same imbdbid
    all_turns = []
    start = all_ids[curr_id]['start']
    end = all_ids[curr_id]['end']
    #print(start, end)
    if curr_lang==1:
        file = args.file2 #tgt file
    else:
        file = args.file1
    for i in range(start+1, end+2):
        all_turns.append(linecache.getline(file, i).strip())
    rand_index_false_resp = random.choice(list(range(0,len(all_turns))))
    false_resp = all_turns[rand_index_false_resp]
    
    while rand_index_false_resp+start in context_ids or false_resp==orig_resp:
        rand_index_false_resp = random.choice(list(range(0,len(all_turns))))
        false_resp = all_turns[rand_index_false_resp]
    return false_resp, rand_index_false_resp+start


def rs_dialogue():
    with open(args.fileids, 'r') as f:
        for index, _ in tqdm(enumerate(f)):
            a = index
    f.close()
    index = 0
    curr_count = 0
    cont_resp = []
    while (curr_count != chat_count) and (index<=a-2):
        k = random.randint(2, max_length)
        curr_lang = coin_toss()
        s = ""
        ids = []
        curr_c = 0
        context_ids = []
        with open(args.file1, "r") as f1, open(args.file2, "r") as f2, open(args.fileids, "r") as fid:
            for index, (line_R1, line_R2, line_id) in tqdm(enumerate(zip(f1, f2, fid))): #specify start
                if index>a-2:
                    break
                infos = line_id.split('\t')[0].split('/')
                curr_year, curr_imdbid = infos[1], infos[2]
                curr_id = line_id.split('\t')[0].split(".")[0]
                if curr_c != k: #concat src language
                    if curr_lang == 1:
                        s = s + " " + line_R1.strip()
                        ids.append(curr_imdbid)
                        context_ids.append(index)
                        cr = 0 #src
                        resp = line_R1.strip()
                    else:
                        s = s + " " + line_R2.strip()
                        ids.append(curr_imdbid)
                        context_ids.append(index)
                        resp = line_R2.strip()
                        cr = 1 #tgt
                    curr_c = curr_c + 1
                else: #append response
                    dials = {}
                    if curr_lang != 1:
                        ids.append(curr_imdbid)
                        cr = 0 #src
                        resp = line_R1.strip()
                        dials['ids'] = curr_id
                        dials['curr_lang'] = curr_lang #context_lang = "tgt_lang", resp_lang = "en"
                        dials['context'] = s
                        dials['context_ids'] = context_ids
                        dials['response'] = line_R1.strip()
                        dials['response_id'] = index
                        dials['false_response'], dials['false_response_id'] = get_hard_neg_response(args, all_ids, curr_id, curr_lang, line_R1.strip(), context_ids)
                        context_ids = []
                    else:
                        #s = s + " " + line_R2.strip()
                        ids.append(curr_imdbid)
                        resp = line_R2.strip()
                        cr = 1 #tgt
                        dials['ids'] = curr_id
                        dials['curr_lang'] = curr_lang
                        dials['context'] = s
                        dials['context_ids'] = context_ids
                        dials['response'] = line_R2.strip()
                        dials['response_id'] = index
                        dials['false_response'], dials['false_response_id'] = get_hard_neg_response(args, all_ids, curr_id, curr_lang, line_R2.strip(), context_ids)
                        context_ids = []
                    #if (cr==0 and len(s)+len(resp.split()) < 256 and len(s)+len(dials['false_response'].split()) < 256) or (cr==1 and len(s.split())+len(resp) < 256 and len(s.split())+len(dials['false_response']) < 256): #->for chinese
                    #if (cr==0 and len(s)+len(resp) < 256 and len(s)+len(dials['false_response']) < 256) or (cr==1 and len(s)+len(resp) < 256 and len(s)+len(dials['false_response']) < 256): #->for mono chinese
                    if (cr==0 and len(s.split())+len(resp.split()) < 256 and len(s.split())+len(dials['false_response'].split()) < 256) or (cr==1 and len(s.split())+len(resp.split()) < 256 and len(s.split())+len(dials['false_response'].split()) < 256):
                        if len(set(ids))==1:
                            #if (cr==1 and len(resp)>10 and len(dials['false_response'])>10) or (cr==0 and len(resp.split())>10 and len(dials['false_response'].split())>10): #-> for chinese
                            #if (cr==1 and len(resp)>10 and len(dials['false_response'])>10) or (cr==0 and len(resp)>10 and len(dials['false_response'])>10): #-> for mono chinese
                            if (cr==1 and len(resp.split())>10 and len(dials['false_response'].split())>10) or (cr==0 and len(resp.split())>10 and len(dials['false_response'].split())>10): #-> for others
                                curr_count = curr_count + 1
                                cont_resp.append(dials)
                                ids = []
                                context_ids = []
                                if curr_count >= chat_count:
                                    break
                        else:
                            #print(ids)
                            #print(s.strip())
                            ids = []
                            context_ids = []
                    if curr_count%10000==0 and curr_count!=0:
                        print(curr_count)
                    curr_c = 0
                    curr_lang = coin_toss()
                    s = ""
                    k = random.randint(2, max_length)
        f1.close()
        f2.close()
    
    bs = 20000
    for i in range(0, len(cont_resp), bs):
        pickle.dump(cont_resp[i:i+bs], open(args.ofile+str(i)+"K.pkl", "wb"), protocol=2) # converts array to binary and writes to output
        print(len(cont_resp[i:i+bs]))

def xdm_dialogue():
    f3 = open(args.ofile, 'w')
    curr_count = 0
    while curr_count != chat_count:
        k = random.randint(2, max_length)
        curr_lang = coin_toss()
        s = ""
        ids = []
        curr_c = 0
        with open(args.file1, "r") as f1, open(args.file2, "r") as f2, open(args.fileids, "r") as fid:
            for index, (line_R1, line_R2, line_id) in tqdm(enumerate(zip(f1, f2, fid))): #specify start
                infos = line_id.split('\t')[0].split('/')
                curr_year, curr_imdbid = infos[1], infos[2]
                if curr_c != k: #concat src language
                    if curr_lang == 1:
                        s = s + " " + line_R1.strip()
                        ids.append(curr_imdbid)
                        cr = 0 #src
                        resp = line_R1.strip()
                    else:
                        s = s + " " + line_R2.strip()
                        ids.append(curr_imdbid)
                        resp = line_R2.strip()
                        cr = 1 #tgt
                    curr_c = curr_c + 1
                else: #append response
                    if curr_lang != 1:
                        s = s + " " + line_R1.strip()
                        ids.append(curr_imdbid)
                        cr = 0 #src
                        resp = line_R1.strip()
                    else:
                        s = s + " " + line_R2.strip()
                        ids.append(curr_imdbid)
                        resp = line_R2.strip()
                        cr = 1 #tgt

                    if len(s.split()) < 256:
                        if len(set(ids))==1:
                            #if (cr==1 and len(resp)>10) or (cr==0 and len(resp.split())>10): #-> for chinese
                            if (cr==1 and len(resp.split())>10) or (cr==0 and len(resp.split())>10):
                                curr_count = curr_count + 1
                                #f3.write(" ".join(ids) + '\n')
                                f3.write(s.strip() + '\n')
                                #print(set(ids))
                                #print(s.strip())
                                ids = []
                                #print(index)
                                #index = random.randint(5, 9300000)
                                if curr_count >= chat_count:
                                    break
                        else:
                            print(ids)
                            print(s.strip())
                            ids = []

                    curr_c = 0
                    curr_lang = coin_toss()
                    s = ""
                    k = random.randint(2, max_length)
        f1.close()
        f2.close()

def tlm_dialogue():
    f3 = open(args.ofile, 'w')
    curr_count = 0
    while curr_count != chat_count:
        k = random.randint(2, max_length)
        curr_lang = coin_toss()
        s = ""
        t = ""
        ids = []
        curr_c = 0
        with open(args.file1, "r") as f1, open(args.file2, "r") as f2, open(args.fileids, "r") as fid:
            for index, (line_R1, line_R2, line_id) in tqdm(enumerate(zip(f1, f2, fid))): #specify start
                infos = line_id.split('\t')[0].split('/')
                curr_year, curr_imdbid = infos[1], infos[2]
                if curr_c != k:
                    s = s + " " + line_R1.strip()
                    t = t + " " + line_R2.strip()
                    ids.append(curr_imdbid)
                    curr_c = curr_c + 1
                else:
                    s = s + " " + line_R1.strip()
                    t = t + " " + line_R2.strip()
                    ids.append(curr_imdbid)
                    #if len(s.split())+len(t) < 256: #if tgt==chinese
                    if len(s.split())+len(t.split()) < 256:
                        if len(set(ids))==1:
                            curr_count = curr_count + 1
                            if curr_lang == 1: # tgt, src
                                s = t + " " + s
                            else:
                                s = s + " " + t
                            f3.write(s.strip() + '\n')
                            ids=[]
                            #print(index)
                            if curr_count >= chat_count:
                                break
                        else:
                            print(ids)
                            print(s.strip())
                            print(t.strip())
                            ids = []

                    curr_c = 0
                    curr_lang = coin_toss()
                    s = ""
                    t = ""
                    k = random.randint(2, max_length)

        f1.close()
        f2.close()
    
    
def parallel_dialogue():
    f3 = open(args.ofile, 'w')
    curr_count = 0
    while curr_count != chat_count:
        k = random.randint(2, max_length)
        curr_lang = coin_toss()
        s = ""
        curr_c = 0
        with open(args.file1, "r") as f1, open(args.file2, "r") as f2:
            for index, (line_R1, line_R2) in tqdm(enumerate(zip(f1, f2))):
                if curr_c != k:
                    if curr_lang == 1:
                        s = s + " " + line_R1.strip()
                    else:
                        s = s + " " + line_R2.strip()
                    curr_c = curr_c + 1
                else:
                    if curr_lang != 1:
                        s = s + " " + line_R1.strip()
                    else:
                        s = s + " " + line_R2.strip()

                    if len(s.split()) < 512:
                        curr_count = curr_count + 1
                        f3.write(s.strip() + '\n')
                        if curr_count >= chat_count:
                            break

                    curr_c = 0
                    curr_lang = coin_toss()
                    s = ""
                    k = random.randint(2, max_length)

        f1.close()
        f2.close()


def response_dialogue():
    f3 = open(args.ofile, 'w')
    curr_count = 0
    while curr_count != chat_count:
        k = random.randint(2, max_length)
        curr_lang = coin_toss()
        s = ""
        curr_c = 0
        with open(args.file1, "r") as f1, open(args.file2, "r") as f2:
            for index, (line_R1, line_R2) in tqdm(enumerate(zip(f1, f2))):
                if curr_c != k:
                    if curr_lang == 1:
                        s = s + " " + line_R1.strip()
                    else:
                        s = s + " " + line_R2.strip()
                    curr_c = curr_c + 1
                else:
                    if curr_lang != 1:
                        s = s + " <S> " + line_R1.strip()
                    else:
                        s = s + " <S> " + line_R2.strip()

                    if len(s.split()) < 512:
                        curr_count = curr_count + 1
                        f3.write(s.strip() + '\n')
                        if curr_count >= chat_count:
                            break

                    curr_c = 0
                    curr_lang = coin_toss()
                    s = ""
                    k = random.randint(2, max_length)

        f1.close()
        f2.close()


def single_dialogue():
    f3 = open(args.ofile, 'w')
    curr_count = 0
    while curr_count != chat_count:
        k = random.randint(2, max_length)
        s = ""
        curr_c = 0
        with open(args.file1, "r") as f1:
            for line_R1 in f1:
                s = s + " " + line_R1.strip()
                curr_c = curr_c + 1
                if curr_c == k:
                    if len(s.split()) < 512:
                        curr_count = curr_count + 1
                        f3.write(s.strip() + '\n')
                        if curr_count >= chat_count:
                            break
                    curr_c = 0
                    s = ""
                    k = random.randint(2, max_length)

        f1.close()


def mixed_dialogue():
    f3 = open(args.ofile, 'w')
    curr_count = 0
    while curr_count != chat_count:
        k = random.randint(2, max_length)
        s = ""
        curr_c = 0
        set_flag = False
        with open(args.file1, "r") as f1, open(args.file2, "r") as f2:
            for index, (line_R1, line_R2) in tqdm(enumerate(zip(f1, f2))):
                if not set_flag:
                    curr_lang = coin_toss()

                if curr_c != k:
                    if curr_lang == 1:
                        s = s + " " + line_R1.strip()
                    else:
                        s = s + " " + line_R2.strip()
                    curr_c = curr_c + 1
                else:
                    if len(s.split()) < 512:
                        curr_count = curr_count + 1
                        f3.write(s.strip() + '\n')
                        if curr_count >= chat_count:
                            break

                    curr_c = 0
                    s = ""
                    k = random.randint(2, max_length)
                    set_flag = coin_toss()
                    if set_flag == 1:
                        set_flag = True
                        curr_lang = coin_toss()
                    else:
                        set_flag = False

        f1.close()
        f2.close()
    f3.close()

def bilingual_dialogue():
    f3 = open(args.ofile, 'w')
    curr_count = 0
    while curr_count != chat_count:
        k = random.randint(2, max_length)
        curr_lang = coin_toss()
        s = ""
        curr_c = 0
        with open(args.file1, "r") as f1, open(args.file2, "r") as f2:
            for index, (line_R1, line_R2) in tqdm(enumerate(zip(f1, f2))):
                if curr_c != k:
                    if curr_lang == 1:
                        s = s + " " + line_R1.strip()
                    else:
                        s = s + " " + line_R2.strip()
                    curr_c = curr_c + 1
                else:
                    if curr_lang == 1:
                        s = s + " " + line_R1.strip()
                    else:
                        s = s + " " + line_R2.strip()

                    if len(s.split()) < 512:
                        curr_count = curr_count + 1
                        f3.write(s.strip() + '\n')
                        if curr_count >= chat_count:
                            break

                    curr_c = 0
                    curr_lang = coin_toss()
                    s = ""
                    k = random.randint(2, max_length)

        f1.close()
        f2.close()

def split():
    with open(args.ofile, 'r') as f:
        data = f.read().split('\n')
        data = data[0:500000]
    print("Original data size: {}".format(len(data)))
    random.shuffle(data)
    train, test = train_test_split(data, test_size=0.10, random_state=0)
    train = train[0:200000+10000]
    test = test[0:10000+10000]
    print("Training data size: {}".format(len(train)))
    print("Testing data size: {}".format(len(test)))
    #print(test[-1])
    save_file("xdm_mlm_10K."+args.ofile.split('.')[-2]+".txt", test, max_len=10000)
    save_file("xdm_mlm_200K."+args.ofile.split('.')[-2]+".txt", train, max_len=200000)
    
def save_file(file_name, corpus_list, max_len = 10000):
    characters = ["", "{", "}", "", "", "", "", ""]
    c = 0
    with open(file_name, 'w') as s:
        for i, element in enumerate(corpus_list):
            if not any(j for j in element if unicodedata.category(j).startswith('C')) and not any(j in element for j in characters):
                element = element.strip()
                c+=1
                s.write("{}\n".format(element))
            else:
                print("False: {}".format(element))
            if i%10000==0:
                print(i)
            if c>max_len-1:
                break


if __name__ == '__main__':
    args = parse_args()
    chat_count = args.count
    max_length = args.max_length
    
    if args.xdm:
        parallel_dialogue()
    if args.mixed:
        mixed_dialogue()
    if args.single:
        single_dialogue()
    if args.tlm:
        tlm_dialogue()
    if args.monodm:
        bilingual_dialogue()

    if args.response:
        response_dialogue()

    if args.xdm_dialogue:
        xdm_dialogue()
    
    if args.rs_dialogue:
        all_ids = load_dial_json(args.fileids_lines)
        rs_dialogue()
        
    if args.split:
        split()
        
        