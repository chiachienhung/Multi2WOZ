import json
import argparse
import linecache

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_file', type = str, help="src language file")
    parser.add_argument('--tgt_file', type = str, help="tgt language file")
    parser.add_argument('--ids', type = str, help="ids file")
    parser.add_argument('--save_file_name', type=str, help = "save_file_name as json")
    return parser.parse_args()

def save_file(dict_to_save, filepath):
    with open(filepath, 'w') as file:
        json.dump(dict_to_save, file)

if __name__ == '__main__':
    args = parse_args()

    with open(args.ids, "r") as f3:
        k=0
        a = {}
        for index, line_ids in enumerate(f3):
            imdbid = line_ids.split('\t')[0].split(".")[0]
            curr_c = index
            prev_c = index-1
            if imdbid not in a.keys(): #new ids
                a[imdbid] = {}
                a[imdbid]['start']=index
                if index!=0:
                    prev_imdbid = linecache.getline(args.ids, index-1).split('\t')[0].split(".")[0]
                    a[prev_imdbid]['end']=index-1
            if index%2000000==0 and index!=0:
                print("Process index {}...".format(index))
        a[imdbid]['end']=index

        f3.close()
    print("Total utterances: {}".format(str(index)))
    print("Total imdbs: {}".format(str(sum([len(v) for k, v in a.items()]))))
    save_file(a, args.save_file_name)
