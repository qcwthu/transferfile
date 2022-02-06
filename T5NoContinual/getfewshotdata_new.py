import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import sys
import argparse
import matplotlib
import pdb
import numpy as np
import time
import random
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import trange
from sklearn import metrics
from torch.utils import data
from collections import Counter
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler

def seed_everything(args):
    random.seed(args.seed)
    os.environ['PYTHONASSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def handlefile(inputfile,outputfile,shotnum):
    f = open(inputfile,'r')
    allres = {}
    while True:
        line = f.readline().strip()
        if not line:
            break
        linelist = line.split("\t")
        if len(linelist) != 2:
            continue
        entitylist = linelist[1]
        if entitylist == 'end':
            continue
        thistype = entitylist.split(";")[0].split("!")[1].strip(' ')
        #print(thistype)
        if thistype not in allres:
            allres[thistype] = []
            allres[thistype].append(line)
        else:
            allres[thistype].append(line)
    f.close()
    print(len(allres))
    for key in allres.keys():
        print(key," : ",len(allres[key]))
    tousetres = {}
    for key in allres.keys():
        if shotnum < len(allres[key]):
            thisres = random.sample(allres[key], shotnum)
        else:
            thisres = allres[key]
        tousetres[key] = thisres

    fo = open(outputfile, 'w')
    allres = []
    for key in tousetres.keys():
        allres.extend(tousetres[key])
    random.shuffle(allres)
    for oneres in allres:
        fo.write(oneres+"\n")
    fo.close()

def getinitialfile(inputfile,originfile,outputfile):
    f = open(inputfile, 'r')
    allres = {}
    while True:
        line = f.readline().strip()
        if not line:
            break
        linelist = line.split("\t")
        if len(linelist)!=2:
            continue
        thissen = linelist[0]
        if thissen not in allres:
            allres[thissen] = 1
        else:
            print("error!!!!!!")
    f.close()
    print(len(allres))
    f1 = open(originfile,'r')
    data = []
    sentence = []
    label = []
    for line in f1:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                tousesen = ' '.join(sentence)
                if tousesen in allres.keys() and allres[tousesen] == 1:
                    #print("this one is used!")
                    #print(tousesen)
                    data.append((sentence, label))
                    allres[tousesen] = -1
                sentence = []
                label = []
            continue
        splits = line.split('\t')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])
    if len(sentence) > 0:
        tousesen = ' '.join(sentence)
        if tousesen in allres.keys() and allres[tousesen] == 1:
            #print("this one is used!")
            data.append((sentence, label))
            allres[tousesen] = -1
        sentence = []
        label = []
    f1.close()
    print(len(data))
    fo = open(outputfile,'w')
    for (onesen,onelable) in data:
        #print(onesen,onelable)
        for i in range(len(onesen)):
            fo.write(onesen[i]+" "+onelable[i]+"\n")
        fo.write("\n")
    fo.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="latentRE")

    parser.add_argument("--model", dest="model", type=str,
                        default="T5", help="{T5}")
    ###first 42
    ###second 160
    parser.add_argument("--seed", dest="seed", type=int,
                        default=242, help="seed for network")

    args = parser.parse_args()

    seed_everything(args)

    if args.model == "T5":
        #t5model = T5ForConditionalGeneration.from_pretrained(args.model_name)
        print("right!")
        shotnumber = 16

        handlefile("/data/qin/T5/T5forNER/data_conll/newtrain.txt","NERdata/train.txt",shotnumber)
        handlefile("/data/qin/T5/T5forNER/data_conll/newvalid.txt", "NERdata/valid.txt",shotnumber)
    else:
        raise Exception("No such model! Please make sure that `model` takes the value in {T5}")


