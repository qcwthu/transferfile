import torch
import os
import numpy as np
import random
import csv
import pickle
def seed_everything(args):
    random.seed(args.seed)
    os.environ['PYTHONASSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def getfewshot(inpath,outpath,fewshotnum):
    ###read from inpath
    intrain = inpath + "/train.txt"
    invalid = inpath + "/valid.txt"
    intest = inpath + "/test.txt"
    alltrainres = []
    allvalidres = []
    alltestres = []

    f = open(intrain,'r')
    while True:
        oneline = f.readline().strip()
        if not oneline:
            break
        linelist = oneline.split('\t')
        if len(linelist) != 2:
            continue
        alltrainres.append(oneline)
    f.close()

    f = open(invalid, 'r')
    while True:
        oneline = f.readline().strip()
        if not oneline:
            break
        linelist = oneline.split('\t')
        if len(linelist) != 2:
            continue
        allvalidres.append(oneline)
    f.close()

    f = open(intest, 'r')
    while True:
        oneline = f.readline().strip()
        if not oneline:
            break
        linelist = oneline.split('\t')
        if len(linelist) != 2:
            continue
        alltestres.append(oneline)
    f.close()

    ######select few shot for train valid and test
    ###outpath
    fewtrainname = outpath + "/train.txt"
    fewvalidname = outpath + "/valid.txt"
    fewtestname = outpath + "/test.txt"

    tousetrainres = random.sample(alltrainres, fewshotnum)
    tousevalidres = random.sample(allvalidres, fewshotnum)
    #testnum = 500
    testnum = 1000
    tousetestres = random.sample(alltestres, testnum)

    f = open(fewtrainname,'w')
    for one in tousetrainres:
        f.write(one + "\n")
    f.close()

    f = open(fewvalidname, 'w')
    for one in tousevalidres:
        f.write(one + "\n")
    f.close()

    ####test
    f = open(fewtestname, 'w')
    for one in tousetestres:
        f.write(one + "\n")
    f.close()

def getpromptembedding(model,tokenizer,promptnumber, alllabeltouse):
    t5_embedding = model.model.get_input_embeddings()
    promptinitembedding = torch.FloatTensor(promptnumber, t5_embedding.weight.size(1))
    #print(promptinitembedding)
    startindex = 0
    #print(promptinitembedding.shape)
    #print(t5_embedding.weight.shape)
    # print(tokenizer.get_vocab())
    # alllabel = ["name entity recognition", "person", "organization", "location", "mix",
    #             "sentence classification", "ag news", "world", "sports", "business", "science",
    #             "summarization", "cnn daily mail"]
    alllabel = alllabeltouse
    print(alllabel)
    for one in alllabel:
        encoderes = tokenizer.batch_encode_plus([one], padding=False, truncation=False, return_tensors="pt")
        touse = encoderes["input_ids"].squeeze()[:-1]
        # print(touse)
        # print(touse.shape)
        embeddingres = t5_embedding(touse).clone().detach()
        if embeddingres.shape[0] > 1:
            embeddingres = torch.mean(embeddingres, 0, keepdim=True)
        promptinitembedding[startindex] = embeddingres
        startindex += 1
        # print(embeddingres.shape)
    #print(promptinitembedding)
    # alltokens = {}
    fr = open('allnumber.pickle', 'rb')
    alltokens = pickle.load(fr)
    #print(len(alltokens))
    # print(alltokens)
    sortedalltoken = sorted(alltokens.items(), key=lambda item: item[1], reverse=True)
    # print(sortedalltoken)
    top5000 = []
    for one in sortedalltoken:
        if one[0] == 2:
            continue
        else:
            if len(top5000) < 5000:
                top5000.append(one)
            else:
                break
    #print(len(top5000))
    vocab = tokenizer.get_vocab()
    # print(vocab)
    # for one in top5000:
    #    print(one[0],"\t",one[1],"\t",tokenizer.convert_ids_to_tokens(one[0]))
    randomtokennum = promptnumber - len(alllabel)
    touse = random.sample(top5000, randomtokennum)
    print(touse)
    for one in touse:
        promptinitembedding[startindex] = t5_embedding.weight[one[0]].clone().detach()
        startindex += 1
    # print(startindex)
    # print(promptinitembedding)
    # print(t5_embedding.weight[2040])
    return promptinitembedding


def getonebatchresult(sen,target,preds):
    typedic = {"org": "ORG", "location": "LOC", "person": "PER", "mix": "MISC"}
    sennum = len(sen)
    restar = []
    respred = []
    for i in range(sennum):
        thissen, thistar, thispred = sen[i], target[i], preds[i]

        thissenlow = thissen.lower()

        sensplit = thissen.split(' ')
        sensplitlow = thissenlow.split(' ')

        tarres = ['O' for j in range(len(sensplit))]
        predres = ['O' for j in range(len(sensplit))]

        if thistar == 'end' and thispred == 'end':
            restar.append(tarres)
            respred.append(predres)
            continue

        if len(thistar) > 0 and thistar[-1] == ';':
            thistar = thistar[:-1]

        tarsplit1 = thistar.split(';')

        if thistar != 'end':
            for j in range(len(tarsplit1)):
                tarsplit2 = tarsplit1[j].split('!')
                if len(tarsplit2) != 2:
                    # logger.error('len error!')
                    # logger.error(sensplit)
                    # logger.error(thissen)
                    # logger.error(thistar)
                    # logger.error(tarsplit1)
                    # logger.error(tarsplit2)
                    # logger.error("-----------------------")
                    continue
                entity = tarsplit2[0].strip(' ')
                entitylow = entity.lower()
                type = tarsplit2[1].strip(' ')
                if type not in typedic:
                    # logger.error('dic error!')
                    # logger.error(type)
                    # logger.error(sensplit)
                    # logger.error(thistar)
                    # logger.error(tarsplit1)
                    # logger.error(tarsplit2)
                    # logger.error("-----------------------")
                    continue
                #if thissen.find(entity) == -1:
                if thissenlow.find(entitylow) == -1:
                    # logger.error('find error!')
                    # logger.error(entity)
                    # logger.error(sensplit)
                    # logger.error(thistar)
                    # logger.error(tarsplit1)
                    # logger.error(tarsplit2)
                    # logger.error("-----------------------")
                    continue
                trueindex = -100
                #entitysplit = entity.split(' ')
                entitysplit = entitylow.split(' ')
                for k in range(len(sensplit)):
                    #if sensplit[k] == entitysplit[0]:
                    if sensplitlow[k] == entitysplit[0] or entitysplit[0] in sensplitlow[k]:
                        iftrue = True
                        for l in range(1, len(entitysplit)):
                            if sensplitlow[k + l] != entitysplit[l] and (entitysplit[0] not in sensplitlow[k]):
                                iftrue = False
                                break
                        if iftrue:
                            trueindex = k
                            break
                if trueindex == -100:
                    # logger.error('error! Not find this entity')
                    # logger.error(entity)
                    # logger.error(sensplit)
                    # logger.error(thistar)
                    # logger.error(tarsplit1)
                    # logger.error(tarsplit2)
                    # logger.error("-----------------------")
                    continue
                for k in range(trueindex, trueindex + len(entitysplit)):
                    if k == trueindex:
                        tarres[k] = 'B-' + typedic[type]
                    else:
                        tarres[k] = 'I-' + typedic[type]

        if len(thispred) > 0 and thispred[-1] == ';':
            thispred = thispred[:-1]

        tarsplit3 = thispred.split(';')

        if thispred != "end":
            for j in range(len(tarsplit3)):
                tarsplit4 = tarsplit3[j].split('!')
                if len(tarsplit4) != 2:
                    # logger.error('len error!')
                    # logger.error(sensplit)
                    # logger.error(thispred)
                    # logger.error(tarsplit3)
                    # logger.error(tarsplit4)
                    # logger.error("**********************")
                    continue
                entity = tarsplit4[0].strip(' ')
                entitylow = entity.lower()
                type = tarsplit4[1].strip(' ')
                if type not in typedic:
                    # logger.error('dic error!')
                    # logger.error(type)
                    # logger.error(sensplit)
                    # logger.error(thispred)
                    # logger.error(tarsplit3)
                    # logger.error(tarsplit4)
                    # logger.error("**********************")
                    continue
                #if thissen.find(entity) == -1:
                if thissenlow.find(entitylow) == -1:
                    # logger.error('find error!')
                    # logger.error(entity)
                    # logger.error(sensplit)
                    # logger.error(thispred)
                    # logger.error(tarsplit3)
                    # logger.error(tarsplit4)
                    # logger.error("**********************")
                    continue
                trueindex = -100
                #entitysplit = entity.split(' ')
                entitysplit = entitylow.split(' ')
                for k in range(len(sensplit)):
                    #if sensplit[k] == entitysplit[0]:
                    if sensplitlow[k] == entitysplit[0] or entitysplit[0] in sensplitlow[k]:
                        iftrue = True
                        for l in range(1, len(entitysplit)):
                            #if sensplit[k + l] != entitysplit[l]:
                            if sensplitlow[k + l] != entitysplit[l] and (entitysplit[0] not in sensplitlow[k]):
                                iftrue = False
                                break
                        if iftrue:
                            trueindex = k
                            break
                if trueindex == -100:
                    # logger.error('error! Not find this entity')
                    # logger.error(entity)
                    # logger.error(sensplit)
                    # logger.error(thispred)
                    # logger.error(tarsplit3)
                    # logger.error(tarsplit4)
                    # logger.error("**********************")
                    continue
                else:
                    for k in range(trueindex, trueindex + len(entitysplit)):
                        if k == trueindex:
                            predres[k] = 'B-' + typedic[type]
                        else:
                            predres[k] = 'I-' + typedic[type]
        restar.append(tarres)
        respred.append(predres)
    return restar, respred




