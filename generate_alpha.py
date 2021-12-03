import xml.etree.ElementTree as ET
import pandas as pd
from collections import OrderedDict
import pickle
import random
import math


def generate_csv(file_name):
    tree = ET.ElementTree(file=file_name)
    root = tree.getroot()

    sentences = []
    poss = []
    targets = []
    targets_index_start = []
    targets_index_end = []
    lemmas = []

    for doc in root:
        for sent in doc:
            sentence = []
            pos = []
            target = []
            target_index_start = []
            target_index_end = []
            lemma = []
            for token in sent:
                assert token.tag == 'wf' or token.tag == 'instance'
                if token.tag == 'wf':
                    for i in token.text.split(' '):
                        sentence.append(i)
                        pos.append(token.attrib['pos'])
                        target.append('X')
                        lemma.append(token.attrib['lemma'])
                if token.tag == 'instance':
                    target_start = len(sentence)
                    for i in token.text.split(' '):
                        sentence.append(i)
                        pos.append(token.attrib['pos'])
                        target.append(token.attrib['id'])
                        lemma.append(token.attrib['lemma'])
                    target_end = len(sentence)
                    assert ' '.join(sentence[target_start:target_end]) == token.text
                    target_index_start.append(target_start)
                    target_index_end.append(target_end)
            sentences.append(sentence)
            poss.append(pos)
            targets.append(target)
            targets_index_start.append(target_index_start)
            targets_index_end.append(target_index_end)
            lemmas.append(lemma)

    gold_keys = []
    with open(file_name[:-len('.data.xml')] + '.gold.key.txt', "r", encoding="utf-8") as m:
        key = m.readline().strip().split()
        while key:
            gold_keys.append(key[1])
            key = m.readline().strip().split()

    output_file = file_name[:-len('.data.xml')] + '.csv'
    with open(output_file, "w", encoding="utf-8") as g:
        g.write('sentence\ttarget_index_start\ttarget_index_end\ttarget_id\ttarget_lemma\ttarget_pos\tsense_key\n')
        num = 0
        for i in range(len(sentences)):
            for j in range(len(targets_index_start[i])):
                sentence = ' '.join(sentences[i])
                target_start = targets_index_start[i][j]
                target_end = targets_index_end[i][j]
                target_id = targets[i][target_start]
                target_lemma = lemmas[i][target_start]
                target_pos = poss[i][target_start]
                sense_key = gold_keys[num]
                num += 1
                g.write('\t'.join(
                    (sentence, str(target_start), str(target_end), target_id, target_lemma, target_pos, sense_key)))
                g.write('\n')


def generate_auxiliary_sent_cls(gold_key_file_name, train_file_name, train_file_final_name, alpha=1):
    sense_data = pd.read_csv("./wordnet/index.sense.gloss", sep="\t", header=None).values
    print(len(sense_data))
    print(sense_data[1])

    d = dict()
    for i in range(len(sense_data)):
        s = sense_data[i][0]
        pos = s.find("%")
        try:
            d[s[:pos + 2]].append((sense_data[i][0], sense_data[i][-1]))
        except:
            d[s[:pos + 2]] = [(sense_data[i][0], sense_data[i][-1])]

    print(len(d))
    print(len(d["happy%3"]))
    print(d["happy%3"])
    print(len(d["happy%5"]))
    print(d["happy%5"])
    print(len(d["hard%3"]))
    print(d["hard%3"])

    train_data = pd.read_csv(train_file_name, sep="\t", na_filter=False).values
    print(len(train_data))
    print(train_data[0])

    gold_keys = []
    with open(gold_key_file_name, "r", encoding="utf-8") as f:
        s = f.readline().strip()
        while s:
            tmp = s.split()[1:]
            gold_keys.append(tmp)
            s = f.readline().strip()
    print(len(gold_keys))
    print(gold_keys[6])

    with open(train_file_final_name, "w", encoding="utf-8") as f:
        f.write('target_id\tlabel\tsentence\tgloss\tsense_key\n')
        for i in range(len(train_data)):
            assert train_data[i][-2] == "NOUN" or train_data[i][-2] == "VERB" or train_data[i][-2] == "ADJ" or \
                   train_data[i][-2] == "ADV"
            orig_sentence = train_data[i][0].split(' ')
            start_id = int(train_data[i][1])
            end_id = int(train_data[i][2])
            sentence = []
            for w in range(len(orig_sentence)):
                if w == start_id or w == end_id:
                    sentence.append('"')
                sentence.append(orig_sentence[w])
            sentence = ' '.join(sentence)
            orig_word = ' '.join(orig_sentence[start_id:end_id])

            for category in ["%1", "%2", "%3", "%4", "%5"]:
                word = train_data[i][-3]
                query = word + category
                gold_str = ""
                sample_list = list()
                try:
                    sents = d[query]
                    gold_key_exist = 0
                    for j in range(len(sents)):
                        str_ = ""
                        if sents[j][0] in gold_keys[i]:
                            str_ = train_data[i][3] + "\t" + "1"
                            gold_str = train_data[i][3] + "\t" + "1" + "\t" + train_data[i][0] + "\t" + sents[j][1] + "\t" + sents[j][0] + "\n"
                            gold_key_exist = 1
                        else:
                            str_ = train_data[i][3] + "\t" + "0"
                        str_ += "\t" + train_data[i][0] + "\t" + sents[j][1] + "\t" + sents[j][0] + "\n"
                        sample_list.append(str_)
                    assert gold_key_exist == 1
                except:
                    pass
                for _ in range(math.ceil(len(sample_list)**alpha)-1):
                    f.write(random.choice(sample_list))
                f.write(gold_str)



def generate_auxiliary_sent_cls_ws(gold_key_file_name, train_file_name, train_file_final_name, alpha=1):
    sense_data = pd.read_csv("./wordnet/index.sense.gloss", sep="\t", header=None).values
    print(len(sense_data))
    print(sense_data[1])

    d = dict()
    for i in range(len(sense_data)):
        s = sense_data[i][0]
        pos = s.find("%")
        try:
            d[s[:pos + 2]].append((sense_data[i][0], sense_data[i][-1]))
        except:
            d[s[:pos + 2]] = [(sense_data[i][0], sense_data[i][-1])]

    print(len(d))
    print(len(d["happy%3"]))
    print(d["happy%3"])
    print(len(d["happy%5"]))
    print(d["happy%5"])
    print(len(d["hard%3"]))
    print(d["hard%3"])

    train_data = pd.read_csv(train_file_name, sep="\t", na_filter=False).values
    print(len(train_data))
    print(train_data[0])

    gold_keys = []
    with open(gold_key_file_name, "r", encoding="utf-8") as f:
        s = f.readline().strip()
        while s:
            tmp = s.split()[1:]
            gold_keys.append(tmp)
            s = f.readline().strip()
    print(len(gold_keys))
    print(gold_keys[6])

    with open(train_file_final_name, "w", encoding="utf-8") as f:
        f.write('target_id\tlabel\tsentence\tgloss\tsense_key\n')
        for i in range(len(train_data)):
            assert train_data[i][-2] == "NOUN" or train_data[i][-2] == "VERB" or train_data[i][-2] == "ADJ" or \
                   train_data[i][-2] == "ADV"
            orig_sentence = train_data[i][0].split(' ')
            start_id = int(train_data[i][1])
            end_id = int(train_data[i][2])
            sentence = []
            for w in range(len(orig_sentence)):
                if w == start_id or w == end_id:
                    sentence.append('"')
                sentence.append(orig_sentence[w])
            if end_id == len(orig_sentence):
                sentence.append('"')
            sentence = ' '.join(sentence)
            orig_word = ' '.join(orig_sentence[start_id:end_id])

            for category in ["%1", "%2", "%3", "%4", "%5"]:
                word = train_data[i][-3]
                query = word + category
                gold_str = ""
                sample_list = list()
                try:
                    sents = d[query]
                    gold_key_exist = 0
                    for j in range(len(sents)):
                        str_ = ""
                        if sents[j][0] in gold_keys[i]:
                            str_ = train_data[i][3] + "\t" + "1"
                            gold_str = train_data[i][3] + "\t" + "1" + "\t" + sentence + "\t" + orig_word + " : " + sents[j][1] + "\t" + sents[j][0] + "\n"
                            gold_key_exist = 1
                        else:
                            str_ = train_data[i][3] + "\t" + "0"
                        str_ += "\t" + sentence + "\t" + orig_word + " : " + sents[j][1] + "\t" + sents[j][0] + "\n"
                        sample_list.append(str_)
                    assert gold_key_exist == 1
                except:
                    pass
                for _ in range(math.ceil(len(sample_list)**alpha)-1):
                    f.write(random.choice(sample_list))
                f.write(gold_str)
                # word = train_data[i][-3]
                # query = word + category
                # try:
                #     sents = d[query]
                #     gold_key_exist = 0
                #     for j in range(len(sents)):
                #         if sents[j][0] in gold_keys[i]:
                #             f.write(train_data[i][3] + "\t" + "1")
                #             gold_key_exist = 1
                #         else:
                #             f.write(train_data[i][3] + "\t" + "0")
                #         f.write("\t" + sentence + "\t" + orig_word + " : " + sents[j][1] + "\t" + sents[j][0] + "\n")
                #     assert gold_key_exist == 1
                # except:
                #     pass


def generate_auxiliary_token_cls(gold_key_file_name, train_file_name, train_file_final_name, alpha=1):

    sense_data = pd.read_csv("./wordnet/index.sense.gloss",sep="\t",header=None).values
    print(len(sense_data))
    print(sense_data[1])


    d = dict()
    for i in range(len(sense_data)):
        s = sense_data[i][0]
        pos = s.find("%")
        try:
            d[s[:pos + 2]].append((sense_data[i][0],sense_data[i][-1]))
        except:
            d[s[:pos + 2]]=[(sense_data[i][0], sense_data[i][-1])]

    print(len(d))
    print(len(d["happy%3"]))
    print(d["happy%3"])
    print(len(d["happy%5"]))
    print(d["happy%5"])
    print(len(d["hard%3"]))
    print(d["hard%3"])



    train_data = pd.read_csv(train_file_name,sep="\t",na_filter=False).values
    print(len(train_data))
    print(train_data[0])

    gold_keys=[]
    with open(gold_key_file_name,"r",encoding="utf-8") as f:
        s=f.readline().strip()
        while s:
            tmp = s.split()[1:]
            gold_keys.append(tmp)
            s=f.readline().strip()
    print(len(gold_keys))
    print(gold_keys[6])

    with open(train_file_final_name,"w",encoding="utf-8") as f:
        f.write('target_id\tlabel\tsentence\tgloss\ttarget_index_start\ttarget_index_end\tsense_key\n')
        for i in range(len(train_data)):
            assert train_data[i][-2]=="NOUN" or train_data[i][-2]=="VERB" or train_data[i][-2]=="ADJ" or train_data[i][-2]=="ADV"

            for category in ["%1", "%2", "%3", "%4", "%5"]:
                word = train_data[i][-3]
                query = word + category
                gold_str = ""
                sample_list = list()
                try:
                    sents = d[query]
                    gold_key_exist = 0
                    for j in range(len(sents)):
                        str_ = ""
                        if sents[j][0] in gold_keys[i]:
                            str_ = train_data[i][3] + "\t" + "1"
                            gold_str = train_data[i][3] + "\t" + "1" + "\t" + train_data[i][0]+"\t"+sents[j][1]+"\t"+str(train_data[i][1])+"\t"+str(train_data[i][2])+"\t"+sents[j][0]+"\n"
                            gold_key_exist = 1
                        else:
                            str_ = train_data[i][3] + "\t" + "0"
                        str_ += "\t"+train_data[i][0]+"\t"+sents[j][1]+"\t"+str(train_data[i][1])+"\t"+str(train_data[i][2])+"\t"+sents[j][0]+"\n"
                        sample_list.append(str_)
                    assert gold_key_exist == 1
                except:
                    pass
                for _ in range(math.ceil(len(sample_list)**alpha)-1):
                    f.write(random.choice(sample_list))
                f.write(gold_str)
                # word = train_data[i][-3]
                # query = word+category
                # try:
                #     sents = d[query]
                #     gold_key_exist = 0
                #     for j in range(len(sents)):
                #         if sents[j][0] in gold_keys[i]:
                #             f.write(train_data[i][3]+"\t"+"1")
                #             gold_key_exist = 1
                #         else:
                #             f.write(train_data[i][3]+"\t"+"0")
                #         f.write("\t"+train_data[i][0]+"\t"+sents[j][1]+"\t"+str(train_data[i][1])+"\t"+str(train_data[i][2])+"\t"+sents[j][0]+"\n")
                #     assert gold_key_exist == 1
                # except:
                #     pass

if __name__ == "__main__":
    alpha = 0.8
    eval_dataset = ['senseval3', 'semeval2007', 'semeval2013', 'semeval2015']
    # train_dataset = ['senseval2', 'senseval2+OMSTI']
    train_dataset = ['senseval2']

    file_name = []
    for dataset in eval_dataset:
        file_name.append('./dataset/' + dataset + '/' + dataset + '.data.xml')
    for dataset in train_dataset:
        file_name.append('./dataset/' + dataset + '/' + dataset.lower() + '.data.xml')

    for file in file_name:
        print(file)
        generate_csv(file)

    file_path = []
    for dataset in eval_dataset:
        file_path.append('./dataset/' + dataset + '/' + dataset)
    for dataset in train_dataset:
        file_path.append('./dataset/' + dataset + '/' + dataset.lower())

    for file_name in file_path:
        gold_key_file_name = file_name + '.gold.key.txt'
        train_file_name = file_name + '.csv'
        if file_name == './dataset/senseval2/senseval2':
            train_file_final_name_sent = file_name + '_train_sent_cls.csv'
            train_file_final_name_sent_ws = file_name + '_train_sent_cls_ws.csv'
            train_file_final_name_token = file_name + '_train_token_cls.csv'

        else:
            train_file_final_name_sent = file_name + '_test_sent_cls.csv'
            train_file_final_name_sent_ws = file_name + '_test_sent_cls_ws.csv'
            train_file_final_name_token = file_name + '_test_token_cls.csv'

        print(gold_key_file_name)
        print(train_file_name)
        print(train_file_final_name_sent)
        print(train_file_final_name_sent_ws)
        print(train_file_final_name_token)
        generate_auxiliary_sent_cls(gold_key_file_name, train_file_name, train_file_final_name_sent, alpha=alpha)
        generate_auxiliary_sent_cls_ws(gold_key_file_name, train_file_name, train_file_final_name_sent_ws, alpha=alpha)
        generate_auxiliary_token_cls(gold_key_file_name, train_file_name, train_file_final_name_token, alpha=alpha)

    sense2index = OrderedDict()
    index2sense = OrderedDict()
    lemma2sense_dict = OrderedDict()
    lemma2index_dict = OrderedDict()

    with open("./wordnet/index.sense", "r", encoding='utf-8') as f:
        s = f.readline().strip().split()
        while s:
            sense2index[s[0]] = len(sense2index)
            index2sense[len(index2sense)] = s[0]
            pos = s[0].find("%")
            lemma = s[0][:pos]
            try:
                lemma2sense_dict[lemma].append(s[0])
            except:
                lemma2sense_dict[lemma] = [s[0]]

            try:
                lemma2index_dict[lemma].append(sense2index[s[0]])
            except:
                lemma2index_dict[lemma] = [sense2index[s[0]]]

            s = f.readline().strip().split()

    with open('./wordnet/sense2index.pkl', 'wb') as g, open('./wordnet/index2sense.pkl', 'wb') as h, \
            open('./wordnet/lemma2sense_dict.pkl', 'wb') as j, open('./wordnet/lemma2index_dict.pkl', 'wb') as k:
        pickle.dump(sense2index, g)
        pickle.dump(index2sense, h)
        pickle.dump(lemma2sense_dict, j)
        pickle.dump(lemma2index_dict, k)
