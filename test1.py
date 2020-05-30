'''
仿照jupyter notebook的文本分类模型
简略框架
'''

import numpy as np
import csv
from string import punctuation
import re

import nltk
import ssl
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

def read_csv(path, encoding='utf-8-sig', headers=None, sep=',', dropna=True):
    with open(path, 'r', encoding=encoding) as csv_file:
        f = csv.reader(csv_file, delimiter=sep)
        start_idx = 0

        if headers is None:
            headers = next(f)
            # print(headers)
            start_idx += 1

        # ID,txt,Label
        sentences = []
        labels = []
        for line_idx, line in enumerate(f, start_idx):
            contents = line

            _dict = {}
            for header, content in zip(headers, contents):
                if str(header).lower() == "label":
                    labels.append(content)
                else:
                    _dict[header] = str(content).lower()    #小写
            sentences.append(_dict)

    return sentences, labels, headers

def preprocess_input(data, input_cols):
    texts = []
    all_words  = []
    # stopwords = get_stopwords()
    for line in data:
        #每行是一个字典
        for key in line:
            if key in input_cols:
                new_line = deletebr(str(line[key]))
                words = ''.join([c for c in new_line if c not in punctuation])
                texts.append(words)
                all_words.extend(words.split())
    return texts, all_words

def preprocess_labels(data):
    pass

def get_stopwords():
    stop_words = set(stopwords.words('english'))
    return stop_words

def deletebr(line):
    new_line = re.sub(r'<br\s*.?>', r'', line)
    return new_line

def printten(data):
    for i in range(10):
        print(data[i])

if __name__ == '__main__':
    ## Read csv data
    path = "data/train.csv"
    sentences, labels, headers = read_csv(path)
    labels = np.array(labels)

    ## Preprocess input
    ingore_cols = ['ID']
    input_cols = ['txt']

    texts, all_words = preprocess_input(sentences, input_cols)
    # printten(texts)
    # printten(all_words)
    # print(len(labels), type(labels))

    ## Removing outliers
    sentence_lens = Counter([len(x.split()) for x in texts])
    # print(sentence_lens)
    # print("Minimum review length: {}".format(min(sentence_lens)))
    # print("Maximum review length: {}".format(max(sentence_lens)))
    # 去除空字符串
    non_zero_idx = [ii for ii, review in enumerate(texts) if len(review.split()) != 0]
    texts = [texts[ii] for ii in non_zero_idx]
    labels = [labels[ii] for ii in non_zero_idx]

    # print(len(texts), len(labels))








