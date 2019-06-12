import numpy as np
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))

# spherical k-means
from spherecluster import SphericalKMeans
skm = SphericalKMeans(n_clusters=2)

import re
pat = re.compile(r'^[0-9a-z]+\)?$')
stop = {':', '+', '-', '*', '/', '.', '(', ')', '）', '（', '的', '是', '本', '于', '与', '和', '即', '了', '在', '更', '上', '从', '或', '也', '很', '看', '这', '并', '而', '未', '所', '使', '地', '正', '有', '但', '着', '非常', '该', '正在', '为', '去', '都', '等', '及', '共', '篇', '一个', '闭'}

def distill(model, words, lang='en'):
    if lang == 'en':
        words = [x for x in words  if x not in stopWords and len(x) > 1]
    if lang == 'zh':
        words = [x for x in words if x not in stop and not pat.match(x) and x.find('研究') == -1 and x.find('发表') == -1]
    if len(words) <= 1: return words
    wv = np.array([model[w] for w in words])
    result = skm.fit(wv)
    labels = result.labels_
    # centers = result.cluster_centers_
    # centers can be used to train a classifier, but not for now
    lens = [[], []]
    for v, l in zip(words, labels):
        lens[l].append(len(v))
    if len(lens[0]) == 0 or len(lens[1]) == 0: return words
    pred = 0 if sum(lens[0]) / len(lens[0]) > sum(lens[1]) / len(lens[1]) else 1
    # print([v for v, l in zip(words, labels) if l==1-pred])
    return [v for v, l in zip(words, labels) if l==pred]
