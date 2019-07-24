import numpy as np
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))

# spherical k-means
from spherecluster import SphericalKMeans
skm = SphericalKMeans(n_clusters=2)

import re
pat = re.compile(r'^\(?[0-9a-z]+\)?$')
stop = {':', '+', '-', '*', '/', '.', '（', '）', '（', '的', '是', '本', '于', '与', '和', '即', '了', '在', '更', '上', '从', '或', '也', '很', '看', '这', '并', '而', '未', '所', '使', '地', '正', '有', '但', '着', '非常', '该', '正在', '为', '去', '都', '等', '及', '共', '篇', '将', '中', '一个', '闭'}

def is_stop(x):
    for i in x[0].split(' '):
        if i not in stopWords:
            return False
    return True

import numpy as np
def cosine(x, y):
    if x.dot(x) == 0 or y.dot(y) == 0: return -1
    return float(np.dot(x, y) / np.linalg.norm(x) / np.linalg.norm(y))

def distill(model, words, lang='en'):
    if lang == 'en':
        words = [x for x in words  if not is_stop(x) and len(x[0]) > 1]
    if lang == 'zh':
        words = [x for x in words if x[0] not in stop and not pat.match(x[0]) and x[0].find('研究') == -1 and x[0].find('发表') == -1]
    if len(words) == 0: return words
    if len(words) == 1: return [(words[0][0], words[0][1], 1.0)]
    wv = np.array([model[w[0]] for w in words])
    result = skm.fit(wv)
    labels = result.labels_
    centers = result.cluster_centers_
    # centers can be used to train a classifier, but not for now
    lens = [[], []]
    for v, l in zip(words, labels):
        lens[l].append(len(v[0]))
    if len(lens[0]) == 0:
        pred = 1
    elif len(lens[1]) == 0:
        pred = 0
    else:
        pred = 0 if sum(lens[0]) / len(lens[0]) > sum(lens[1]) / len(lens[1]) else 1
    # print([v for v, l in zip(words, labels) if l==1-pred])
    return [(v[0], v[1], cosine(vec, centers[l])) for vec, v, l in zip(wv, words, labels) if l==pred]
