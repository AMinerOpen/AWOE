import numpy as np
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))

# spherical k-means
from spherecluster import SphericalKMeans
skm = SphericalKMeans(n_clusters=2)

def distill(model, words, lang='en'):
    if lang == 'en':
        words = [x for x in words  if x not in stopWords and len(x) > 1]
    wv = np.array([model[w] for w in words])
    result = skm.fit(wv)
    labels = result.labels_
    # centers = result.cluster_centers_
    # centers can be used to train a classifier, but not for now
    lens = [[], []]
    for v, l in zip(words, labels):
        lens[l].append(len(v))
    pred = 0 if sum(lens[0]) / len(lens[0]) > sum(lens[1]) / len(lens[1]) else 1
    return [v for v, l in zip(words, labels) if l==pred]
