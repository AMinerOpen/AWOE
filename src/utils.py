from gensim.models import KeyedVectors
from src.tokenizer import tokenizer, zh_process_func, en_process_func
from src.distill import distill
import numpy as np
from spherecluster import SphericalKMeans
from collections import defaultdict

class Mono:
    def __init__(self, lang='en', model_path=None, model=None):
        assert lang in ['en', 'zh']
        print('loading model...')
        if model_path is None:
            model_path = 'tmp/keywords_aminer_{}'.format(lang)
        if model is None:
            self.model = KeyedVectors.load(model_path)
        else:
            self.model = model
        self.lang = lang
        if lang == 'zh':
            self.tok = tokenizer(set(self.model.wv.index2word), zh_process_func)
        elif lang == 'en':
            self.tok = tokenizer(set(self.model.wv.index2word), en_process_func)
        print('loaded.')
    def tokenize(self, text):
        return self.tok.tokenize(text)
    def filt_words(self, text):
        words = self.tokenize(text)
        return distill(self.model, words, lang=self.lang)
    def extract_keywords(self, text):
        words = self.filt_words(text)
        word2weight = defaultdict(float)
        for w in words:
            word2weight[w[0]] += w[2]
        return sorted(word2weight.items(), key=lambda x:x[1], reverse=True)
    def sent2vec(self, words):
        vec = np.zeros(self.model.wv.vector_size)
        for w in words:
            vec += self.model[w[0]]
        return vec
    def cluster(self, docs, k):
        vecs = []
        words = []
        cnt = 0
        for doc in docs:
            cnt += 1
            #print('processing doc {}'.format(cnt), end='\r')
            ws = self.extract_keywords(doc)
            words.append(ws)
            vecs.append(self.sent2vec(ws))
        print('processing doc {} over.'.format(cnt))
        skm = SphericalKMeans(n_clusters=k)
        result = skm.fit(np.array(vecs))
        return result.labels_, words

