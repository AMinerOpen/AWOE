import jieba
from nltk.tokenize import word_tokenize

class zh_process_func:
    def split_func(x):
        return jieba.lcut(x) # list(x)
    def merge_func(x):
        return ''.join(x)
    def pre_process(x):
        return x.strip().lower()

class en_process_func:
    def split_func(x):
        return word_tokenize(x)
    def merge_func(x):
        return ' '.join(x)
    def pre_process(x):
        return x.strip().lower()

class tokenizer:
    def __init__(self, dic, process_func):
        self.dic = dic
        self.split_func = process_func.split_func
        self.merge_func = process_func.merge_func
        self.pre_process = process_func.pre_process
        self.max_len = max([len(x) for x in self.dic])

    def choose(self, universe, parts, start, end):
        if start < 0 or end < 0 or start >= len(universe) or end >= len(universe) \
                or not parts:
            return []
        c = {len(word): (word, i, j, k) for word, i, j, k in parts}
        w, i, j, k = c[max(c.keys())]
        partial_parts_left = list(filter(lambda x: x[1] < i and x[2] < i, parts))
        partial_parts_right = list(filter(lambda x: x[1] > j and x[2] > j, parts))
        partial_res_left = self.choose(universe, partial_parts_left, start, i - 1)
        partial_res_right = self.choose(universe, partial_parts_right, j + 1, end)
        return partial_res_left + [(w, k)] + partial_res_right

    def tokenize(self, text, max_len=None):
        '''Globally longest word first'''
        if max_len is None:
            max_len = self.max_len
        text = self.pre_process(text)
        text = self.split_func(text)
        inds = []
        now = ""
        for w in text:
            inds.append(len(now))
            now = self.merge_func([now, w])
        words = []
        parts = []
        for i in range(len(text) + 1):
            for j in range(i + 1, min(len(text) + 1, i + max_len)):
                w = self.merge_func(text[i: j])
                if w in self.dic:
                    parts.append((w, i, j - 1, inds[i]))
        words += self.choose(text, parts, 0, len(text) - 1)
        return words

    def tokenize_greedy(self, text, max_len=None, oov=False):
        '''Locally longest word first, also good when doing from backward to frontward'''
        if max_len is None:
            max_len = self.max_len
        text = self.pre_process(text)
        text = self.split_func(text)
        inds = []
        now = ""
        for w in text:
            inds.append(len(now))
            now = self.merge_func([now, w])
        text = text[::-1]
        inds = inds[::-1]
        words = []
        L = len(text)
        i = 0
        while i < L:
            flag = -1
            for j in range(min(L, i + max_len), i, -1):
                if self.merge_func(text[i:j][::-1]) in self.dic:
                    flag = j
                    break
            if flag != -1:
                words.append((self.merge_func(text[i:flag][::-1]), inds[flag-1]))
                i = flag
            else:
                if oov: words.append((text[i], inds[i]))
                i += 1
        return words[::-1]

if __name__ == '__main__':
    zh_tokenizer = tokenizer({'机器翻译', '自然语言处理'}, zh_process_func)
    print(zh_tokenizer.tokenize_greedy('机器翻译是自然语言处理的一个子领域', oov=True))
    en_tokenizer = tokenizer({'machine translation', 'natural language processing'}, en_process_func)
    print(en_tokenizer.tokenize_greedy('Machine translation is a sub-domain of natural language processing.'))
