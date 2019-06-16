# AWOE

**A**cademic **WO**rd **E**mbedding based on [AMiner](https://www.aminer.cn) 2 billion publication data and [gensim](https://radimrehurek.com/gensim/index.html).

# Monolingual

## English Paper Keywords (EPK)

* Corpus: 120 million English paper keywords
* Pre-processing: lower cased and striped
* Parameters: Word2Vec(sg=1, min\_count=10, size=200, window=10, iter=10)
* Vocabulary size: 1966374

[Download](https://lfs.aminer.cn/misc/awoe/keywords_aminer_en.zip) model and unzip it to a path.

```python
from gensim.models import KeyedVectors

model_en = KeyedVectors.load("path/to/model")

print(model_en["data mining"])
print(model_en.most_similar("data mining"))
'''
[-0.03671319 -0.43231151 -0.53570551 -0.15723452  0.0011634  -0.02426389
 -0.18189366  0.05359208 -0.35148171  0.09248555  0.07982133 -0.32470939
 -0.1510863  -0.26247889 ... 0.16393639  0.23222639]
[('algorithm design and analysis', 0.7523155212402344), ('classification algorithms', 0.6936554908752441), ('data models', 0.6934263110160828), ('knowledge discovery', 0.6912256479263306), ('information analysis', 0.6770250201225281), ('computational modeling', 0.6744495630264282), ('association rules', 0.6620141267776489), ('commodities interflow', 0.6523971557617188), ('frequent closed itemset mining', 0.6514651775360107), ('learning artificial intelligence', 0.6466258764266968)]
'''
```

## Chinese Paper Keywords (CPK)

* Corpus: 73 million Chinese paper keywords
* Pre-processing: lower cased and striped
* Parameters: Word2Vec(sg=1, min\_count=10, size=200, window=10, iter=10)
* Vocabulary size: 1544192

[Download](https://lfs.aminer.cn/misc/awoe/keywords_aminer_zh.zip) model and unzip it to a path.

```python
from gensim.models import KeyedVectors

model_zh = KeyedVectors.load("path/to/model")

print(model_zh["数据挖掘"])
print(model_zh.most_similar("数据挖掘"))
'''
[  4.90961999e-01   2.09306851e-01  -4.08969492e-01   1.44380927e-01
  -1.43410265e-02   6.50532782e-01  -1.76238433e-01   1.68822825e-01
   6.65765345e-01 ... -2.03358024e-01  -6.11832917e-01]
[('关联规则', 0.9038501381874084), ('apriori算法', 0.8566242456436157), ('知识发现', 0.8078784346580505), ('频繁项集', 0.794908881187439), ('数据仓库', 0.7892417907714844), ('apriori', 0.7818102240562439), ('联机分析处理', 0.7706191539764404), ('fp-growth', 0.7693448066711426), ('数据挖掘技术', 0.7669618129730225), ('挖掘算法', 0.7620916366577148)]
'''
```

# Bilingual

## Transformation Matrix

We train a transformation matrix between EPK and CPK by Procrustes Method demonstrated in [1](http://arxiv.org/abs/1702.03859). We select the most frequent 500k keywords, and use Google Translate API to translate them. If a Chinese keyword can translate into English and translate back into the same Chinese keyword, we choose it as a supervision seed. In this way, we get 69405 seeds as supervision and use 65000 to train and 4405 to test. The P@1 is 18.34 (not bad for selecting nn among 500k words).

[Download](https://lfs.aminer.cn/misc/awoe/W_en2zh.pkl) the matrix and move it to a path.

```python
import pickle

with open("path/to/W_en2zh.pkl", "rb") as f:
    W = pickle.load(f)

print(model_zh.similar_by_vector(model_en["data mining"].dot(W)))
'''
[('数据挖掘', 0.621910572052002), ('apriori', 0.598727822303772), ('map reduce', 0.5853731036186218), ('频繁集', 0.5818564891815186), ('k-means', 0.5817596912384033), ('数据库知识发现', 0.5783699750900269), ('机器学习', 0.5778008103370667), ('知识发现', 0.5774385929107666), ('sliq', 0.5769015550613403), ('相似性搜索', 0.5721484422683716)]
'''
# Since Procrustes restrict W.dot(W.T) = I, we can also reversely use it.
print(model_en.similar_by_vector(model_zh["数据挖掘"].dot(W.T)))
'''
[('knowledge discovery', 0.6947450637817383), ('knowledge discovery in databases (kdd)', 0.6842040419578552), ('data miming', 0.6666584610939026), ('commodities interflow', 0.6620808243751526), ('data preparation technique', 0.6564021706581116), ('fp_growth algorithm', 0.6534382104873657), ('mining classification rules', 0.6477227210998535), ('association rule induction', 0.6470234394073486), ('apriori association rule mining', 0.6454548835754395), ('click-stream', 0.644127607345581)]
'''
```

# Tokenizer

Our keyword model can also been used for tokenizing academic text.

```python
from src.tokenizer import tokenizer, zh_process_func, en_process_func

tok_zh = tokenizer(set(model_zh.wv.index2word), zh_process_func)
tok_en = tokenizer(set(model_en.wv.index2word), en_process_func)

print(tok_zh.tokenize("机器翻译是自然语言处理的一个子领域。", max_len=None))
print(tok_en.tokenize("Machine translation is a sub-domain of natural language processing.", max_len=None)) # Maybe you need nltk.download('punkt') if failed
'''
[('机器翻译', 0), ('是', 4), ('自然语言处理', 5), ('的', 11), ('一个', 12), ('子领域', 14)]
[('machine translation', 0), ('is a', 20), ('sub-domain', 25), ('of', 36), ('natural language processing', 39), ('.', 67)]
'''
```
Since our tokenizer will do some pre-processing and omit the out-of-vocabulary words, the original index of the word is also preserved. Besides, If you need a more efficient way to tokenize, modify ```tokenize``` to ```tokenize_greedy``` is also ok.

As you see, there are some noise in the extracted words, so we also provide a function to filter the result. It utilizes [spherecluster](https://github.com/jasonlaska/spherecluster) package to cluster all the words into two clusters, and predict the one with longer average length as a better group.

```python
from src.distill import distill

words = tok_zh.tokenize("机器翻译是自然语言处理的一个子领域。")
print(distill(model_zh, words, lang='zh'))
words = tok_en.tokenize("Machine translation is a sub-domain of natural language processing.")
print(distill(model_en, words, lang='en'))
'''
[('机器翻译', 0, 0.9248400926589966), ('自然语言处理', 5, 0.9248400926589966)]
[('machine translation', 0, 0.9285963177680969), ('natural language processing', 39, 0.9285963177680969)]
'''
```

We also allocate a weight value for each word, which is the cosine similarity between the word and the center of cluster.

# Applications

* [XTool](https://lab.aminer.cn/xtool/)
* APIs
    * https://innovaapi.aminer.cn/keyword/?label=clustering&lang=en
    * https://innovaapi.aminer.cn/keyword/?label=聚类&lang=cn

# Citation

If our work helps you in some way, please consider citing the following publication(s):

* Jie Tang, Jing Zhang, Limin Yao, Juanzi Li, Li Zhang, and Zhong Su. ArnetMiner: Extraction and Mining of Academic Social Networks. In Proceedings of the Fourteenth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (SIGKDD’2008).
