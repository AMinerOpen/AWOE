# AWOE

**A**cademic **WO**rd **E**mbeddings based on [AMiner](https://www.aminer.cn) 2 billion publication data and [gensim](https://radimrehurek.com/gensim/index.html) and their applications.

# Dependencies

* Python 3
* gensim
* spherecluster

# Overview

## Pre-trained Models

* English Paper Keywords (EPK): [Download](https://lfs.aminer.cn/misc/awoe/keywords_aminer_en.zip)
* Chinese Paper Keywords (CPK): [Download](https://lfs.aminer.cn/misc/awoe/keywords_aminer_zh.zip)
* Bilingual Transformation Matrix: [Download](https://lfs.aminer.cn/misc/awoe/W_en2zh.pkl)

For details for these models, see [docs/word2vec.md](docs/word2vec.md). (If you just want to use these models, ignore them.)

We hvae prepared a download bash script for you, you can use it on your need. For example, if you only need Chinese, just run ```./download.sh zh```.

```bash
chmod +x download.sh
./download.sh zh
./download.sh en
wget https://lfs.aminer.cn/misc/awoe/W_en2zh.pkl -P tmp/
```

## Utils

We provide some utils to use the above models, including tokenization, keyword extraction, sentense to vector, etc. Here are some use examples.

Before using these modules, download the required models first.

### Mono-lingual

Docs to complete. You can run ```test.py``` for now.

### Bi-lingual

Docs to complete.

# Citation

If our work helps you in some way, please consider citing the following publication(s):

* Jie Tang, Jing Zhang, Limin Yao, Juanzi Li, Li Zhang, and Zhong Su. ArnetMiner: Extraction and Mining of Academic Social Networks. In Proceedings of the Fourteenth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (SIGKDD’2008).
