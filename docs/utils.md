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


