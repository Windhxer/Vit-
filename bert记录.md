### BERT记录

BERT——Bidirectional Encoder Representation from Transformers，双向Transformer的Encoder，因为decoder是不能获取要预测的信息的。

#### 1.模型结构

![img](https://pic1.zhimg.com/80/v2-d942b566bde7c44704b7d03a1b596c0c_1440w.webp)

- 对比OpenAI GPT(Generative pre-trained transformer)：BERT是双向的Transformer block连接，就像单向RNN和双向RNN的区别，直觉上来讲效果会好一些。

- 对比ELMo：虽然都是“双向”，但目标函数其实是不同的。ELMo是分别以P(wi|w1,...,wi-1)和 P(wi|wi+1,...,wn)作为目标函数，独立训练处两个representation然后拼接，而BERT则是以 P(wi|w1,...,wi-1,wi+1,...,wn)作为目标函数训练LM。

#### 2.区别于transformer的 **Embedding**

![img](https://pic2.zhimg.com/80/v2-11505b394299037e999d12997e9d1789_1440w.webp)

其中：

- Token Embeddings是词向量，第一个单词是CLS标志，可以用于之后的分类任务。句子之间使用特殊token[SEP]进行分割。形状(1,n, 768)。
- Segment Embeddings用来做句子序列的标号，帮助BERT区分成对的输入序列，也更方便进行二分类任务。形状(1,n, 768)。
- Position Embeddings和之前文章中的Transformer不一样，不是三角函数而是学习出来的。形状(1,n, 768)。让BERT知道其输入具有时间属性。

```python
    word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
    for i, w in enumerate(word_list):
        word_dict[w] = i + 4
    number_dict = {i: w for i, w in enumerate(word_dict)}
    vocab_size = len(word_dict)
```

如上即为token的处理过程。

#### 3.**Pre-training Task**

第一步预训练的目标就是做语言模型，从上文模型结构中看到了这个模型的不同，即bidirectional。**关于为什么要如此的bidirectional**，作者在[reddit](https://link.zhihu.com/?target=http%3A//www.reddit.com/r/MachineLearning/comments/9nfqxz/r_bert_pretraining_of_deep_bidirectional/)上做了解释，意思就是如果使用预训练模型处理其他任务，那人们想要的肯定不止某个词左边的信息，而是左右两边的信息。而考虑到这点的模型ELMo只是将left-to-right和right-to-left分别训练拼接起来。觉上来讲我们其实想要一个deeply bidirectional的模型，但是普通的LM又无法做到，因为在训练时可能会“穿越”。作者用了一个加mask的trick。

```python
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:## 取其中的三个；masked_pos=[6, 5, 17] 注意这里对应的是position信息；masked_tokens=[13, 9, 16] 注意这里是被mask的元素之前对应的原始单字数字；
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:  # 80%
                input_ids[pos] = word_dict['[MASK]'] # make mask
            elif random() < 0.5:  # 10%
                index = randint(0, vocab_size - 1) # random index in vocabulary
                input_ids[pos] = word_dict[number_dict[index]] # replace
```

在训练过程中作者随机mask 15%的token，而不是把像cbow一样把每个词都预测一遍。**最终的损失函数只计算被mask掉那个token。**

Mask如何做也是有技巧的，如果一直用标记[MASK]代替（在实际预测时是碰不到这个标记的）会影响模型，所以随机mask的时候10%的单词会被替代成其他单词，10%的单词不替换，剩下80%才被替换为[MASK]。

因为涉及到QA和NLI之类的任务，增加了第二个预训练任务，目的是让模型理解两个句子之间的联系。训练的输入是句子A和B，B有一半的几率是A的下一句，输入这两个句子，模型预测B是不是A的下一句。预训练的时候可以达到97-98%的准确度。**作者特意说了语料的选取很关键，要选用document-level的而不是sentence-level的，这样可以具备抽象连续长序列特征的能力。**

#### 4.**Fine-tunning**

分类：对于sequence-level的分类任务，BERT直接取第一个[CLS]token的final hidden state加一层权重后softmax预测label proba。其他预测任务需要进行一些调整。

![img](https://pic2.zhimg.com/80/v2-b054e303cdafa0ce41ad761d5d0314e1_720w.webp)

可以调整的参数和取值范围有：

- Batch size: 16, 32
- Learning rate (Adam): 5e-5, 3e-5, 2e-5
- Number of epochs: 3, 4

因为大部分参数都和预训练时一样，精调会快一些，所以作者推荐多试一些参数。