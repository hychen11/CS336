https://qa9vavmvcb6.feishu.cn/docx/U48Odz8tEovBq0xlaOKck1KknBh

# Tokenizer

### 分词方法划分

本质：围绕 1.语义信息量（信息是否丢失）2.词表大小（计算与内存）3.oov问题（未知词）

问到分词方法的优缺点、词表大小，围绕这三点答即可划分为三种词粒度：

a. Word-based Tokenizer 单词级：存在13问题

b.Character-based Tokenizer 字符级：存在12问题

c. Subword-based Tokenizer 子词级：兼容123以下方法均为子词级的分词方法

#### Byte Pair Encoding (BPE)

https://zhuanlan.zhihu.com/p/448147465

核心思想在于数据压缩

* 词频统计
* 词表合并

Basic idea: train the tokenizer on raw text to automatically determine the vocabulary.

```python
# 创建一个分词器实例
tokenizer = BPETokenizer(params)

# 原始文本
string = "the quick brown fox"  # @inspect string

# 对文本进行编码，得到索引序列
indices = tokenizer.encode(string)  # @inspect indices

# 将索引序列解码回文本
reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string

# 验证解码后的字符串是否与原始字符串一致
assert string == reconstructed_string
```

BPE 首先将词分成单个字符，然后依次用另一个字符替换频率最高的**一对字符** ，直到循环次数结束。

BPE 的优点就在于，可以很有效地平衡词典大小和编码步骤数

随着合并的次数增加，词表大小通常先增加后减小。迭代次数太小，大部分还是字母，没什么意义；迭代次数多，又重新变回了原来那几个词。所以词表大小要取一个中间值。

![img](https://pic1.zhimg.com/v2-18c257daa993fd940d2b42744a50ed90_1440w.jpg)

对于同一个句子, 可能会有不同的 Subword 序列。不同的 Subword 序列会产生完全不同的 id 序列表示，这种歧义可能在解码阶段无法解决。在翻译任务中，不同的 id 序列可能翻译出不同的句子，这显然是错误的。

BPE 的贪心算法无法对随机分布进行学习

```python
import re, collections


def get_vacab(filename):
    vocab = collections.defaultdict(int)
    with open(filename, 'r', encoding='utf-8') as fhand:
        for line in fhand:
            words = line.strip().split()  # 这里strip去除首位空格
            for word in words:
                vocab[' '.join(list(list(word)) + '</w>')] += 1
    return vocab


def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


def get_tokens(vocab):
    tokens = collections.defaultdict(int)
    for word, freq in vocab.items():
        word_tokens = word.split()
        for token in word_tokens:
            tokens[token] += freq
    return tokens

```

#### BBPE算法

从字符提升到字节

merge前，文本编码成字节序列

>  如何理解？就是相当于一个字符是"你"， UTF-8编码成3个0-255的字节，初始词表就255大小

#### WordPiece 算法， 基于似然函数

找到一套子词，使用这套子词表示整个训练文本时，其可能性最大

除第一个字母，添加#前缀，word->[w,##o,##r,##d]

$score(A,B)= \frac{P(AB)}{P(A)\times{P(B)}}$

###### 为什么使用似然函数的思想？

> 1. **初始状态：**
>     WordPiece 算法开始时，有一个非常小的词汇表，通常只包含单个字符（比如 `'a'`, `'b'`, `'c'`, ...）。
> 2. **贪婪合并过程：**
>     在每一步迭代中，WordPiece 会遍历所有的 **可能的子词对**（例如 `'a'` 和 `'b'` 组成 `'ab'`；`'e'` 和 `'d'` 组成 `'ed'`；等等），评估如果将它们合并成一个新子词并添加到词汇表中，会带来多大的 **似然提升**。
> 3. **似然提升的计算：**
>     这个“似然提升”正是衡量了将现有文本（训练数据）用新的、**更大的子词词汇表示**编码时，其联合概率（或者说“似然”）会增加多少。
>
> ------
>
> 例如，假设原始文本中有很多次 `"walking"` 这个词：
>
> - **在只有字符的词汇表下：**
>    `"walking"` 可能被编码为 `"w"`, `"a"`, `"l"`, `"k"`, `"i"`, `"n"`, `"g"`。
> - **如果我们发现合并 `"walk"` 和 `"##ing"` 能带来巨大提升：**
>    这意味着在训练数据中，`"walk"` 和 `"ing"` 经常一起出现，并且作为独立单元（子词）比拆分成单个字符能更好地解释文本。
>
> ------
>
> 如果我的词汇表包含这个新合并的子词，那么用这个词汇表来表示训练文本，文本的整体“可能性”会更大。这种可能性（似然值）越高，说明该模型在给定参数（词表）下更符合观察数据。

#### Unigram 拆分的似然函数

Expectation-Maximization算法训练

假设句子中每个子词都是独立生成的，一个句子，找到一种拆分方式是的所有拆分出来的子词联合概率达到最大

**迭代删掉最不重要的 token**：

- 先准备一个超大的候选词表（可能几十万子词）。
- 在训练中，逐步删除 **低概率、对整体似然提升不大**的 token。
- 这样剩下的 token 就是：
  - 既能高效覆盖语料
  - 又不会太多冗余

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
sen = "hello world tokenizer test wordpiece"
inputs = tokenizer(sen, padding="max_length", max_length=15)
print(inputs)
```

# Embedding

one-hot编码，非常稀疏，大部分是0，无法捕获词语之间的语义关系

映射到连续低纬的向量空间，可以捕捉词语之间的语义关系，减少向量维度

## 静态编码

### Word2Vec

训练模型本质上是只具有一个隐含层的神经元网络，就是相当于预训练了一个权重Q，可以选择微调Q或者冻结Q

Input是一个one-hot编码

#### CBOW Continuous Bag of Words

**输入**：上下文词（context words），比如一个词的左右各 2 个词。

**输出**：目标词（target word）。

**直观理解**：看周围的词，猜中间的词。



使用围绕目标单词的其他单词（语境）作为输入，在映射层做加权处理后输出目标单词

<img src="https://picx.zhimg.com/v2-8fcd03fa3dad0cf4d0af1a890ace5193_1440w.jpg" alt="img" style="zoom:33%;" />

#### SG Skip-gram

**输入**：目标词（center word）。

**输出**：上下文词（context words）。

**直观理解**：看一个词，猜它周围的词。

根据当前单词预测语境，罕见词处理好，无法处理一词多义，苹果手机等

“There is an apple on the table”作为训练数据，CBOW的输入为（is,an,on,the），输出为apple。而Skip-gram的输入为apple，输出为（is,an,on,the）

<img src="https://pic4.zhimg.com/v2-a04dca66f5e8456f50b4b43fb87b98dd_1440w.jpg" alt="img" style="zoom:33%;" />

![img](https://picx.zhimg.com/v2-0d3bbbe2ab92b36d40ff0acb9170f311_1440w.jpg)

NNLM 侧重于预测下一个单词，而Word2Vec侧重点在于得到词向量后反向传播更新权重矩阵Q

### FastText

FastText 可以看作是 Word2Vec 的改进版，它主要解决了 Word2Vec 对未登录词（OOV, Out-Of-Vocabulary）和词形变化敏感的问题

为什么要FastText呢？因为word2vec无法处理**OOV（out-of-vocabulary）**的问题，并且变化多端语言无法有效捕捉它们之间共享的含义。而实际存储FastText可以只存n-gram向量，节省存储空间

也就是FastText的优点就是

**对词形变化更鲁棒**：

- “play”, “playing”, “player” 会共享 “play” 这个子词。

**能处理未登录词（OOV）**：

- 新词可以通过它的子词来组合向量，不会完全“未知”。

#### 例子

n-gram，一个词还可以拆分成更小的字符片段

把词拆成字符 n-grams，比如 n=3（trigram）。

- “apple” → `<ap`, `app`, `ppl`, `ple`, `le>`（加上词首 `<` 和词尾 `>` 标记）。

* 每个 n-gram 学一个 embedding。

* 一个词的 embedding = 它所有 n-grams embedding 的和（或平均）。

* 用这个 embedding 训练 Skip-Gram / CBOW。

## 动态编码

静态编码一个词只能学出一个词向量，但是不同上下文可能有不同含义，需要动态编码

训练无监督，下有任务有监督

### ELMo

**ELMo的核心思想是：一个词的准确含义，取决于它所在的上下文语境。** 它解决了传统词向量（如 Word2Vec、GloVe）最大的一个痛点——**“一词多义”** 问题。

**1. 预训练一个双向语言模型**

- ELMo使用一个**深度双向LSTM**网络，在一个大型语料库（如维基百科）上进行训练。
- 这个训练的目标很简单：根据上下文预测下一个词。但它的巧妙之处在于“双向”：
  - **前向LSTM**：从左到右阅读句子，根据前面的词预测当前词。
  - **后向LSTM**：从右到左阅读句子，根据后面的词预测当前词。
- 预训练结束后，我们得到的不是一个静态的“词向量表”，而是一个**训练好的双向LSTM模型**。

**2. 提取多层特征**

- 当一个新句子输入到这个预训练好的模型中时，模型会为句子中的每个词生成一系列的特征表示。
- **关键点**：ELMo不仅仅使用LSTM最后一层的输出。它利用了**所有层**的隐藏状态：
  - **底层（靠近输入的层）**：通常捕捉到的是词法、语法等基础特征（例如，词性、前缀后缀）。
  - **中层**：捕捉到的是句法特征（例如，短语结构）。
  - **高层（靠近输出的层）**：捕捉到的是丰富的语义特征（例如，词在上下文中的真正含义）。
- 所以，对于句子中的每个词，我们都能得到多个不同层次的向量表示。

**3. 任务特定的加权融合**

- 对于不同的下游任务（如情感分析、问答系统），不同层次的语言信息重要性是不同的。
- 因此，ELMo为每个下游任务**学习一组权重**，将这些来自不同层的向量进行**加权求和**，从而得到一个最终的任务专属词向量。
- 这个过程通常会在下游任务的微调过程中一起完成。

### GPT

自左向右单向生成

1. **模型架构**：
   - 使用Transformer Decoder结构
   - 掩码自注意力机制确保每个词只能关注其左侧上下文
2. **预训练任务**：
   - 自回归语言模型训练
   - 任务目标：给定前文词汇，预测下一个词
3. **核心能力**：
   - 通过预训练获得强大的文本生成能力
   - 具备基础的语言理解能力
4. **下游应用**：
   - 通过添加线性层进行任务微调
   - 将预训练能力迁移到具体任务

- 标准Transformer：Encoder理解源语言，Decoder生成目标语言
- **GPT (纯Decoder架构)**：
  - 无源语言概念
  - 所有Q、K、V均来自自身处理的序列
  - 通过自注意力机制，基于已有上下文预测下一个token

### BERT

MLM

#### **1. 核心目标与原因**

- **目标**：实现**双向上下文理解**。
- **为何需要Mask**：
  - **a. 强制双向理解**：通过遮蔽词语，迫使模型必须同时利用**左侧和右侧的上下文**来预测被遮蔽的词，从而学习到真正的双向语义表示。
  - **b. 防止信息泄露**：如果不Mask，模型在预测时能直接“看到”答案，无法学会依赖上下文进行推断。

#### **2. 实施方法**

1. **随机选择**：
   在输入序列中，随机选择 **15%** 的词汇（Token）进行处理。
2. **三种掩码策略**（对选中的15%词汇按以下比例处理）：
   - **80% 的情况**：替换为 `[MASK]`。
     - 例如：`今天天气很` **`[MASK]`** → 模型需预测“好”。
     - **目的**：最直接的“完形填空”，核心训练方式。
   - **10% 的情况**：替换为**一个随机词**。
     - 例如：`今天天气很` **`苹果`** → 模型需纠正并预测“好”。
     - **目的**：增强模型的**抗干扰和纠错能力**，使其对输入噪声更鲁棒。
   - **10% 的情况**：**保持原词不变**。
     - 例如：`今天天气很` **`好`** → 模型需强化“好”在这个上下文中的合理性。
     - **目的**：**平衡预训练与微调**。由于下游任务输入中没有`[MASK]`，此策略确保模型能良好处理正常文本，避免对`[MASK]`符号产生过度依赖。

<img src="/Users/chenhaoyang/Library/Application Support/typora-user-images/image-20251001200032112.png" alt="image-20251001200032112" style="zoom:50%;" />

<img src="/Users/chenhaoyang/Library/Application Support/typora-user-images/image-20251001200114799.png" alt="image-20251001200114799" style="zoom:50%;" />

BERT 输入是三部分，词embedding，position embedding，segment（区分句子，第一个句子0，第二个1）

<img src="/Users/chenhaoyang/Library/Application Support/typora-user-images/image-20251001200218016.png" alt="image-20251001200218016" style="zoom:50%;" />

https://www.bilibili.com/video/BV1GV4y1g7b9/?vd_source=5bf2abd640b441eb6f95f5cd173690fa

这里BERT利用transformer是真正双向的编码，看得到上下文，而ELMo是伪的，就是两个方向上单向编码拼接而成的，本质是单向编码

**Transformer Encoder + 双向掩码MLM预训练 + 任务微调**

<img src="/Users/chenhaoyang/Library/Application Support/typora-user-images/image-20251001200342110.png" alt="image-20251001200342110" style="zoom:50%;" />

### 模型参数量计算

BERT 词表WordPiece 切词，很小30k，H词维度，整体30K*H

QKV，Multi-Head `H*H*3+H*H`

注意，这里QKV的d_model和H一样的！是设置的，然后N head 的话，每个head的维度就是H/N！将所有头的输出拼接后（维度恢复为 H），需要通过一个线性层（矩阵 WO）整合信息，参数量为 H×H

两层Feed Forward H\*4H->4H\*H

L个Block，就是重复L次

总参数`30K*H+[3*H*H+H*H+H*4H+4H*H]*L=300000H+12H*H*L`

BERT Base H=786, L=12, N=12, 110M

BERT Large H=1024, L=24, N=16, 340M 

# Positional Encoding

### RoPE

> - 每个维度 i 对应不同的频率（波长），因为 100002i/dmodel 随着 i 增大而增大，波长变长。
> - 低维（小的 i）→ 高频变化 → 对近距离位置敏感
>   高维（大的 i）→ 低频变化 → 能区分较远的位置
> - 这样模型既能捕捉局部位置，也能捕捉长程位置关系。

<img src="/Users/chenhaoyang/Library/Application Support/typora-user-images/image-20251002154354081.png" alt="image-20251002154354081" style="zoom:50%;" />

这里注意力机制点积 

<img src="/Users/chenhaoyang/Library/Application Support/typora-user-images/image-20251002154505859.png" alt="image-20251002154505859" style="zoom:50%;" />

#### 优化

RoPE可以进行分块并行化矩阵运算提高效率

序列定长的任务预计算旋转矩阵

<img src="/Users/chenhaoyang/Library/Application Support/typora-user-images/image-20251002154620020.png" alt="image-20251002154620020" style="zoom:50%;" />

#### 本质

自带相对位置编码

长序列泛化能力强，三角函数

##### 缺点

计算复杂

模型维度有要求，比如将向量拆分成长度为2的子向量，词维度d要even

# Self-attention

这里row是seq_len，col是word embedding

自注意力 QKV都是自己，注意力机制一般都是KV同源，交叉注意力机制 Q decoder，KV encoder

<img src="/Users/chenhaoyang/Library/Application Support/typora-user-images/image-20251002162152876.png" alt="image-20251002162152876" style="zoom:50%;" />

除以 $\sqrt d_k$，这个作用是什么，将点积方差归一化到1，**稳定模型的训练，防止梯度消失**

q和k方差1，qk方差$d_k$，保持梯度活性，**防止注意力分数进入 Softmax 的饱和区，从而避免梯度消失，确保模型能够稳定、高效地进行训练，并产生更丰富的注意力分布。**

> “饱和区”指的是当 Softmax 的输入值中存在**一个或几个远远大于其他值**的数时，Softmax 的输出会变得极度不平衡。
>
> - **那个极大的值**对应的输出概率会无限接近 **1**。
> - **其他所有的值**对应的输出概率会无限接近 **0**。
>
> 这时，我们就说 Softmax 函数“饱和”了。它的输出不再是一个平滑的概率分布，而变成了一个近乎**one-hot**的向量（一个位置是1，其他全是0）。

```python
def attention(query, key, value, mask = No)
		batch, heads, seq_k, d_k = key.shape
  	att_ = torch.matmul(query, key.transpose(-1,-2))/ d_k**0.5
		if mask is not None:
      	att_ = att_.masked_fill(mask, -1e9)  #masked_fill 就是把mask里特定值替换为特定的填充值
    att_score = torch.softmax(att_, dim = -1) #dim=i就是在第i个维度上进行操作，-1就是最后一个维度
    return torch.matmul(att_score, value)
```

### Positional Encoding

RoPE 不是concat的！而是相加的

**位置编码向量 `e_i`（或 `p_i`）和词嵌入向量 `a_i` 的维度是完全相同的。** 

在原始Transformer论文和所有标准实现中，就是最简单的加法： `Final_Input = Token_Embedding + Position_Embedding` 这是一个纯粹的、逐元素的数学加法。

> **a) 信息融合而非信息隔离**
>
> - **拼接** 像是把两个信息块（语义和位置）放在不同的“隔间”里。模型需要后期通过复杂的变换来学习它们之间的关系。
> - **相加** 则是从一开始就强制进行了**信息融合**。位置信息直接“调制”或“修饰”了词嵌入的每一个维度。这相当于在输入的起点就创建了一个统一的、同时包含语义和位置信息的表示。
>
> **b) 维度保持与计算效率**
>
> - 拼接会使维度翻倍（从 `d_model` 到 `2 * d_model`），这会显著增加后续所有线性层（Q、K、V投影等）的计算量和参数数量。
> - 相加保持了原始的 `d_model` 维度，计算上更加高效。
>
> **因为词嵌入空间和位置编码空间在数学上是高度结构化的，且它们各自承载的信息类型不同。**
>
> **词嵌入空间的特点：**
>
> - 语义相似的词在嵌入空间中彼此靠近
> - 主要捕获**词汇间的语义关系**
>
> **位置编码空间的特点（以正弦编码为例）：**
>
> - 每个位置有唯一且确定的位置编码向量
> - 位置编码向量之间的关系是固定的（相对位置可以通过线性变换捕获）
>
> **当两者相加时：**
>
> - 模型有能力在后续层中**学习如何分离和利用**这两种信息
> - 通过训练，模型学会识别出“这部分信号模式来自位置编码，那部分信号模式来自词嵌入”

### Multi-head Attention

原始维度 d_model，nhead后就是d_head=d_model/nhead，最后得到的结果concat一起就行

<img src="/Users/chenhaoyang/Library/Application Support/typora-user-images/image-20251002165342610.png" alt="image-20251002165342610" style="zoom:50%;" />

单头注意力 = 只有一盏手电筒照一个方向。

多头注意力 = 拿好几盏手电筒，从不同角度同时照，就能看清更多结构。

关键在于 **并行 & 组合**：

1. **多个子空间并行学习**
   - 每个头虽然低维（64），但是学到的关注点不同。
   - 最后拼接时，得到的是 $[h \cdot d_v] = d_{model}$ 维度，和原始维度一样。
   - 信息不是减少，而是“分工协作”。
2. **投影矩阵补偿**
   - 每个头有自己独立的投影矩阵 $W_i^Q, W_i^K, W_i^V$，学习不同的线性变换。
   - 虽然每个头维度小，但整体的参数量其实比单头还大。
3. **更稳定的学习**
   - 单个大维度 head → 学到的模式容易“混在一起”。
   - 多个小维度 head → 分开学习不同模式，更清晰也更稳定。

# Transformer

映射到一个点或者向量，然后model通过两者位置得到逻辑关系，通过真实值和预测值偏差反向更新这个位置

### embedding

### encoder

residual + LayerNorm

LayerNorm(X+MultiheadAttention(X)) 

LayerNorm(X+FeedForward(X))

<img src="/Users/chenhaoyang/Library/Application Support/typora-user-images/image-20251002171015451.png" alt="image-20251002171015451" style="zoom:50%;" />

#### residual

y=F(x)+x 直接加 x和z

避免梯度消失

#### norm

避免数值差别过大以及数值太大导致梯度爆炸

#### FFN

FFN是Transformer的 **"特征变换和知识存储中心"**

如果说注意力机制是**搜索相关信息的搜索引擎**，那么FFN就是**处理和理解这些信息的智能处理器**。

Feed forward 引入非线性，增加表征，类似于信息加工，2个全连接层+1个relu

FFN(x) = max(0,xW1+b1)W2+b2

<img src="/Users/chenhaoyang/Library/Application Support/typora-user-images/image-20251002170838025.png" alt="image-20251002170838025" style="zoom:50%;" />

```python
# 标准FFN实现
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # 扩展维度
        self.linear2 = nn.Linear(d_ff, d_model)  # 压缩回原维度
        self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))
```

`d_model → d_ff → d_model`（通常 `d_ff = 4 * d_model`）

### decoder

encoder生成词向量，decoder接受词向量，并且生成翻译结果

#### mask

**Mask的核心思想**：在Softmax之前，将某些位置的注意力分数设置为一个极小的值（如 `-1e9`），这样经过Softmax后，这些位置的权重就变成了0，模型就不会关注这些位置

##### padding mask

**处理变长序列**：在一个batch中，不同句子的长度不同，我们需要用padding（通常是0）将短句子填充到相同长度。Padding Mask就是用来**阻止模型关注这些填充位置**。

##### look ahead mask

**防止信息泄露**：在生成任务中（如GPT），要确保在生成第t个词时，只能看到前面t-1个词，不能看到未来的词。

上三角mask

```python
mask = torch.triu(torch.ones(seq_len,seq_len),diagnal=1)
# tensor([[0., 1., 1., 1., 1.],
#         [0., 0., 1., 1., 1.],
#         [0., 0., 0., 1., 1.],
#         [0., 0., 0., 0., 1.],
#         [0., 0., 0., 0., 0.]])
```

<img src="/Users/chenhaoyang/Library/Application Support/typora-user-images/image-20251002173143938.png" alt="image-20251002173143938" style="zoom:50%;" />

### KV encoder, Q decoder

- **Encoder（知识库）**：把输入序列（如英文句子）编码成一组**键值对（K-V）**
  - **Key**：就像文档的"关键词"或"索引"
  - **Value**：就像文档的"具体内容"
- **Decoder（用户）**：生成输出序列（如中文句子）时，每个步骤都产生一个**查询（Q）**
  - **Query**：就像用户的"搜索问题"

源语句 KV，目标语句 Q已经生成的词

在Decoder中已经生成的词Q和Encoder中的源词KV做cross attention，从已经生成的词去全部词里挑重点

<img src="/Users/chenhaoyang/Library/Application Support/typora-user-images/image-20251002174342628.png" alt="image-20251002174342628" style="zoom:50%;" />

# Trival

## Torch

`torch.triu` 返回上三角`diagonal=0 `默认**包含对角线及以上的元素**

`diagonal=1` **对角线向上移动1位，不包含主对角线**

`diagonal=-1` **对角线向下移动1位，包含主对角线和下一条对角线**

### dimension

- **`unsqueeze`**：增加一个维度（在指定位置插入维度1）
  - unsqueeze(i) 就是在i维度插入1
  - `unsqueeze(dim=k)` 里的 `dim` 必须满足 `0 <= dim <= x.dim()`
- **`squeeze`**：删除所有维度为1的维度（压缩）

```python
x = torch.tensor([1,2,3]) #不考虑最外层的维度，shape(3,)\
y = x.unsqueeze(0)        # (1, 3) 
z = x.unsqueeze(1)        # (3, 1)


a = torch.tensor([[[1], [2], [3]]])   # (1, 3, 1)
b = a.squeeze()												 # (3,)
```

## Pre-Norm vs Post-Norm

- **Pre-Norm**：`LayerNorm(输入) → 网络层 → 残差连接`
- **Post-Norm**：`网络层 → 残差连接 → LayerNorm(结果)`

Pre-Norm更容易训练，每一层的输出都直接累加到最终结果中，梯度有"快速通道"可以直达底层，不容易出现梯度消失

```python
# Pre-Norm: 信息可以无损传递
x_{t+1} = x_t + F_t(Norm(x_t))
# 展开后：
x_l = x_0 + F_0(Norm(x_0)) + F_1(Norm(x_1)) + ... + F_{l-1}(Norm(x_{l-1}))
```

```
x_{t+1} ≈ x_t （因为相对变化很小）
F_t(Norm(x_t)) + F_{t+1}(Norm(x_{t+1})) ≈ F_t(Norm(x_t)) + F_{t+1}(Norm(x_t))
```

**这相当于把多个层"并联"而不是"串联"**，相当于增加了网络宽度而不是深度。而深度比宽度对模型性能更重要。

Post-Norm为什么难训练：梯度消失问题，**关键问题**：底层信息（`x_0`）和底层变换（`F_0`）的贡献随着层数增加而**指数级衰减**。

残差连接"名存实亡"

残差连接的本意是提供梯度快速通道，但Post-Norm中的Normalization操作削弱了这个通道：

- 原始残差：`输出 = 输入 + 变换`
- Post-Norm残差：`输出 = Norm(输入 + 变换)`

**Normalization操作破坏了恒等映射**，使得梯度不能无损传递。

<img src="/Users/chenhaoyang/Library/Application Support/typora-user-images/image-20251002175021829.png" alt="image-20251002175021829" style="zoom:50%;" />

<img src="/Users/chenhaoyang/Library/Application Support/typora-user-images/image-20251002175130081.png" alt="image-20251002175130081" style="zoom:50%;" />

## Adam优化器为什么更抗梯度消失 vs SGD

SGD

```python
update = learning_rate * gradient
# 如果gradient很小，update也很小 → 训练停滞
```

Adam

```python
# 更新量有下限保障
update = learning_rate * (动量估计 / sqrt(方差估计 + ε))
```

1. **动量机制**：`m_t` 累积历史梯度，即使当前梯度很小，动量仍可能较大
2. **自适应学习率**：`v_t` 根据梯度幅度调整，小梯度对应参数会获得相对更大的更新
3. **更新量下限**：理论上只要梯度不为零，更新量就有 `O(η)` 量级

- **SGD**：像在平缓的斜坡上步行，坡度小了就走得慢
- **Adam**：像开着有动力的车，即使当前坡度平缓，靠惯性也能前进

<img src="/Users/chenhaoyang/Library/Application Support/typora-user-images/image-20251002175359591.png" alt="image-20251002175359591" style="zoom:50%;" />

<img src="/Users/chenhaoyang/Library/Application Support/typora-user-images/image-20251002175545665.png" alt="image-20251002175545665" style="zoom:50%;" />
