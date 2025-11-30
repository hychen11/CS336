https://stanford-cs336.github.io/spring2025/

# Lec1

basics, systems, scaling laws, data, alignment

* Data processing: avoid wasting precious compute updating on bad / irrelevant data
* Tokenization: working with raw bytes is elegant, but compute-inefficient with today's model architectures.
* Model architecture: many changes motivated by reducing memory or FLOPs (e.g., sharing KV caches, sliding window attention)
* Training: we can get away with a single epoch
* Scaling laws: use less compute on smaller models to do hyperparameter tuning
* Alignment: if tune model more to desired use cases, require smaller base models

### Tokenization

convert between strings and sequences of integers (tokens)

#### **Byte-Pair Encoding (BPE) tokenizer**

#### Byte-based tokenization

 Unicode encoding is UTF-8 (compression rate 1, bad)

#### Word-based tokenization

#### Byte Pair Encoding (BPE)

https://zhuanlan.zhihu.com/p/448147465

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

### Variants:

Activation functions: ReLU, SwiGLU

Positional encodings: sinusoidal, RoPE

Normalization: LayerNorm, RMSNorm

Placement of normalization: pre-norm versus post-norm

MLP: dense, mixture of experts

Attention: full, sliding window, linear

Lower-dimensional attention: group-query attention (GQA), multi-head latent attention (MLA)

State-space models: Hyena

### Training

Optimizer (e.g., AdamW, Muon, SOAP)

Learning rate schedule (e.g., cosine, WSD) 

Batch size (e..g, critical batch size)

Regularization (e.g., dropout, weight decay)

Hyperparameters (number of heads, hidden dimension): grid search

### Inference

Includes **prefill and decode**

Prefill (similar to training): tokens are given, can process all at once (compute-bound)

Decode: need to generate one token at a time (memory-bound)

* Use cheaper model (via model pruning, quantization, distillation)

* Speculative decoding: use a cheaper "draft" model to generate multiple tokens, then use the full model to score in parallel (exact decoding!)

* Systems optimizations: KV caching, batching

### scaling laws

Goal: do experiments at small scale, predict hyperparameters/loss at large scale

given a FLOPs budget (C), use **bigger model (N) or train on more tokens**

D = 20N (like 1.4B parameter model should be trained on 28B tokens)

> 1. 提前预测最终模型效果，知道每次训练的大概能到什么程度，要是不及预期可以根据预算再进行调整
> 2. 在小尺寸模型上做置信的实验，进行数据、算法策略验证，降低实验的时间、资源成本
> 3. 在真正的大规模预训练中，随时监测模型效果是否符合预期

### Evaluation

### Data curation

### Data processing

### Supervised finetuning (SFT)

Supervised learning: fine-tune model to maximize p(response | prompt)

# Lec 2

* **primitives needed to train a model**
* **go bottom-up from tensors to models to optimizers to the training loop.**
* **pay close attention to efficiency**

### training

```python
note_about_randomness()  
data_loading()

optimizer()
train_loop()
checkpointing()
mixed_precision_training()
```

##### `note_about_randomness()`

记录或控制随机性，保证训练的 可重复性（Reproducibility）

```python
import torch
import random
import numpy as np

def note_about_randomness():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
```

##### `data_loading()`

`torch.utils.data.DataLoader`

* 加上 `num_workers` 参数，开启多线程加载
* **预处理 pipeline：**
  - 数据增强（augmentation）
  - tokenization / normalization
  - batch 拼接
  - pin_memory 加速 GPU 内存传输

##### `optimizer()`

* 定义优化器（如 SGD, Adam），负责根据 loss 更新模型参数
* `optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)`

##### `train_loop()`

```python
for batch in dataloader:
    outputs = model(batch)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

这里`loss.backward()`：只是把梯度算出来，暂时存储在每个参数的 `.grad` 属性里。

`optimizer.step()`：读取 `.grad`，做优化算法的参数更新。

如果不清零，第二次 `.backward()` 会把梯度叠加到第一次的基础上，导致错误。``optimizer.zero_grad()`清空参数的 `.grad` 属性

* 分布式训练（如 DDP）时，需在这里调用同步/异步通信

##### `checkpointing()`

保存训练的中间状态

- 模型参数
- 优化器状态
- 学习率调度器状态
- 当前 epoch/step

```python
torch.save({'model': model.state_dict(),
            'optimizer': optimizer.state_dict()}, 'checkpoint.pth')
```

##### `mixed_precision_training()`

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()
with autocast():
    outputs = model(batch)
    loss = loss_fn(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### **FLOPs**

70B 15T token

```python
#total_flops = 6 * N_params * N_tokens
total_flops = 6*70e9*15e12
```

1 次 **前向传播**（forward pass）大概 ≈ 2 × 参数数量

1 次 **反向传播**（backward pass）大概 ≈ 4 × 参数数量

这里4就是需要计算两次矩阵乘法

一个计算梯度权重，一个计算梯度的输入

```python
bytes_per_parameter = 4 + 4 + (4 + 4) # parameters, gradients, optimizer state
```

use float32 for parameters and gradients, also use bf16 for parameters and gradients (2 + 2) and keep an extra float32 copy of the parameters (4). This doesn't save memory, but is faster.

activations are not accounted for

H100s support two variants of FP8: E4M3 (range [-448, 448]) and E5M2  ([-57344, 57344])

## tensor

### `tensors_basics()`

Tensors are the basic building block for storing everything: parameters, gradients, optimizer state, data, activations.

```python
x=torch.tensor([[1.,2,3],[4,5,6]])
x=torch.zeros(4,8)
x=torch.ones(4,8)
x=torch.randn(4,8)

x=torch.empty(4,8)
#set the value later
nn.init.trunc_normal_(x,mean=0,std=1,a=-2,b=2) #截断正态分布初始化函数，a截断下界，b截断上界
```

### `tensors_memory()`

**Float32**, fp32 is the default.  (4bytes)

1 sign + 8 exponent + 23 fractions

**Float16** (2bytes)

fp16 5exp

1 sign + 5 exponent + 10 fractions

```
x = torch.tensor([1e-8], dtype=torch.float16)  # @inspect x
assert x == 0  # Underflow!
```

表示非常小的数时的限制，即**数值下溢（underflow）**问题。

**bfloat16**

1 sign + 8 exponent + 7 fractions

没有underflow了

bf16 8exp

**Fp8**

### `tensors_on_gpus()`

tensor stores in CPU memory and need to move to GPU memory

```python
x.device == torch.device("cpu")
torch.cuda.is_available()
num_gpus=torch.cuda.device_count()
for i in range(num_gpus):
    properties = torch.cuda.get_device_properties(i)
    
memory_allocated = torch.cuda.memory_allocated()
y = x.to("cuda:0")
y.device = torch.device("cuda",0)
```

### `tensor_operations()`

#### `tensor_storage()`

```python
x.stride(dim)
# 在 PyTorch 中，x.stride(dim) 表示：在维度 dim 上移动一步（索引 +1）时，在底层内存中要跳过多少个元素。
```

#### `tensor_slicing()`

Many operations simply provide a different view of the tensor.

This does not make a copy, and therefore mutations in one tensor affects the other.

```python
x = torch.tensor([[1., 2, 3], [4, 5, 6]])

y = x[0] # row 0
assert torch.equal(y, torch.tensor([1., 2, 3]))
assert same_storage(x, y)

y = x[:,1] #column 1
assert torch.equal(y, torch.tensor([2, 5]))
assert same_storage(x, y)

y = x.view(3, 2) # @inspect y
assert torch.equal(y, torch.tensor([[1, 2], [3, 4], [5, 6]]))
assert same_storage(x, y)

y = x.transpose(1, 0) # @inspect y
y.view(2,3)
assert torch.equal(y, torch.tensor([[1, 4], [2, 5], [3, 6]]))
assert same_storage(x, y)

# One can enforce a tensor to be contiguous first
y = x.transpose(1, 0).contiguous().view(2, 3) # @inspect y
assert not same_storage(x, y)
```

#### `tensor_elementwise()`

```python
# triu takes the upper triangular part of a matrix.
x = torch.ones(3, 3).triu()
# causal attention mask
```

#### `tensor_matmul()`

#### `tensor_einops()`

Einops is a library for manipulating tensors

```python
z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")
# Or can use ... to represent broadcasting over any number of dimensions
z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2") 

x: Float[torch.Tensor, "batch seq hidden"] = torch.ones(2, 3, 4)
y = x.mean(dim=-1)
y = reduce(x, "... hidden -> ...", "sum")

# rearrange
x: Float[torch.Tensor, "batch seq total_hidden"] = torch.ones(2, 3, 8)
# ...where total_hidden is a flattened representation of heads * hidden1
w: Float[torch.Tensor, "hidden1 hidden2"] = torch.ones(4, 4)
# Break up total_hidden into two dimensions (heads and hidden1)
x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2) 
# Perform the transformation by w:
x = einsum(x, w, "... hidden1, hidden1 hidden2 -> ... hidden2") # @inspect x
# Combine heads and hidden2 back together:
x = rearrange(x, "... heads hidden2 -> ... (heads hidden2)") # @inspect x
```

**A100 has a peak performance of 312 teraFLOP/s**

```
assert a100_flop_per_sec == 312e12
```

**H100 has a peak performance of 1979 teraFLOP/s with sparsity, 50% without**

```
assert h100_flop_per_sec == 1979e12 / 2
```

### **Model FLOPs utilization (MFU)**

Definition: (actual FLOP/s) / (promised FLOP/s) [ignore communication/overhead]

mfu = actual_flop_per_sec / promised_flop_per_sec # @inspect mfu

**Usually, MFU of >= 0.5 is quite good (and will be higher if matmuls dominate)**

**comparing bfloat16 to float32, the actual FLOP/s is higher**

Putting it togther:

Forward pass: 2 (# data points) (# parameters) FLOPs

Backward pass: 4 (# data points) (# parameters) FLOPs

Total: 6 (# data points) (# parameters) FLOPs

### data_loading

`orig_data.tofile("data.npy")`

Use memmap to lazily load only the accessed parts into memory.

`data = np.memmap("data.npy", dtype=np.int32)`

By default, CPU tensors are in paged memory. We can explicitly pin

x = x.pin_memory()

**This allows us to copy** x **from CPU into GPU asynchronously.**

`x = x.to(device, non_blocking=True)`

This allows us to do two things in parallel (not done here):

Fetch the next batch of data into CPU

Process x on the GPU.

**Let's define the AdaGrad optimizer**

* momentum = SGD + exponential averaging of grad

* AdaGrad = SGD + averaging by grad^2

* RMSProp = AdaGrad + exponentially averaging of grad^2

* Adam = RMSProp + momentum

# Lec 3

### Pre-vs-post norm

just use Pre norm!!!

post norm will be unstable and have to do some careful lr warm-up style things

Grok and gamma 2 add layer norm after FFN

![](./assets/L3_1.png)

> why layer norm in the residual path bad?
>
> the residual gives you the identity connection from top to bottom, this makes gradient propagation very easy. Put layer in the residual might mess that kind of gradient behavior

### LayerNorm vs RMSNorm

nearly all the model use RMSNorm

Modern explanation – it’s faster (and just as good).

• Fewer operations (no mean calculation)

• Fewer parameters (no bias term to store)

![](./assets/L3_2.png)

important to think about the memory, not just about FLOPS

![](./assets/L3_3.png)

### dropping bias terms

Most modern transformers don’t have bias terms.

$FFN(x)=\sigma(xW_1)W_2$

Reasons: memory (similar to RMSnorm) and optimization stability

### recap

Basically everyone does pre-norm.

* Intuition – keep the good parts of residual connections

* Observations – nicer gradient propagation, fewer spike

* Some people add a second norm outside the residual stream (NOT post-norm)

Most people do RMSnorm

* In practice, works as well as LayerNorm

* But, has fewer parameters to move around, which saves on wallclock time

* People more generally drop bias terms since the compute/param tradeoffs are not great.

### Activations

Llama, PaLM,T5 v1.1, most models post 2023 use SwiGLU/GeGLU

SwiGLU (swish is 𝑥 ∗ sigmoid(𝑥))

> Note: Gated models use smaller dimensions for the 𝑑𝑓𝑓 by 2/3???

可以把 **gating activation（门控制激活）** 理解成：

> **网络在每个位置用一个门（gate）来决定：信息要放大、通过、还是被压下去。**

它不是只做“激活”，而是在激活 *前后* 加上一个可学习的“门”，让模型动态控制流经的信号量。

下面用你能最快抓住直觉的方式解释。

y = gate(x) ⊙ activation(x)

门（gate）一般是：

- sigmoid
- swish
- GLU gate: linear(x1) + sigmoid(x2)
- 或者更复杂如 SwiGLU, GeGLU, ReGLU  

### consensus hyperparameter 1

There are two dimensions that are relevant – the feedforward dim (𝑑𝑓𝑓) and model dim (𝑑𝑚𝑜𝑑𝑒𝑙 ). What should their relationship be?

$d_{ff} = 4d_{model}$

This is almost always true. There’s just a few exceptions.

Remember that GLU variants scale down by 2/3rd. This means most GLU variants have 𝑑𝑓𝑓 =8/3𝑑𝑚𝑜𝑑𝑒𝑙 .

The ‘default’ choices of 𝑑𝑓𝑓 = 4𝑑𝑚𝑜𝑑𝑒𝑙 and GeLU 𝑑𝑓𝑓 = 2.66𝑑𝑚𝑜𝑑𝑒𝑙 have worked well for nearly all modern LLMs.

### Aspect ratios

𝒅𝒎𝒐𝒅𝒆𝒍/𝒏𝒍𝒂𝒚𝒆𝒓

Extremely deep models are harder to parallelize and have higher latency

### vocabulary sizes

Monolingual models – 30-50k vocab

Multilingual / production systems 100-250k

### Dropout and other regularization

pretraining dont need regularization?

pretraining usually need one epoch

There is a lot of data (trillions of tokens), more than parameters., SGD only does a single pass on a corpus (hard to memorize)

drop out 用的少了，一般用weight decay? Intuition violation!

> #### weight decay
>
> 传统梯度下降更新是：
> $$
> w \leftarrow w - \eta \cdot \nabla L(w)
> $$
> 加入 weight decay（L2 正则）后：
> $$
> w \leftarrow w - \eta \cdot (\nabla L(w) + \lambda w)
> $$
> 也就是多加了一个项：
> $$
> \lambda w
> $$
> 这就等价于每次都把权重往 0 拉一点。
>
> 因为“大的参数”容易导致模型更复杂、更加拟合训练数据。 Weight Decay 让模型更平滑、更简单，从而提高泛化能力。可以防止过拟合

Many older models used dropout during pretraining

Newer models (except Qwen) rely only on weight decay

#### Why weight decay LLMs

优化损失函数提升性能

It’s not to control overfitting

为什么不用drop out了呢？是因为单epoch情况下没有必要

### Stability tricks

Softmaxes – can be ill-behaved due to exponentials / divison by zero

原本的softmax 就是Z(x) = $\Sigma e^{x_i}$

然后z loss 就是在loss function上

它是一个额外加到 loss 上的小项，形式如下：减去
$$
L_{z} = \alpha \cdot \left( \sum_i z_i^2 \right)
$$

#### Softmax

假设模型输出 logits $\mathbf{z} = [z_1, z_2, \dots, z_V]$（Vocab size 为 $V$）：
$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^V e^{z_j}}
$$
通常训练的目标是 **交叉熵 loss**：
$$
L_{\text{CE}} = - \log \frac{e^{z_{y}}}{\sum_j e^{z_j}} = -z_y + \log \sum_j e^{z_j}
$$

#### z loss

$$
L_z = \alpha \, s^2 = \alpha \, (\log \sum_i e^{z_i})^2
$$

训练总 loss：
$$
L = L_{\text{CE}} + L_z = -z_y + \log \sum_j e^{z_j} + \alpha (\log \sum_j e^{z_j})^2
$$
优化成功的话，logZ(x)恒等于0

### QK norm

The query and keys are Layer (RMS) normed before going into the softmax operation.

![](./assets/L3_4.png)

inference 也会保留

### Attention heads

GQA / MQA : Saving inference costs by reducing the number of heads

#### Multi-Query Attention (MQA) 

have multiple queries, but just one dimension for keys and values

#### Does MQA hurt? 

Small PPL hit w/ MQA [Shazeer 2019] 

#### GQA

Don’t go all the way to one dimension of KV – have fewer dims

Simple knob to control expressiveness (key-query ratio) and inference efficiency

![](./assets/L3_5.png)

### Sparse window attention

Attending to the entire context can be expensive (quadratic).

Build sparse / structured attention that trades off expressiveness vs runtime (GPT3)

### sliding window attention

Just use the main part of the strided pattern – let depth extend effective context (Mistral)

简单说sparse更灵活

**局部 + 全局**（Longformer/BigBird）

- 局部是固定窗口
- 全局 token 可以 attend 所有位置

**随机稀疏**（Reformer、一些 block-sparse Transformer）

- 每个 token 的注意力位置部分随机选取
- 用于长序列时降低复杂度

### Current standard trick – interleave ‘full’ and ‘LR’ attention

From Cohere Command A – Every 4th layer is a full attention

- 大部分层（3/4）使用 **滑动窗口注意力**（高效，计算量小）
- 每隔 3 层，第 4 层使用 **全注意力**（捕捉全局信息）

Long-range info via NoPE, short-range info via RoPE + SWA.

# LMs

![](./assets/TS1.png)

FF layers use SwiGLU, not ReLU

## Pre-vs-post norm

![](./assets/TS2.png)

If putting LayerNorms in residual streams is bad.. Why not post-norm outside the stream?

## LayerNorm vs RMSNorm

![](./assets/TS3.png)

## Perplexity

“困惑率”（**Perplexity, PPL**）是衡量语言模型预测能力的一个经典指标，尤其在 **概率语言模型和 LLM** 中用得非常广泛

假设模型给定一个序列 $x_1, x_2, \dots, x_N$，语言模型会计算条件概率：
$$
P(x_1, x_2, \dots, x_N) = \prod_{t=1}^{N} P(x_t \mid x_1, \dots, x_{t-1})
$$
**困惑率**定义为：
$$
\text{PPL} = \exp \left( - \frac{1}{N} \sum_{t=1}^{N} \log P(x_t \mid x_1, \dots, x_{t-1}) \right)
$$
或者等价地：
$$
\text{PPL} = 2^{H}, \quad H = \text{交叉熵}
$$
PPL 越小 → 模型越“自信” → 预测越准确

PPL 越大 → 模型越“困惑” → 预测不确定性高
