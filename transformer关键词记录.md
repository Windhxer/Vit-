架构不同以往的关键点：

--抛弃了循环等理念，使用`多头`与`自注意`的模型理念

#### Part 1 导言

1.1 attention：将编码器的信息有效地传递给解码器

1.2 并行化能力

#### Part 2 背景

2.1 卷积神经网络替换时序神经网络

2.2 Transformer仅依赖于自注意力

#### Part 3 模型架构

3.1 使用Z~t~ 序列来表示对输入X~t~ 序列的编码，最终生成Y~m~ 的解码结果序列

3.2 自回归：过去时刻的输出同样作为当前时刻的输入

3.3 架构图：

<img src="C:\Users\23850\AppData\Roaming\Typora\typora-user-images\image-20230708191336589.png" alt="image-20230708191336589" style="zoom:67%;" />

3.4 Attention机制

- 用于计算相关程度并传递整体的序列信息

- 将query(Q)和key-value pairs映射到输出上，输出是V中所有values的加权，而权重则由Q和K的相似度进行Softmax归一化操作得到。
- 在Transformer中使用的Attention是Scaled Dot-Product Attention, 是归一化的点乘Attention，假设输入的query q 、key维度为 d~k~ ,value维度为 d~v~ , 计算query和每个key的点乘操作，并除以 d~k~ 的平方根（cos乘积，如果为0则为正交，无关联），然后应用Softmax函数计算权重。

<img src="C:\Users\23850\AppData\Roaming\Typora\typora-user-images\image-20230708193747580.png" alt="image-20230708193747580" style="zoom:67%;" />

<img src="https://pic1.zhimg.com/80/v2-f51f8363005d880f2f412246ae0ffa14_720w.webp" alt="img" style="zoom: 50%;" />

- 如果只对Q、K、V做一次这样的权重操作是不够的，这里提出了Multi-Head Attention，操作包括：

--首先对Q、K、V做一次线性映射，将输入维度d~model~ 的Q,K,V三个矩阵映射到R^m*dk^ 

--然后在采用Scaled Dot-Product Attention计算出结果

--多次进行上述两步操作，然后将得到的结果进行合并

--将合并的结果进行线性变换

目前理解：尝试学习得到多个不同的映射方法，从而得到更有效的结果

3.5 encoder-decoder结构

- Encoder

Encoder有N=6层，每层包括两个sub-layers:

--第一个sub-layer是multi-head self-attention mechanism，用来计算输入的self-attention

--第二个sub-layer是简单的全连接网络。

每个sub-layer都模拟了残差网络，每个sub-layer的输出都是`LayerName(x+Sublayer(x))`

- Decoder

Decoder也是N=6层，每层包括3个sub-layers：

--第一个是Masked multi-head self-attention，也是计算输入的self-attention，但是因为是生成过程，因此在时刻i的时候，大于i的时刻都没有结果，只有小于i 的时刻有结果，因此需要做Mask

--第二个sub-layer是对encoder的输入进行attention计算,其中encoder给出的输入作为key-value键值对，decoder的第二层输出作为query

--第三个sub-layer是全连接网络，与Encoder相同

- 在Encoder-Decoder架构中，有三处Multi-head Attention模块，分别是：
  1. Encoder模块的Self-Attention，在Encoder中，每层的Self-Attention的输入 Q=K=V , 都是上一层的输出。Encoder中的每个position都能够获取到前一层的所有位置的输出。
  2. Decoder模块的Mask Self-Attention，在Decoder中，每个position只能获取到之前position的信息，因此需要做mask，即将i时刻之后的权重设为极小的负数，使其在进行Softmax之后可以归零
  3. Encoder-Decoder之间的Attention，其中 Q来自于之前的Decoder层输出，K,V 来自于encoder的输出，这样decoder的每个位置都能够获取到输入序列的所有位置信息。

3.6 Position-wise Feed-forward Networks

在进行了Attention操作之后，encoder和decoder中的每一层都包含了一个全连接前向网络，对每个position的向量分别进行相同的操作，包括两个线性变换和一个ReLU激活输出，每一层的参数都不同：

<img src="C:\Users\23850\AppData\Roaming\Typora\typora-user-images\image-20230708203840052.png" alt="image-20230708203840052" style="zoom:67%;" />

3.7 Position Embedding

因为模型不包括recurrence/convolution，因此是无法捕捉到序列顺序信息的，例如将K、V按行进行打乱，那么Attention之后的结果是一样的。但是序列信息非常重要，代表着全局的结构，因此必须将序列的token相对或者绝对position信息利用起来。

这里每个token的position embedding 向量维度也是 d~model~=512, 然后将原本的input embedding和position embedding加起来组成最终的embedding作为encoder/decoder的输入。其中position embedding计算公式如下

<img src="C:\Users\23850\AppData\Roaming\Typora\typora-user-images\image-20230708203956476.png" alt="image-20230708203956476" style="zoom:67%;" />

其中 pos 表示位置index， i 表示dimension index。

选取sin position embedding而不是训练的position embedding的原因：

1. 这样可以直接计算embedding而不需要训练，减少了训练参数
2. 这样允许模型将position embedding扩展到超过了training set中最长position的position，例如测试集中出现了更大的position，sin position embedding依然可以给出结果，但不存在训练到的embedding。

