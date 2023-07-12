### transformer代码研读

#### 框架解读

```
Code
-Encoder
--N*layer
---two sublayer
----layer 1:Multi-Head Attention(多头的自注意机制)
----layer 2:Feed Forward(全连接前馈网络)
-----sublayer operation
------residual残差连接
------layer normalization
-Decoder
--N*layer
---three sublayer
----layer 1 like the one in Encoder
----layer 2:Masked Multi-Head Attention
----layer 3 like the layer 2 in Encoder
-----the same sublayer operation
-Input
--Input Embedding
--Positional Encoding
-Output
--softmax result of Decoder's output
```

#### 多维矩阵补充理解

这个地方是之前不能理解代码里面矩阵数据流动的主要原因。

**所有大于二维的，最终都是以二维为基础堆叠在一起的！！**

**所以在矩阵运算的时候，其实最后都可以转成我们常见的二维矩阵运算，遵循的原则是：在多维矩阵相乘中，需最后两维满足shape匹配原则，最后两维才是有数据的矩阵，前面的维度只是矩阵的排列而已！**

#### 模型输入

##### 数据处理

1. batch与max_len：注意设置源数据的长度上限并进行P符号填充，从而形成矩阵
2. 



