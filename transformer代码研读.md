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

#### 模型输入

##### 数据处理

1. batch与max_len：注意设置源数据的长度上限并进行P符号填充，从而形成矩阵
2. 



