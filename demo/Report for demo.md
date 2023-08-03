### Report list for demo

#### 文件说明

- vit_model.py：vit模型构建

- train.py：导入预训练参数与模型，进行微调训练
- predict.py：导入微调训练结果，展示图片分类的可能性

- 本次模型构建中采用的图片大小与patch大小均采用vit_base的值，即图片大小224 * 224，patch大小16 * 16，embedings维度16 * 16 * 3。

#### 注意力机制与transformer的理解

注意力机制的基本思想是根据输入的不同部分的重要程度，动态地分配模型的注意力，即数学上根据输入之间的相似性分数进行加权聚合，从而使得自注意力机制可以实现全局信息的建模和非局部关系的捕捉。

<img src="C:\Users\23850\AppData\Roaming\Typora\typora-user-images\image-20230803214150522.png" alt="image-20230803214150522" style="zoom:50%;" />

<img src="C:\Users\23850\AppData\Roaming\Typora\typora-user-images\image-20230708191336589.png" alt="image-20230708191336589" style="zoom:67%;" />

<img src="C:\Users\23850\AppData\Roaming\Typora\typora-user-images\image-20230803214605217.png" alt="image-20230803214605217" style="zoom:50%;" />

<img src="C:\Users\23850\AppData\Roaming\Typora\typora-user-images\image-20230803211503217.png" alt="image-20230803211503217" style="zoom:50%;" />

![vit](https://img-blog.csdnimg.cn/20210626105321101.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70#pic_center)

<img src="https://img-blog.csdnimg.cn/20210704114505695.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70#pic_center" alt="encoder" style="zoom:67%;" />

<img src="https://img-blog.csdnimg.cn/20210704124600507.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70#pic_center" alt="vit-b/16" style="zoom:67%;" />

#### 

#### 代码结构

<img src="C:\Users\23850\AppData\Roaming\Typora\typora-user-images\image-20230803220101616.png" alt="image-20230803220101616"  />