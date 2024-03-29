# CNN Note
## 概念
* 卷积：从上一层矩阵n*m计算特征值，若有颜色三通道，则为特征值相加
* 特征图：特征值组成的图，可能有多个
* 卷积核：内积参数矩阵（内积完还要加参数）
* 多层卷积：再上一层卷积的特征图上再做卷积
* 卷积参数共享：卷积核对于每一个区域参数相同
* 池化层：压缩/下采样downsample，常用：最大池化（优于平均池化）
* 全连接层：拉长为特征向量，生成最后输出层
## 公式
* 特征图size: h=(h0-f+2*p)/s+1, f卷积核高度，s滑动窗口步长，p边缘填充圈数，h0原始高度（宽同理）
## 参数
* 滑动窗口步长：每一次卷积范围的移动步长（常用3*3，step=1）
* 卷积核尺寸
* 边缘填充(防止边缘数据权重过小)：zero-padding，一般添加一圈
* 卷积核个数：特征图的个数