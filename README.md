# 矩阵分析SP2023

![](https://img.shields.io/badge/release-v0.0.1-blue)
![](https://img.shields.io/badge/unit%20tests-passing-brightgreen)
![](https://img.shields.io/github/stars/PKUcoldkeyboard/Matrix_Analysis_SP2023?style=social)

[English](README-en.md) | [简体中文](README.md)

## 简介
2023春季学期矩阵分析课程编程作业

## 任务
### 1. QR Algorithm
1） 令 $A_1=A$ ，对 $A_k=Q_k R_k$ 进行QR分解，更新 $A_{k+1}=R_k Q_k$ ，直到 $A_{k+1}$ 接近于一个上三角矩阵。我们设定tolerance值为 $\epsilon=10^{-10}$ ，即当 $|a_{ij}^{(k)}| < \epsilon, \forall i > j$ 时，停止迭代。请将QR算法作为一个特征分解函数来实现，并提供实现的代码(15%)。

2） 请考虑以下矩阵

$$
A_{1}=\left(\begin{matrix}
10 & 7 & 8 & 7 \\
7 & 5 & 6 & 5 \\
8 & 6 & 10 & 9 \\
7 & 5 & 9 & 10
\end{matrix}\right), \quad A_{2} = \left(\begin{matrix}
2 & 3 & 4 & 5 & 6 \\
4 & 4 & 5 & 6 & 7 \\
0 & 3 & 6 & 7 & 8 \\
0 & 0 & 2 & 8 & 9 \\
0 & 0 & 0 & 1 & 0
\end{matrix}\right), \quad A_{3} = \left(\begin{matrix}
1 & \frac{1}{2} & \cdots & \frac{1}{6} \\
\frac{1}{2} & \frac{1}{3} & \cdots & \frac{1}{7} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{1}{6} & \frac{1}{7} & \cdots & \frac{1}{11}
\end{matrix}\right)
$$

请说明，对于上述矩阵，通过QR算法得到的矩阵序列 $A_k$ 收敛于上三角矩阵。(15%)

3） 请计算上述每个矩阵的特征值和矩阵2-norm条件数（即 $cond_2 (𝐴) = \Vert 𝐴 \Vert_2 \Vert 𝐴 \Vert_{2}^{-1}$ ）。（请指出如何计算每个矩阵的2-norm条件数并提供结果，20%）。


### 2. Matrix Factorization for Data Processing
由于维度的复杂性和维度诅咒，直接处理高维数据需要大量的计算资源。非负矩阵分解（NMF）作为一种降维技术被提出，在图像处理中得到了重要的应用。通过采用NMF，非负的高维矩阵可以被分解成两个非负的低维矩阵，其中一个包括列向量，可以被视为数据空间中的基向量，另一个则包含缩放这些基向量的系数行。此外，NMF也可用于文本数据处理。我们可以检查系数矩阵中的每一列，并确定具有最大系数的行号，其中行号表示原始矩阵中各列的聚类ID。这种聚类特性意味着NMF可以用于数据聚类。

任务：

1） 请从`./dataset/images`中下载数据集，其中包含从动漫脸部数据集中采样的RGB图像。根据每个颜色通道，该数据集中的RGB图像可以被载入三个单通道矩阵，即 $M_R$ , $M_G$ 和 $M_B$ ，每个矩阵都包含 $K=H \times W$ 个像素和$N'$个脸部请将三个单通道矩阵合并为矩阵 $M \in R^{𝐾×𝑁}$ ，其中 $N=3 \times N'$ 。(请提供代码，不需要将矩阵保存到文件中，10%)

2） 试着应用SVD方法将矩阵 $M \in R^{K\times N}$ 分解成两个低秩矩阵，即矩阵 $U \in R^{K \times d}$ 和矩阵 $I\in R^{d \times N}$ ，其中d是实践中的经验参数，本实验中设定为16。请完成以下任务。你可以直接在本测验中应用现有的API。(请提供代码和结果，20%)

  ⯎ 在报告中提供来自`./dataset/images`的数据的奇异值。

  ⯎ 在报告中通过改造来自`./dataset/images`的数据的低秩矩阵 $U \in R^{K \times 16}$ 的每一列的形状来提供图像。

  ⯎ 在报告中提供对应于重建矩阵 $UI \in R^{K\times N}$ 的前20张重建的RGB脸部图像。


3） 我们可以不使用SVD，而是通过非负矩阵分解法（NMF）将非负矩阵 $M \in  R^{K\times N}$ 分解为两个低秩矩阵，即：

$$
M \approx W H
$$

其中 $W\in R^{K \times r}$ 和 $H \in R^{r \times N}$ 是两个非负矩阵，即 $W \geq 0$ 和 $H \geq 0$ 。矩阵W代表捕捉数据特征的基向量，而矩阵H是表示每个基向量对重建原始数据的贡献的权重。NMF中的非负约束允许学习整体数据的部分表征[1, 2]，而这种约束在SVD中是不允许的。为了找到 $M \approx W H$ 的近似解，定义了基于欧氏距离的成本函数来量化近似的质量，即:

$$
Q=\Vert M - W H \Vert_{F}^2 = \sum_{i, j} (M_{ij} - (W H)_{ij})^2
$$

由于成本函数Q在W和H中都是非凸的，所以在求解Q的最小值过程中找到全局最小值是不现实的。一些数值优化技术，如梯度下降和共轭梯度，可以被用来寻找局部最小值。然而，梯度下降的收敛速度很慢，共轭梯度的实现很复杂。此外，基于梯度的方法对步长的参数设置很敏感，这对现实应用来说并不方便。为此，我们提出了W和H的multiplicative update rules，作为收敛速度和实现复杂性之间的折中方案，具体如下:

$$
H_{aj} \leftarrow H_{aj} \frac{W^T M_{aj}}{W^T W H_{aj}}, W_{ia} \leftarrow W_{ia} \frac{M H^T_{ia}}{W H H^T_{ia}}
$$

其中，矩阵W和H可以被随机初始化，每个元素都在[0,255]范围内。如果 $WH_{ij}$ 大于255，那么将 $W H_{ij}$ 设置为255。这样的multiplicative update rules可以保证收敛到一个局部最优的矩阵因式分解。如果你对乘法更新规则背后的理论感兴趣，请参考[3]中的理论证明。请完成以下内容任务。在本实验中，经验参数r被设定为16。你可以在不直接使用NMF API的情况下，对测验1）中构建的矩阵M进行分解。（请提供代码和结果，20%）。

  ⯎ 通过重塑低维矩阵 $W=R^{K \times 16}$ ， 提供报告中`./dataset/images`中的数据的每一列图像。

  ⯎ 在报告中提供与重构矩阵相对应的前20幅重构的RGB脸部图像 $W H \in R^{K \times N}$ 。

4）（可选）对于对NMF的聚类特性感兴趣的同学，可以将NMF应用于脑电信号数据聚类。关于脑电信号聚类的更多背景知识，请参考[这里](https://www.kaggle.com/code/joseguzman/spike-classification-based-on-waveforms/notebook)的Kaggle说明。如果可能，请给出聚类的可视化结果（或任何其他分析结果），其中数据集文件可以从`./dataset/ebs/waveform-5000.csv`中下载。(这是一道奖励题，如果其他测验的答案都正确，则不计入编程作业成绩)。

## 许可证
This project is licensed under the MIT License. See the LICENSE file for more information.

## 参考文献
[1] Lee, Daniel, and H. Sebastian Seung. "Unsupervised learning by convex and conic coding. " Advances in neural information processing systems 9 (1996).

[2] Lee, Daniel D., and H. Sebastian Seung. "Learning the parts of objects by non-negative matrix factorization." Nature 401.6755 (1999): 788-791.

[3] Lee, Daniel, and H. Sebastian Seung. "Algorithms for non-negative matrix factorization. " Advances in neural information processing systems 13 (2000).
