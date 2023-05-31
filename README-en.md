# Matrix_Analysis_SP2023

[English](README-en.md) | [简体中文](README.md)

## Introduction
Programming Assignment for Matrix Analysis, Spring 2023

## Task
### 1. QR Algorithm
1） let $A_1=A$ , perform QR decomposition for $A_k=Q_k R_k$ (e.g., the “qr” function in MATLAB could be used), update
$A_{k+1}=R_k Q_k$, until $A_{k+1}$ is close to an upper triangle matrix. Let us set the tolerance as $\epsilon=10^{-10}$(or please specify $\epsilon$ in each of the following questions if it is set as another small positive number), i.e., if $|a_{ij}^{(k)}| < \epsilon, \forall i > j$ , the iteration stops. Please implement the QR algorithm as an Eigen-Decomposition function and provide the code for the implementation (15%).

2） Consider the following matrices:

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

Please show that the matrix sequence obtained by the QR algorithm { $A_k$ } converges to an upper trianglematrix for the above matrices. (15%)

3）Please compute the eigenvalues and matrix 2-norm condition number (i.e. $cond_2 (𝐴) = \Vert 𝐴_2 \Vert \Vert 𝐴 \Vert_{2}^{-1}$ ) for each of the above matrices. (Please indicate how to calculate the matrix 2-norm condition number for each matrix and provide the results, 20%)

### 2. Matrix Factorization for Data Processing
Due to the complexity and curse of dimensionality, directly processing the high-dimensional data requires a huge amount of computational resources. Non-negative matrix factorization (NMF) is proposed as a technique for dimension reduction, which has found significant applications in image processing. By employing NMF, the non-negative high-dimensional matrix can be decomposed into two non-negative lower-dimensional matrices, one comprises column vectors that can be regarded as basis vectors in the data space, and the other contains rows of coefficients that scale those basis vectors. Moreover, NMF also can be utilized for text data processing. We can examine each column within the coefficient matrix and identify the row number with the maximum coefficient, where the row number indicate the cluster ID of the respective column within the original matrix. Such clustering property means that NMF can be applied for data clustering.

<b>Programming assignment instruction:</b>

1） Please download the dataset from `./dataset/images`, which contains RGB images sampled from the Anime face dataset. According to each color channel, the RGB images in this dataset can be loaded into three single-channel matrices, i.e., $M_R$, $M_G$ and $M_B$, each of them contains 𝐾 = 𝐻 × 𝑊 rows (pixels) and 𝑁′ columns (faces). Please combine the three single-channel matrices into the matrix $M \in R^{𝐾×𝑁}$ , where 𝑁 = 3 × 𝑁′. (Please provide the code, no need to save the matrices into files, 10%)

2）Try to apply the SVD method to factorize the matrix $M \in R^{K\times N}$ into two low-rank matrices, namely matrix $U \in R^{K \times d}$ and matrix $I\in R^{d \times N}$ , where 𝑑 is an empirical parameter in practice which is set as 16 in this experiment. Please complete the following tasks. You can directly apply existing API in this quiz. (Please provide the code and results, 20%)

  ⯎ Provide the singular values of the data from `./dataset/images` in the report.

  ⯎ Provide the images by reshaping each column in the low-rank matrix $U \in R^{K \times 16}$ of the data from `./dataset/images` in the report.

  ⯎ Provide the first 20 reconstructed RGB face images corresponding to the reconstructed matrix $UI \in R^{K\times N}$ in the report.

3）Instead of using SVD, we can decompose the non-negative matrix $M \in  R^{K\times N}$ into two low-rank matrices by the non-negative matrix factorization (NMF) [2], i.e.,

$$
M \approx W H
$$

where $W\in R^{K \times r}$ and $H \in R^{r \times N}$ are two non-negative matrices, i.e., $W \geq 0$ and $H \geq 0$ . The matrix $W$ represents the basis vectors that capture the features of the data, and the matrix $H$ is the weights that indicate the contribution of each basis vector to reconstruct the original data. The non-negative constraints in NMF permit to learn parts representation of the holistic data [1, 2], while such constraint is not permitted in SVD. To find an approximate solution for $M \approx W H$ , the Euclidean distance-based cost function is defined to quantify the quality of the approximation, i.e.,

$$
Q=\Vert M - W H \Vert_{F}^2 = \sum_{i, j} (M_{ij} - (W H)_{ij})^2
$$

Since the cost function 𝑄 is not convex in both 𝑊 and 𝐻, it is unrealistic to find a global minima during solving the minimization of 𝑄 . Several numerical optimization techniques such as gradient descent and conjugate gradient can be applied to find the local minima. However, the gradient descent converges at a slow speed and the conjugate gradient is complicated to implement. Moreover, the gradient-based methods are sensitive to the parameter setting of step size, which is not convenient for realistic applications. To this end, the multiplicative update rules for 𝑊 and 𝐻 are proposed to be a compromise between convergence speed and implementation complexity as follows,

$$
H_{aj} \leftarrow H_{aj} \frac{W^T M_{aj}}{W^T W H_{aj}}, W_{ia} \leftarrow W_{ia} \frac{M H^T_{ia}}{W H H^T_{ia}}
$$

where the matrices 𝑊 and 𝐻 can be randomly initialized with every element in the range of [0, 255]. If (𝑊𝐻)𝑖𝑗 is larger than 255, then (𝑊𝐻)𝑖𝑗 is set as 255. Such multiplicative update rule is guaranteed to converge to a locally optimal matrix factorization. If you are interested in the theory behind the multiplicative update rule, please refer to the theoretical proof in [3]. Please complete the following tasks. The empirical parameter 𝑟 is set as 16 in this experiment. You can decompose the matrix 𝑀 constructed in quiz 1） WITHOUT directly using the NMF API. (Please provide the code and results, 20%)

  ⯎  Provide the images by reshaping each column in the low-dimensional matrix $W=R^{K \times 16}$  of the
data from `./dataset/images` in the report.

  ⯎  Provide the first 20 reconstructed RGB face images corresponding to the reconstructed matrix
 $W H \in R^{K \times N}$ in the report.

4） For students who are interested in the clustering property of the NMF, you may apply the NMF to electrical brain signals data clustering. For more background knowledge about the electrical brain signals clustering, please refer to the Kaggle note here. If possible, please give the visualization results (or any other analytical results) for clustering, where the dataset file can be downloaded from the `./dataset/ebs/waveform-5000.csv`. (This is a bonus question which does not count in the programming assignment grade if answers to all the other quizzes are correct.)

## License
This project is licensed under the MIT License. See the LICENSE file for more information.

## Reference
[1] Lee, Daniel, and H. Sebastian Seung. "Unsupervised learning by convex and conic coding. " Advances in neural information processing systems 9 (1996).

[2] Lee, Daniel D., and H. Sebastian Seung. "Learning the parts of objects by non-negative matrix factorization." Nature 401.6755 (1999): 788-791.

[3] Lee, Daniel, and H. Sebastian Seung. "Algorithms for non-negative matrix factorization. " Advances in neural information processing systems 13 (2000).