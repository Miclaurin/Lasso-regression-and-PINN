# Symbolic-Lasso-regression

我们采用基于参数化字典的物理学动力方程辨识框架，我们主要考虑物理系统的时滞演化，因此可以确定的是微分方程左边为物理变量的一阶时间导数，我们需要构建右边的候选库，将该搜索问题转化为回归问题。

在压缩采样和相关的稀疏性提升方法出现之前，确定非线性动力学系统中的少数非零项将涉及组合蛮力搜索，然而，强大的新理论保证了稀疏解可以使用凸方法以高概率确定，这些凸方法确实可以很好地扩展问题的大小。由此产生的非线性模型识别本质上平衡了模型复杂性（即右侧动力学的稀疏性）与准确性，并且底层凸优化算法确保该方法适用于大规模问题。

\subsubsection{为什么采用L1正则化}
正则化是防止机器学习模型过拟合的技术，其核心作用包括：控制模型复杂度，通过限制参数大小，避免模型对训练数据过度拟合;提升泛化能力，减少对噪声的敏感性，增强模型对新数据的适应能力。

主要有两种正则化技术，分别是L1正则化（Lasso）与L2正则化（Ridge），在本实验报告所设定的问题而言，我们确定了控制方程在高维非线性函数空间中稀疏的事实，因此我们需要使用正则化技术实现这个假定。

我们以普通线性回归模型为例，目标是最小化均方误差：
\[
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} \left( y_i - \left( a_0 + a_1 x_{i1} + a_2 x_{i2} + \cdots + a_p x_{ip} \right) \right)^2
\]
对于Lasso回归通过引入L1正则化项改进目标函数，形式为：
\[
\text{Minimize:} \quad \frac{1}{2N} \sum_{i=1}^{N} \left( y_i - \left( a_0 + a_1 x_{i1} + a_2 x_{i2} + \cdots + a_p x_{ip} \right) \right)^2 + \lambda \sum_{j=1}^{p} |a_j|
\]
其中：\(\lambda \geq 0\) 为调节参数，控制正则化强度，第二项 \(\lambda \sum |a_j|\) 为L1惩罚项（不包含截距项 \(a_0\)）

对于Ridge回归而言通过引入L2正则化项改进目标函数，形式为：
\[
\text{Minimize:} \quad \frac{1}{2N} \sum_{i=1}^{N} \left( y_i - \left( a_0 + a_1 x_{i1} + a_2 x_{i2} + \cdots + a_p x_{ip} \right) \right)^2 + \lambda \sum_{j=1}^{p} a_j^2
\]
其中：\(\lambda \geq 0\) 为调节参数，控制正则化强度， 第二项 \(\lambda \sum a_j^2\) 为L2惩罚项（不包含截距项 \(a_0\)）

下图直观地展示这两种方法在压缩回归系数方面的表现，不同的正则化强度下，Lasso回归和岭回归的回归系数的变化
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\linewidth]{Lasso_vs_Ridge_系数路径对比.png}
\end{figure}

对于Lasso回归（左图）：你会注意到随着 $\lambda$
 值的增加，越来越多的回归系数变成零。这意味着Lasso回归能够实现特征选择，即它会选择哪些特征是重要的，而忽略其他不重要的特征。 而岭回归（右图）：在岭回归中，所有回归系数都逐渐向零靠近但不会完全变成零。这是因为L2正则化会使回归系数尽可能小，但一般不会完全消除任何一个系数。

 下面的图展示了在不同的 （正则化参数）值下，Lasso回归和岭回归的零回归系数的数量。
 \begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\linewidth]{Zero_Coefficients_Comparison.png}
\end{figure}

对于Lasso回归（左图）: 随着 $\lambda$
 值的增加，回归系数被压缩至零的数量迅速增加。这再次证明了Lasso回归能够进行特征选择，即它会选择哪些特征是重要的，而忽略其他不重要的特征。岭回归（右图）: 在所有的 $\lambda$
 值下，岭回归的回归系数都没有被压缩至零。这表明岭回归会使所有回归系数变得较小，但不会将它们压缩至零。

 因此在这样的情况下，我们使用Lasso回归用于微分方程的发掘，对于ODE,PDE以及隐式微分方程而言，构建了对应的Lasso回归问题。可以通过坐标下降法或最小角度回归法进行Lasso问题求解\cite{ranstam2018lasso}。

 
\subsubsection{常微分方程ODE的识别}
本节我们探讨如何从数据中辨识形如下式的非线性动力系统：
$$ \frac{d}{dt}\mathbf{x}(t) = \mathbf{f}(\mathbf{x}(t))   \quad (2.3.2.1)$$
其中，$\mathbf{x}(t) = [x_1(t), x_2(t), \dots, x_n(t)]^T$ 是系统在时间 $t$ 的状态向量，$\mathbf{f}(\mathbf{x}(t))$ 是一个非线性向量函数，定义了系统的动态演化规律。

核心的洞察在于，绝大多数物理系统的控制方程 $\mathbf{f}(\mathbf{x}(t))$ 在一个由众多候选函数构成的函数空间中通常是稀疏的，即 $\mathbf{f}(\mathbf{x}(t))$ 的每一分量 $f_k(\mathbf{x}(t))$ 仅由少数几项非线性函数构成。例如，洛伦兹系统的动力学方程在多项式函数空间中就显得非常稀疏。近年来，压缩感知和稀疏回归领域的突破性进展使得这一稀疏性假设极具应用价值，因为我们现在有能力从一个庞大的候选函数库中精确识别出那些非零的关键项，而无需进行组合爆炸式的穷举搜索。这种方法确保了能够通过凸优化等高效算法大概率地找到稀疏解。由此产生的稀疏模型辨识框架，在本质上实现了模型复杂度（体现为动力学方程右端项的稀疏度）与模型精度之间的平衡，从而有效避免了对观测数据的过拟合。

为了从数据中确定函数 $\mathbf{f}$，我们首先需要收集系统状态 $\mathbf{x}(t)$ 及其时间导数 $\dot{\mathbf{x}}(t)$ 的时间序列数据。假设我们在多个时间点 $t_1, t_2, \dots, t_m$ 对系统进行采样，可以将这些数据组织成两个矩阵：状态历史矩阵 $\mathbf{X}$ 和对应的导数历史矩阵 $\dot{\mathbf{X}}$（或写作 $\mathbf{X}_t$）：
\[
\begin{array}{c}
\mathbf{X}=\left[\begin{array}{c}
\mathbf{x}^{T}\left(t_{1}\right) \\
\mathbf{x}^{T}\left(t_{2}\right) \\
\vdots \\
\mathbf{x}^{T}\left(t_{m}\right)
\end{array}\right]=\left[\begin{array}{cccc}
x_{1}\left(t_{1}\right) & x_{2}\left(t_{1}\right) & \cdots & x_{n}\left(t_{1}\right) \\
x_{1}\left(t_{2}\right) & x_{2}\left(t_{2}\right) & \cdots & x_{n}\left(t_{2}\right) \\
\vdots & \vdots & \ddots & \vdots \\
x_{1}\left(t_{m}\right) & x_{2}\left(t_{m}\right) & \cdots & x_{n}\left(t_{m}\right)
\end{array}\right] \qquad
\dot{\mathbf{X}}=\left[\begin{array}{c}
\dot{\mathbf{x}}^{T}\left(t_{1}\right) \\
\dot{\mathbf{x}}^{T}\left(t_{2}\right) \\
\vdots \\
\dot{\mathbf{x}}^{T}\left(t_{m}\right)
\end{array}\right]=\left[\begin{array}{cccc}
\dot{x}_{1}\left(t_{1}\right) & \dot{x}_{2}\left(t_{1}\right) & \cdots & \dot{x}_{n}\left(t_{1}\right) \\
\dot{x}_{1}\left(t_{2}\right) & \dot{x}_{2}\left(t_{2}\right) & \cdots & \dot{x}_{n}\left(t_{2}\right) \\
\vdots & \vdots & \ddots & \vdots \\
\dot{x}_{1}\left(t_{m}\right) & \dot{x}_{2}\left(t_{m}\right) & \cdots & \dot{x}_{n}\left(t_{m}\right)
\end{array}\right] .
\end{array}
\]

注意，$\dot{\mathbf{X}}$ 的每一列 $\dot{\mathbf{X}}_k = [\dot{x}_k(t_1), \dots, \dot{x}_k(t_m)]^T$ 代表了状态变量 $x_k$ 的时间导数序列。

接下来，我们构建一个候选函数库 $\Theta(\mathbf{X})$，其列由作用于状态历史矩阵 $\mathbf{X}$ 各列（即各状态变量 $x_j(t)$）的候选非线性函数构成。例如，$\Theta(\mathbf{X})$ 可以包含常数项、多项式项以及三角函数项等：
$$ \Theta(\mathbf{X}) = \begin{bmatrix} | & | & | & | & | & | & | \\ \mathbf{1} & \mathbf{X} & \mathbf{X}^{P_2} & \mathbf{X}^{P_3} & \dots & \sin(\mathbf{X}) & \cos(\mathbf{X}) & \dots \\ | & | & | & | & | & | & | \end{bmatrix} $$
其中，$\mathbf{X}^{P_k}$ 表示 $\mathbf{X}$ 中各状态变量的 $k$ 阶多项式组合。例如，对于一个二维系统 $\mathbf{x}(t) = [x_1(t), x_2(t)]^T$，$\mathbf{X}^{P_2}$ 可能包含的列有 $x_1^2(t_j)$, $x_1(t_j)x_2(t_j)$, $x_2^2(t_j)$ 等在所有时间点 $t_j$ 的取值。一个更具体的 $\mathbf{X}^{P_2}$ 例子：
$$ \mathbf{X}^{P_2}(t_j \text{行示例}) = \begin{bmatrix} x_1^2(t_j) & x_1(t_j)x_2(t_j) & \dots & x_n^2(t_j) \end{bmatrix} $$
（注意：$\mathbf{X}^{P_2}(t_j \text{行示例})$ 表示在 $t_j$ 时刻由状态变量构成的二次项行向量，而整个 $\mathbf{X}^{P_2}$ 是一个矩阵块，其各列代表不同的二次项组合在所有 $m$ 个时刻的取值。）

$\Theta(\mathbf{X})$ 的每一列代表了方程(2.3.2.1)右侧 $f_k(\mathbf{x})$ 的一个潜在候选基函数。我们假设 $f_k(\mathbf{x})$ 可以表示为这些基函数的稀疏线性组合。因此，对于每一个状态变量 $x_k$，其导数 $\dot{x}_k(t)$ 可以近似表示为：
$$ \dot{\mathbf{X}}_k \approx \Theta(\mathbf{X})\xi_k $$
其中 $\xi_k$ 是一个稀疏列向量，其非零元素对应于构成 $\dot{x}_k = f_k(\mathbf{x})$ 的活跃函数项的系数。将所有状态变量的导数组合起来，我们可以得到如下的线性系统：
$$ \dot{\mathbf{X}} \approx \Theta(\mathbf{X})\Xi \quad (2.3.2.2) $$
这里 $\Xi = [\xi_1, \xi_2, \dots, \xi_n]$ 是一个 $p \times n$ 的稀疏系数矩阵， $p$ 是候选库 $\Theta(\mathbf{X})$ 中的函数数量。每一列 $\xi_k$ 决定了第 $k$ 个状态变量 $\dot{x}_k$ 的动力学方程。

一旦通过稀疏回归方法（例如采用 $L_1$ 正则化的最小二乘法）求解得到稀疏系数矩阵 $\Xi$，我们就可以重构出每一行控制方程的符号表达式。对于任意一个状态向量 $\mathbf{x} = [x_1, \dots, x_n]^T$，其第 $k$ 个分量的动力学模型为：
$$ \dot{x}_k(t) = f_k(\mathbf{x}(t)) = \Theta(\mathbf{x}^T)\xi_k \quad (2.3.2.3) $$
此处，$\Theta(\mathbf{x}^T)$ 不再是作用于整个历史数据矩阵 $\mathbf{X}$ 的数值矩阵，而是一个 $1 \times p$ 的行向量，其元素是候选库中的各个符号函数（如 $1, x_1, x_2, x_1^2, x_1x_2, \sin(x_1), \dots$）作用于当前状态向量 $\mathbf{x}^T$ 的结果。因此，整个系统的动力学模型可以表示为：
$$ \dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t)) = \Xi^T (\Theta(\mathbf{x}^T))^T \quad (2.3.2.4) $$
其中 $(\Theta(\mathbf{x}^T))^T$ 是一个 $p \times 1$ 的符号函数列向量，$\Xi^T$ 是一个 $n \times p$ 的系数矩阵转置，最终得到一个 $n \times 1$ 的导数向量 $\dot{\mathbf{x}}(t)$。

综上所述，通过构建合适的候选函数库，我们将发现未知ODE这一复杂的符号搜索问题成功地转化为了一个标准的稀疏回归问题$$\min_{\alpha} \left\|\frac{d}{dt}\mathbf{x}(t)-\Xi^T (\Theta(\mathbf{x}^T))^T\right\|_2^2+\lambda\left\|\Xi^T\right\|_1$$进而可以利用如LASSO等成熟的 $L_1$ 正则化技术高效求解稀疏系数矩阵 $\Xi$，从而辨识出系统的控制方程。 
\subsubsection{偏微分方程PDE的识别}
对于许多复杂的物理现象，若其控制方程难以通过第一性原理直接推导，或推导过程不切实际，那么利用实验或观测数据进行数据驱动的方程发现则提供了另一种有效的途径。在本节中，我们聚焦于如何从数据中辨识出由偏微分方程（PDE）描述的物理过程。

我们考虑一个一般的PDE，其动态主要由状态变量 $u$ 的时间导数 $\frac{\partial u}{\partial t}$ 决定。假设这个时间导数可以表示为一系列候选函数项的线性组合。这些候选函数项通常由状态变量 $u$ 本身、其空间导数（如 $\frac{\partial u}{\partial x}, \frac{\partial^2 u}{\partial x^2}, \frac{\partial u}{\partial y}$ 等，具体取决于空间维度）以及可能的外部源项 $Q$ 组合而成。因此，一个广义的PDE可以写作：
$$ \frac{\partial u}{\partial t} = \mathbf{\Phi}(u, Q, \nabla u, \nabla^2 u, \dots) \alpha \quad (2.3.3.1) $$
其中，$u$ 可以是 $u(t, x_1, x_2, \dots, x_d)$，$d$ 是空间维度。$\mathbf{\Phi}$ 是一个包含 $p$ 个候选函数项的行向量（或在后续矩阵形式中表示为一个库矩阵的对应行），例如：
$$ \mathbf{\Phi}(u, Q, \dots) = \left[ 1, u, Q, \frac{\partial u}{\partial x}, \frac{\partial^2 u}{\partial x^2}, \frac{\partial^3 u}{\partial x^3}, u\frac{\partial u}{\partial x}, u^2, \left(\frac{\partial u}{\partial x}\right)^2, \dots \right] \quad (2.3.3.2) $$
$\alpha$ 是一个包含 $p$ 个对应系数的列向量。一个核心的假设是，对于大多数物理系统，其真实的PDE在这样一个扩展的候选函数库中是稀疏的，即向量 $\alpha$ 中只有少数元素是非零的，这些非零元素对应着构成真实PDE的关键项。

当拥有离散的实验或模拟数据时，我们的目标便是利用这些数据来学习并确定稀疏系数向量 $\alpha$。假设数据是在 $m$ 个空间采样点 $(\mathbf{x}^{(i)} = [x_1^{(i)}, x_2^{(i)}, \dots, x_d^{(i)}]$，$i=1, \dots, m$）和 $n$ 个时间点 $t_j$（$j=1, \dots, n$）上获得的。总的数据点数 $N_d = m \times n$。由于PDE (2.3.3.1) 在每个时空点上都应成立，我们可以构建一个大型的线性代数系统。

令 $\frac{\partial U}{\partial t}$ 是一个 $N_d \times 1$ 的列向量，其元素为在所有 $N_d$ 个时空采样点上计算得到的 $u$ 的时间导数值 $\left(\frac{\partial u}{\partial t}\right)\Big|_{\mathbf{x}^{(i)}, t_j}$。
令 $\mathbf{\Phi}_{\text{matrix}}$ 是一个 $N_d \times p$ 的矩阵，其每一行是在对应的时空采样点 $(\mathbf{x}^{(i)}, t_j)$ 上对候选函数库 $\mathbf{\Phi}$ 中所有 $p$ 个函数项求值的结果。例如，第 $k$-行，第 $l$-列的元素是第 $l$ 个候选函数在第 $k$ 个时空采样点上的值。
于是，方程(2.3.3.1)的离散形式可以写作：
\[ \frac{\partial U}{\partial t} \approx \Phi_{\text{matrix}} \alpha \quad (2.3.3.3) \]
展开就是\[\begin{bmatrix}\frac{\partial u}{\partial t}(x_1,t_1)\\\frac{\partial u}{\partial t}(x_2,t_1)\\\vdots\\\frac{\partial u}{\partial t}(x_m,t_1)\\\vdots\\\frac{\partial u}{\partial t}(x_1,t_n)\\\frac{\partial u}{\partial t}(x_2,t_n)\\\vdots\\\frac{\partial u}{\partial t}(x_m,t_n)\end{bmatrix}=\begin{bmatrix}1&u(x_1,t_1)&\cdots&\frac{\partial u}{\partial x}(x_1,t_1)&\cdots\\1&u(x_2,t_1)&\cdots&\frac{\partial u}{\partial x}(x_2,t_1)&\cdots\\\vdots&\vdots&\ddots&\vdots&\ddots\\1&u(x_m,t_1)&\cdots&\frac{\partial u}{\partial x}(x_m,t_1)&\cdots\\\vdots&\vdots&\ddots&\vdots&\ddots\\1&u(x_1,t_n)&\cdots&\frac{\partial u}{\partial x}(x_1,t_n)&\cdots\\1&u(x_2,t_n)&\cdots&\frac{\partial u}{\partial x}(x_2,t_n)&\cdots\\\vdots&\vdots&\ddots&\vdots&\ddots\\1&u(x_m,t_n)&\cdots&\frac{\partial u}{\partial x}(x_m,t_n)&\cdots\end{bmatrix}\begin{bmatrix}\alpha_1\\\alpha_2\\\vdots\\\vdots\\\vdots\\\vdots\\\vdots\\\vdots\\\vdots\\\vdots\end{bmatrix}\]

从数据中学习PDE的本质，就是求解这个超定线性系统以获得稀疏的系数向量 $\alpha$。需要强调的是，要构建上述系统，必须从离散的、可能含有噪声的原始数据 $u(\mathbf{x}^{(i)}, t_j)$ 中准确地计算出时间导数 $\frac{\partial U}{\partial t}$ 以及包含在 $\Phi_{\text{matrix}}$ 中的各项空间导数（如 $\frac{\partial u}{\partial x}, \frac{\partial^2 u}{\partial x^2}$ 等）。这些数值微分过程是数据预处理的关键步骤，其精度直接影响最终PDE发现的准确性（具体方法参考本报告第3章节）。

为了找到稀疏解 $\alpha$，我们采用LASSO回归，其目标是最小化以下包含 $L_1$ 正则化项的损失函数：
$$ \min_{\alpha} \left\| \frac{\partial U}{\partial t} - \Phi_{\text{matrix}} \alpha \right\|_{2}^{2} + \lambda \|\alpha\|_{1} \quad (2.3.3.4) $$
其中，$\| \cdot \|_{2}^{2}$ 表示欧几里得范数的平方（即残差平方和），$\|\alpha\|_{1} = \sum_{k=1}^{p} |\alpha_k|$ 是系数向量的 $L_1$ 范数。$\lambda > 0$ 是一个正则化参数，它控制着解的稀疏程度：$\lambda$ 越大，得到的 $\alpha$ 向量中非零元素的数量就越少。通过调节 $\lambda$（例如结合信息准则或交叉验证，如2.3.5节所述），可以选择出既能良好拟合数据又具有稀疏性的最优模型。


\subsubsection{SINDy-PI框架：隐式动力学的鲁棒性识别}
标准的SINDy方法主要适用于显式常微分方程（ODE）$\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})$ 的辨识。然而，许多物理系统，特别是那些包含有理函数非线性项（例如具有多时间尺度特征的化学反应网络或代谢网络）的系统，其动力学更自然地由隐式微分方程描述，形式通常为：
$$ \mathbf{f}(\mathbf{x}, \dot{\mathbf{x}}) = \mathbf{0} \quad (2.3.4.1) $$
为了处理此类系统，隐式SINDy (implicit-SINDy) 算法被提出（相关实现可见于 PySINDy 软件包）。该方法将候选函数库从仅包含状态变量 $\mathbf{X}$ 的函数 $\Theta(\mathbf{X})$ 推广到同时包含状态变量 $\mathbf{X}$ 及其导数 $\dot{\mathbf{X}}$ 的函数所构成的库，记为 $\Theta(\mathbf{X}, \dot{\mathbf{X}})$。其目标是找到一个稀疏系数矩阵 $\Xi$，使得：
$$ \Theta(\mathbf{X}, \dot{\mathbf{X}}) \Xi \approx \mathbf{0} \quad (2.3.4.2) $$
理想情况下，$\Xi$ 的每一列 $\xi_k$ 都是稀疏的，并且位于矩阵 $\Theta(\mathbf{X}, \dot{\mathbf{X}})$ 的零空间（null space）中。然而，这种基于零空间计算的方法存在一个主要缺陷：它对测量数据中的噪声高度敏感。即使是很小的噪声也可能显著改变零空间的结构，从而导致模型辨识失败或结果不可靠。此外，寻找最稀疏的零空间向量是一个非凸优化问题，通常依赖于如交替方向乘子法（ADM）等迭代算法，其收敛性和解的质量也可能受到影响。

针对隐式SINDy对噪声的敏感性问题，Kaheman等人提出了SINDy-PI框架\cite{Kaheman_2020}，旨在实现对隐式动力学的鲁棒辨识。SINDy-PI的核心创新在于巧妙地规避了直接的零空间计算。其基本思想是：如果我们预先知道真实隐式方程(2.3.4.1)中的某一项（例如，库 $\Theta(\mathbf{x}, \dot{\mathbf{x}})$ 中的第 $j$ 个候选函数 $\theta_j(\mathbf{x}, \dot{\mathbf{x}})$）是方程的组成部分，那么我们就可以将该项移到等式的左边，从而将隐式问题转化为一个显式SINDy问题。具体地，对于数据矩阵 $\mathbf{X}$ 和 $\dot{\mathbf{X}}$，方程可以重写为：
$$ \theta_j(\mathbf{X}, \dot{\mathbf{X}}) \approx \Theta(\mathbf{X}, \dot{\mathbf{X}} | \theta_j) \xi_j \quad (2.3.4.3) $$
其中，$\theta_j(\mathbf{X}, \dot{\mathbf{X}})$ 是一个列向量，表示候选函数 $\theta_j$ 在所有数据点上的取值。$\Theta(\mathbf{X}, \dot{\mathbf{X}} | \theta_j)$ 代表从完整的候选函数库矩阵中移除了与 $\theta_j$ 对应的列之后所形成的子库矩阵。$\xi_j$ 是一个新的稀疏系数向量，对应于子库 $\Theta(\mathbf{X}, \dot{\mathbf{X}} | \theta_j)$ 中的各项。

由于我们事先并不知道哪一项 $\theta_j$ 真正存在于动力学方程中，SINDy-PI采用一种``轮流坐庄''的策略：它迭代地选择库 $\Theta(\mathbf{X}, \dot{\mathbf{X}})$ 中的每一个候选函数项 $\theta_j$ 作为等式(2.3.4.3)的左端项，然后针对右端的子库 $\Theta(\mathbf{X}, \dot{\mathbf{X}} | \theta_j)$ 求解稀疏系数向量 $\xi_j$。这个求解过程可以通过最小化如下的损失函数（通常采用 $L_0$ 范数或其 $L_1$ 范数松弛形式）来实现：
$$ \min_{\xi_j} \left\| \theta_j(\mathbf{X}, \dot{\mathbf{X}}) - \Theta(\mathbf{X}, \dot{\mathbf{X}} | \theta_j) \xi_j \right\|_2^2 + \lambda \|\xi_j\|_0 \quad (2.3.4.4) $$
其中 $\lambda$ 是促进稀疏性的正则化参数。当使用 $L_1$ 范数（$\|\xi_j\|_1$）替代 $L_0$ 范数时，问题变为凸优化问题，可以使用如序列阈值最小二乘法（STLSQ）或者如第3章介绍的ADMM等高效算法求解。由于该方法将原先的隐式、对噪声敏感的零空间问题转化为了一系列显式的、更为鲁棒的稀疏回归问题，SINDy-PI在处理含噪数据时表现出显著的性能提升。

在通过SINDy-PI得到一系列候选模型后，模型选择成为关键步骤。目标是找到既能准确描述数据，又具备简洁（稀疏）结构的方程。Kaheman等人 \cite{Kaheman_2020}讨论了多种模型选择策略，例如基于在留存测试集 $\mathbf{X}_t, \dot{\mathbf{X}}_t$ 上的预测误差。对于一般的隐式模型，其拟合误差可以定义为（对应Kaheman论文Eq.12）：
$$ \text{Error} = \frac{\|\theta_j(\mathbf{X}_t, \dot{\mathbf{X}}_t) - \Theta(\mathbf{X}_t, \dot{\mathbf{X}}_t | \theta_j) \hat{\xi}_j \|_2}{\|\theta_j(\mathbf{X}_t, \dot{\mathbf{X}}_t)\|_2} \quad (2.3.4.5) $$
其中 $\hat{\xi}_j$ 是在训练集上得到的估计系数。
特别地，对于那些状态导数可以表示为有理函数形式的系统，即 $\dot{x}_k = N_k(\mathbf{x}) / D_k(\mathbf{x})$（对应Kaheman论文Eq.14），一种更直接的误差度量是基于对导数 $\dot{\mathbf{X}}_t$ 的预测精度（对应Kaheman论文Eq.13）：
$$ \text{Error} = \frac{\|\dot{\mathbf{X}}_t - \dot{\mathbf{X}}_t^{\text{model}}\|_2}{\|\dot{\mathbf{X}}_t\|_2} \quad (2.3.4.6) $$
其中 $\dot{\mathbf{X}}_t^{\text{model}}$ 是由辨识出的模型（即由 $\theta_j(\mathbf{X}_t, \dot{\mathbf{X}}_t^{\text{model}}) = \Theta(\mathbf{X}_t, \dot{\mathbf{X}}_t^{\text{model}} | \theta_j) \hat{\xi}_j$ 隐式定义的导数）所预测的导数值。选择使得这类误差最小且结构稀疏的模型是最终的目标。

\subsubsection{ 正则化参数 $\lambda$ 的最优选择的自适应策略}

在2.3.2至2.3.4节所讨论的基于稀疏回归的动力系统辨识方法中，无论是用于常微分方程（ODE）、偏微分方程（PDE）还是隐式方程，正则化参数 $\lambda$（在某些算法如STLSQ中表现为阈值）的选择都至关重要。参数 $\lambda$ 直接控制着模型最终的稀疏程度：一个过小的 $\lambda$ 可能导致模型包含过多不必要的项，从而引发过拟合，使得模型在训练数据上表现优异但在新数据上泛化能力差；而一个过大的 $\lambda$ 则可能过度惩罚模型复杂度，导致模型过于简化，遗漏了关键的物理项，从而产生欠拟合。

手动逐个尝试不同的 $\lambda$ 值并评估模型性能无疑是低效且耗时的，且难以保证找到全局最优的平衡点。因此，需要系统性的方法来自动化地选择最优的 $\lambda$。常用的策略主要包括基于信息准则的方法和基于交叉验证的方法。

\subsubsection*{1. 基于信息准则（AIC/BIC）的 $\lambda$ 选择}

赤池信息准则（AIC）和贝叶斯信息准则（BIC）（已在2.2节详细介绍）提供了一种在模型拟合优度与模型复杂度之间进行权衡的有效手段。其应用于 $\lambda$ 选择的基本流程如下：
\begin{itemize}
    \item 候选 $\lambda$ 序列：首先，定义一个候选 $\lambda$ 值的序列（通常是对数等间隔或根据经验选择的一系列值）。
    \item 模型拟合与评估：对于序列中的每一个 $\lambda_i$ 值，使用相应的稀疏回归求解器（如LASSO的坐标下降法、最小角度回归法，或SINDy中的STLSQ算法）拟合动力学模型。对每一个拟合得到的模型，计算其：
    \begin{itemize}
        \item \拟合优度：通常用残差平方和（RSS）或均方误差（MSE）来衡量模型在训练数据上的拟合程度。
        \item 模型复杂度 ($k$)：在线性模型（如$\dot{\mathbf{X}} = \Theta(\mathbf{X})\Xi$）的背景下，$k$ 通常指模型中非零系数的个数。对于更广义的符号回归或复杂模型，它可以是其他复杂度度量（如PySR中的节点数）。
    \end{itemize}
    \item 计算信息准则值：利用得到的拟合优度指标和模型复杂度 $k$，计算每个模型的AIC和BIC值。例如，基于MSE的常用形式为：
    $$ \text{AIC} = n \ln(\text{MSE}) + 2k $$
    $$ \text{BIC} = n \ln(\text{MSE}) + k \ln(n) $$
    其中 $n$ 是样本量。
    \item 选择最优 $\lambda$：选择使得AIC值或BIC值最小的那个模型所对应的 $\lambda_i$ 作为最优正则化参数。通常，BIC由于其对复杂度的惩罚项中包含 $\ln(n)$，在大样本情况下倾向于选择更简洁的模型。您的论文图16清晰地展示了AIC和BIC值随 $(-\log(\lambda))$ 变化的曲线，曲线的谷底即对应了这两个准则下的最优 $\lambda$ 估计。
\end{itemize}

\subsubsection*{2. 基于交叉验证的 $\lambda$ 选择}

交叉验证是一种更为直接评估模型泛化性能的方法，它不依赖于对模型复杂度或数据分布的特定假设。$K$-折交叉验证是其中最常用的技术：
\begin{itemize}
    \item 数据划分：将原始训练数据集随机划分为 $K$ 个互不相交的大小相似的子集（“折”）。
    \item 模型训练与验证迭代：对于每一个候选的 $\lambda_j$ 值：
    \begin{enumerate}
        \item 进行 $K$ 次迭代。在第 $k$ 次迭代中（$k=1, \dots, K$）：
        \begin{itemize}
            \item 将第 $k$ 个子集作为验证集（validation set）。
            \item 将其余 $K-1$ 个子集合并作为该次迭代的训练集（。
            \item 使用当前 $\lambda_j$ 在这个训练集上拟合模型。
            \item 在验证集上评估模型的预测误差（例如，计算MSE）。
        \end{itemize}
        \item 计算这 $K$ 次迭代得到的预测误差的平均值，作为该 $\lambda_j$ 值下的交叉验证误差 $CV(\lambda_j)$。
    \end{enumerate}
    \item 选择最优 $\lambda$：选择使得 $CV(\lambda_j)$ 最小的 $\lambda_j$ 作为最优正则化参数。这种方法旨在找到在新数据上预期表现最好的模型。
    \item “一倍标准误规则”：在实践中，有时会采用此规则来选择一个更简洁的模型。首先找到交叉验证误差最小的模型（对应 $\lambda_{\text{minCV}}$），然后计算该最小误差的一个标准误。接着，在所有交叉验证误差小于（最小误差 + 一个标准误）的模型中，选择那个最简洁（即对应 $\lambda$ 值最大）的模型。这通常能在不显著牺牲预测性能的前提下，获得更稀疏的模型。
\end{itemize}

综上所述，无论是基于信息准则还是交叉验证，这些系统性的方法都能够有效地帮助我们从一系列候选中确定一个合适的正则化参数 $\lambda$，从而在所发现的动力学方程的稀疏性（简洁性）与对数据的拟合精度之间取得理想的平衡。这对于从实验数据中可靠地挖掘物理规律至关重要。
\subsection{数值微分问题描述}
给定一组 m 个离散数据点（观测值）$\{(x_i,y_i)\}$，假定他们之间的映射关系为$g:[a,b]\to\mathbb{R}$。
\[a\leq x_1\leq x_2\leq\cdots\leq x_m\leq b\]
我们希望构造一个近似 U 到一阶导数 G′。微分问题是计算积分问题的倒数。与大多数逆问题一样，它是病态的，因为解$ u\approx g'$ 并不连续依赖于 g。如果我们对导数值直接使用有限差分近似值，那么数据中任意小的相对扰动都可能导致任意大的相对变化，离散问题是病态的。因此，我们需要某种形式的正则化以避免过拟合。除了构造导数 u 之外，我们还计算了一个近似于 g 的平滑函数 f。如果直接从数据计算 u，则可能会将其积分以生成 f 。或者，我们可以将问题视为构造一个平滑近似 f，然后可以对其进行微分以获得$ u = f '$。
\subsection{为什么选择Total variation regularization进行数值微分}
选择总变差正则化（TVRegDiff）作为本报告中主要的数值微分工具，是基于其在处理含噪实验数据方面的几项关键优势。实验数据，尤其是通过传感器采集或图像追踪技术间接获得的数据，往往伴随着一定程度的噪声。传统的数值微分方法，如基于局部多项式拟合的有限差分法（例如前向、后向或中心差分），虽然在理论上对光滑无噪声函数是有效的，但在实际应用中对噪声非常敏感。这些方法在计算导数时，倾向于放大信号中的高频噪声成分，可能导致计算得到的导数序列充满伪影，甚至完全掩盖真实的物理动态信息，从而对后续的物理规律发现过程（如SINDy辨识）造成严重干扰。

相比之下，总变差正则化提供了一种更为鲁棒的解决方案。其核心优势在于著名的“保边去噪”（edge-preserving denoising）特性。TV正则化项，如式(3.3.1)中的 $\lambda \sum_{j} |(\mathbf{D}\mathbf{x})_j|$，通过最小化信号导数的一阶差分的$L_1$范数，倾向于产生分段常数或分段平滑的解。这意味着TVRegDiff能够在有效抑制噪声的同时，较好地保持原始信号中可能存在的锐利边缘、阶跃或其他重要的非平滑特征，而这些特征对于理解物理过程的动态行为可能至关重要。其他一些线性平滑方法，如高斯滤波或宽窗口的移动平均，虽然也能去除噪声，但往往以牺牲信号细节、模糊边缘为代价。

与依赖于局部多项式假设的Savitzky-Golay等滤波器相比，TVRegDiff对信号的局部光滑性要求较低，对于不完全符合多项式模型的物理信号具有更好的适应性。此外，虽然傅里叶变换求导等方法在特定条件下（如周期信号、高信噪比）表现优异，但它们对信号的全局特性和噪声频谱分布有较强依赖，且可能引入如吉布斯振铃等问题。

TVRegDiff方法将数值微分问题构建为一个定义良好的凸优化问题（或可通过ADMM等算法高效求解的近似问题），能够从理论上保证解的稳定性和唯一性（给定正则化参数$\lambda$）。众多采用SINDy框架进行数据驱动发现的研究均证实了TVRegDiff及其变体在从含噪数据中提取可靠导数方面的有效性。因此，在本报告的实验数据处理中，我们选择TVRegDiff作为获取状态变量各阶导数的主要手段，以期为后续的动力学方程辨识提供高质量的输入数据。接下来，我们将简要介绍其原理及常用的求解算法。
\subsection{总变差正则化（TVRegDiff）}
我们假设 $g(a)$ 是给定的（在阻尼振动实验中，g(a)相当于已知的初始条件），并且 g 在索博列夫空间 $H^{k+1}(a,b)）$ 中。我们将问题表述为计算算子方程的近似解 $u \in H^k(a， b)$，即\[Au(x)=\int_a^xu(t)dt=\hat{g}(x),x\in[a,b]\]
其中$\hat{g}(x)=g(x)-g(a)$，我们用均匀网格中点处离散值的向量$ \mathbf{u} $来表示 u：\[u_j=f^{\prime}(a+(j-1)\Delta t+\Delta t/2),\quad(j=1,\ldots,n)\]
对于$ \Delta t = (b − a)/n$。离散化系统由 n 个未知数$A\mathbf{u}=\hat{y}$中的 m 个方程组成，其中$ A_{ij} $的大小是 $[a, x_i] \cap [t_j, t_{j+1}] $并且$\hat y_i = y_i − g(a)$ 。因此我们需要最小化的能量泛函是
\[E(\mathbf{u})=\frac{1}{2}\|A\mathbf{u}-\hat{\mathbf{y}}\|^{2}+\alpha\sum_{j=1}^{n-1}\sqrt{(u_{j+1}-u_{j})^{2}+\epsilon}\]其中 $‖ ·‖$ 表示欧几里得范数，$\alpha$ 是非负正则化参数。正则化项是 BV 半范数 $F (u) = \int_a^b |u'|$使用梯形法则和自然结束条件：$u'(a) = u'(b) = 0$。 由于 F 的梯度为 $\nabla F (u) = −(u'/|u'|)'$,当$u'=0$时，小正数$\epsilon$的扰动对于 E 的微分性是必需的 ,考虑到小正数的存在，理论上这里我们使用广义 Sobolev 梯度方法来最小化能量泛函

E 的梯度为\[\nabla E(\mathbf{u})=A^t(A\mathbf{u}-\hat{\mathbf{y}})-\alpha\mathbf{s}^{\prime}\]
其中符号向量 s 是 $u'/|u'|$的近似值：\[s_i=(u_{i+1}-u_i)/\sqrt{(u_{i+1}-u_i)^2+\epsilon}\quad(i=1,\ldots,n-1)\]而且满足\[(s^{\prime})_1=s_1,\quad(s^{\prime})_i=s_i-s_{i-1}\quad(i=2,\ldots,n-1),\quad(s^{\prime})_n=-s_{n-1}\]
E 在 $\mathbf{u}$ 处的 Hessian 矩阵是\[H(\mathbf{u})=A^tA+\alpha D_u^t(I-\mathrm{diag}(\mathbf{s})^2)D_u\]其中 $D_u$ 是将$ v $映射到 $v'/\sqrt{|u'|}$的离散化算符，并且$1-s_i^2=\epsilon/((u_{i+1}-u_i)^2+\epsilon)$

我们通过预调节器近似$ H(u)$:\[A_u=A^tA+\alpha\Delta tD_u^tD_u\]索博列夫梯度 $Au^{-1}\nabla E(u)$ 是与加权索博列夫内积相关的梯度\[\langle\mathbf{v},\mathbf{w}\rangle_{u}=(A\mathbf{v})^{t}(A\mathbf{w})+\alpha\Delta t(D_{u}\mathbf{v})^{t}(D_{u}\mathbf{w})\]
因此最陡的下降迭代是\[\mathbf{u}_{k+1}=\mathbf{u}_k-\beta A_{\boldsymbol{u}}^{-1}\nabla E(\mathbf{u}_{\mathbf{k}})\]具有恒定的步长 $\beta$，初始值 $u_0 $通过差分 y 值获得。由于步骤 k 的梯度和内积取决于$ u_k$，因此这是一种可变度量方法。所以f 的网格点值计算为\[f_j=\Delta t\sum_{i=1}^{j-1}u_i+g(a)\qquad(j=1,\ldots,n+1).\]


\subsection{对于本报告中实验的TVR方案}

为增强SINDy等数据驱动方法在从含噪实验数据中发掘控制方程时的鲁棒性，对原始数据进行恰当的预处理，特别是精确计算其各阶导数，是至关重要的环节。考虑到实验数据往往包含噪声且可能非光滑，直接使用标准的有限差分法计算导数会将噪声放大，严重影响后续模型辨识的精度和可靠性。因此，在本报告涉及的各实验中，我们优先采用总变差正则化（Total Variation Regularization Difference, TVRegDiff）方法来获取或平滑所需的导数值。该方法旨在找到一个导数序列，它既能较好地逼近由原始数据简单差分得到的初始导数估计，又具有自身的分段光滑特性（通过最小化其全变差实现）。

对于一个给定的离散信号向量 $\mathbf{s}$（例如时间序列数据，如位移 $x(t)$、角度 $\theta(t)$ 或温度 $T(t, \mathbf{p})$ 在固定空间点 $\mathbf{p}$ 的时间演化），我们首先计算其一个初始的、可能含噪的导数估计 $\mathbf{s}'$（例如，通过前向差分 $\mathbf{s}'_i = (s_{i+1}-s_i)/\Delta t$，并在末尾补零或采用其他边界处理）。TVRegDiff的目标是寻找到一个平滑的导数向量 $\mathbf{x}$（代表 $\dot{s}$ 或其他阶导数），该向量通过最小化如下形式的能量泛函（或目标函数）$E(\mathbf{x})$ 得到：
$$ E(\mathbf{x}) = \frac{1}{2} \| \mathbf{x} - \mathbf{s}' \|_2^2 + \lambda \sum_{j} |(\mathbf{D}\mathbf{x})_j| \quad (3.3.1) $$
其中，$\| \cdot \|_2^2$ 表示欧几里得范数的平方，$\mathbf{D}$ 是一阶差分算子矩阵，使得 $(\mathbf{D}\mathbf{x})_j = x_{j+1} - x_j$ （或其转置，取决于定义）。$\lambda \ge 0$ 是正则化参数，用以权衡数据保真项（第一项，即平滑导数 $\mathbf{x}$ 与初始导数估计 $\mathbf{s}'$ 的接近程度）与解的平滑度（第二项，即导数序列 $\mathbf{x}$ 的全变差）。此类优化问题可通过交替方向乘子法（ ADMM）等高效算法求解。

\subsubsection*{ADMM求解TVRegDiff问题概述}
式 (3.3.1) 所描述的优化问题，由于其包含 $L_1$ 范数正则化项（即 $\lambda \sum_{j} |(\mathbf{D}\mathbf{x})_j|$），导致目标函数非平滑。ADMM通过引入辅助变量和增广拉格朗日量的概念，将原问题分解为若干个更易于求解的子问题。
为了应用ADMM，我们将式 (3.3.1) 中的 $L_1$ 正则化项通过引入一个辅助变量 $\mathbf{z}$ 来解耦。令 $\mathbf{z} = \mathbf{D}\mathbf{x}$，则原问题可以等价地表述为：
\begin{align*}
    \text{minimize}_{\mathbf{x},\mathbf{z}} \quad & \frac{1}{2} \| \mathbf{x} - \mathbf{s}' \|_2^2 + \lambda \|\mathbf{z}\|_1 \\
    \text{subject to} \quad & \mathbf{D}\mathbf{x} - \mathbf{z} = \mathbf{0}
\end{align*}
该约束优化问题的增广拉格朗日函数为：
$$ \mathcal{L}_{\rho}(\mathbf{x}, \mathbf{z}, \mathbf{u}) = \frac{1}{2} \| \mathbf{x} - \mathbf{s}' \|_2^2 + \lambda \|\mathbf{z}\|_1 + \mathbf{u}^T (\mathbf{D}\mathbf{x} - \mathbf{z}) + \frac{\rho}{2} \| \mathbf{D}\mathbf{x} - \mathbf{z} \|_2^2 $$
其中，$\mathbf{u}$ 是拉格朗日乘子向量（对偶变量），$\rho > 0$ 是增广拉格朗日惩罚参数。
ADMM算法通过以下迭代步骤 $(k=0, 1, 2, \dots)$ 来交替更新变量 $\mathbf{x}$, $\mathbf{z}$, 和 $\mathbf{u}$ 直至收敛：
\begin{enumerate}
    \item \textbf{$\mathbf{x}$-更新步}：固定 $\mathbf{z}^{(k)}$ 和 $\mathbf{u}^{(k)}$，求解关于 $\mathbf{x}$ 的最小化问题：
    $$ \mathbf{x}^{(k+1)} = \arg\min_{\mathbf{x}} \left( \frac{1}{2} \| \mathbf{x} - \mathbf{s}' \|_2^2 + \frac{\rho}{2} \left\| \mathbf{D}\mathbf{x} - \mathbf{z}^{(k)} + \frac{1}{\rho}\mathbf{u}^{(k)} \right\|_2^2 \right) $$
    这对应于求解线性系统 $(\mathbf{I} + \rho \mathbf{D}^T\mathbf{D})\mathbf{x} = \mathbf{s}' + \rho \mathbf{D}^T(\mathbf{z}^{(k)} - \frac{1}{\rho}\mathbf{u}^{(k)})$。

    \item \textbf{$\mathbf{z}$-更新步}：固定 $\mathbf{x}^{(k+1)}$ 和 $\mathbf{u}^{(k)}$，求解关于 $\mathbf{z}$ 的最小化问题：
    $$ \mathbf{z}^{(k+1)} = \arg\min_{\mathbf{z}} \left( \lambda \|\mathbf{z}\|_1 + \frac{\rho}{2} \left\| \mathbf{D}\mathbf{x}^{(k+1)} - \mathbf{z} + \frac{1}{\rho}\mathbf{u}^{(k)} \right\|_2^2 \right) $$
    其解通过软阈值算子 \cite{donoho2002noising} 给出 。令 $\mathbf{v}^{(k)} = \mathbf{D}\mathbf{x}^{(k+1)} + \frac{1}{\rho}\mathbf{u}^{(k)}$，则 $ z_j^{(k+1)} = \text{sgn}(v_j^{(k)}) \max(0, |v_j^{(k)}| - \lambda/\rho) $。

    \item \textbf{$\mathbf{u}$-更新步（对偶变量更新）}：
    $$ \mathbf{u}^{(k+1)} = \mathbf{u}^{(k)} + \alpha \rho (\mathbf{D}\mathbf{x}^{(k+1)} - \mathbf{z}^{(k+1)}) $$
    其中 $\alpha$ 是松弛参数。
\end{enumerate}
迭代直至满足收敛准则 。

\subsubsection*{3.4.1 阻尼振动实验}
在阻尼振动实验中，我们采集了物块位移 $x(t)$ 的时间序列。为辨识其控制方程，需要计算速度 $\dot{x}(t)$ 和加速度 $\ddot{x}(t)$。
\begin{enumerate}
    \item \textbf{速度 $\dot{\mathbf{x}}(t)$ 的计算}：
    令原始位移数据为 $\mathbf{x}_{\text{data}}$，其简单差分导数估计为 $\mathbf{x}'_{\text{data}}$。我们求解平滑的速度 $\dot{\mathbf{x}}$，其最小化目标函数为：
    $$ E(\dot{\mathbf{x}}) = \frac{1}{2} \| \dot{\mathbf{x}} - \mathbf{x}'_{\text{data}} \|_2^2 + \lambda_{\dot{x}} \sum_{j} |(\mathbf{D}\dot{\mathbf{x}})_j| $$
    该问题通过上述ADMM算法求解。
    \item \textbf{加速度 $\ddot{\mathbf{x}}(t)$ 的计算}：
    令上一步得到的平滑速度为 $\dot{\mathbf{x}}_{\text{smooth}}$，其简单差分导数估计为 $\dot{\mathbf{x}}'_{\text{smooth}}$。我们求解平滑的加速度 $\ddot{\mathbf{x}}$，其最小化目标函数为：
    $$ E(\ddot{\mathbf{x}}) = \frac{1}{2} \| \ddot{\mathbf{x}} - \dot{\mathbf{x}}'_{\text{smooth}} \|_2^2 + \lambda_{\ddot{x}} \sum_{j} |(\mathbf{D}\ddot{\mathbf{x}})_j| $$
    同样使用ADMM算法求解，其中正则化参数 $\lambda_{\ddot{x}}$ 可能与 $\lambda_{\dot{x}}$ 不同。
\end{enumerate}

\subsubsection*{3.4.2 双摆实验}
双摆系统的状态由两个角度 $\theta_1(t), \theta_2(t)$ 及其角速度 $\dot{\theta}_1(t), \dot{\theta}_2(t)$ 和角加速度 $\ddot{\theta}_1(t), \ddot{\theta}_2(t)$ 描述。若实验仅测量角度数据，则需进行数值微分：
\begin{enumerate}
    \item \textbf{角速度 $(\dot{\theta}_1, \dot{\theta}_2)$ 的计算}：
    对于每个摆角 $\theta_k(t)$ ($k=1,2$)，令其原始数据为 $\Theta_{k,\text{exp}}$，其简单差分导数估计为 $\Theta'_{k,\text{exp}}$。我们求解平滑的角速度 $\dot{\Theta}_k$，其最小化目标函数为：
    $$ E(\dot{\Theta}_k) = \frac{1}{2} \| \dot{\Theta}_k - \Theta'_{k,\text{exp}} \|_2^2 + \lambda_{\theta_k} \sum_{j} |(\mathbf{D}\dot{\Theta}_k)_j| \quad \text{} $$
    其中 $\lambda_{\theta_k}$ 是为该角度数据调整的正则化参数 。
    \item \textbf{角加速度 $(\ddot{\theta}_1, \ddot{\theta}_2)$ 的计算}：
    令上一步得到的平滑角速度为 $\dot{\Theta}_{k,\text{smooth}}$，其简单差分导数估计为 $\dot{\Theta}'_{k,\text{smooth}}$。我们求解平滑的角加速度 $\ddot{\Theta}_k$，其最小化目标函数为：
    $$ E(\ddot{\Theta}_k) = \frac{1}{2} \| \ddot{\Theta}_k - \dot{\Theta}'_{k,\text{smooth}} \|_2^2 + \lambda_{\dot{\theta}_k} \sum_{j} |(\mathbf{D}\ddot{\Theta}_k)_j| \quad \text{} $$
    其中 $\lambda_{\dot{\theta}_k}$ 是为该角速度数据调整的正则化参数。
\end{enumerate}
如代码中所述，这些步骤通过调用 \texttt{totalVariationDiff} 函数完成。这些处理后的数据构成了后续SINDy-PI模型辨识的状态矩阵 $\mathbf{Data}$ 及其导数矩阵 $\mathbf{dData}$ 。代码中亦提及可选择使用滑动平均滤波 (\texttt{movmean}) 进行辅助平滑 。

\subsubsection*{3.4.3 热传导实验}
在热传导实验中，我们需要处理温度场数据 $T(t, x, y)$（瞬态）或 $T(x, y)$（稳态），并计算其对时间的一阶偏导数 $\frac{\partial T}{\partial t}$ 以及对空间坐标的各阶偏导数，如 $\frac{\partial T}{\partial x}, \frac{\partial T}{\partial y}, \frac{\partial^2 T}{\partial x^2}, \frac{\partial^2 T}{\partial y^2}, \frac{\partial^2 T}{\partial x \partial y}$ 等。
对于每一个待求的偏导数，我们可以将其视为在一系列固定其他变量的“切片”上进行的一维信号微分问题，并应用TVRegDiff。
\begin{itemize}
    \item \textbf{计算 $\frac{\partial T}{\partial t}(\mathbf{p}, t)$ （在固定空间点 $\mathbf{p}=(x_i,y_j)$ 处）}：
    令原始温度时间序列为 $\mathbf{T}_{\mathbf{p}}(t)$，其简单差分导数估计为 $\mathbf{T}'_{\mathbf{p}}(t)$。求解平滑的时间导数 $\mathbf{x}_{\partial t} = \frac{\partial T}{\partial t}(\mathbf{p}, t)$，其目标函数为：
    $$ E(\mathbf{x}_{\partial t}) = \frac{1}{2} \| \mathbf{x}_{\partial t} - \mathbf{T}'_{\mathbf{p}}(t) \|_2^2 + \lambda_t \sum_{j} |(\mathbf{D}_t \mathbf{x}_{\partial t})_j| $$
    其中 $\mathbf{D}_t$ 表示对时间序列的差分。
    \item \textbf{计算 $\frac{\partial T}{\partial x}(t_k, x, \mathbf{q})$ （在固定时间 $t_k$ 和其他空间坐标 $\mathbf{q}$ 处）}：
    令原始温度空间序列（沿x轴）为 $\mathbf{T}_{t_k,\mathbf{q}}(x)$，其简单差分导数估计为 $\mathbf{T}'_{t_k,\mathbf{q}}(x)$。求解平滑的空间导数 $\mathbf{x}_{\partial x} = \frac{\partial T}{\partial x}(t_k, x, \mathbf{q})$，其目标函数为：
    $$ E(\mathbf{x}_{\partial x}) = \frac{1}{2} \| \mathbf{x}_{\partial x} - \mathbf{T}'_{t_k,\mathbf{q}}(x) \|_2^2 + \lambda_x \sum_{j} |(\mathbf{D}_x \mathbf{x}_{\partial x})_j| $$
    其中 $\mathbf{D}_x$ 表示对x方向序列的差分。对于 $\frac{\partial T}{\partial y}$ 的计算同理。
    \item \textbf{计算二阶偏导数，如 $\frac{\partial^2 T}{\partial x^2}$}：
    首先计算得到平滑的一阶偏导数 $\mathbf{x}_{\partial x} = \frac{\partial T}{\partial x}$。令其简单差分导数估计为 $\mathbf{x}'_{\partial x}$。求解平滑的二阶导数 $\mathbf{x}_{\partial^2 x} = \frac{\partial^2 T}{\partial x^2}$，其目标函数为：
    $$ E(\mathbf{x}_{\partial^2 x}) = \frac{1}{2} \| \mathbf{x}_{\partial^2 x} - \mathbf{x}'_{\partial x} \|_2^2 + \lambda_{xx} \sum_{j} |(\mathbf{D}_x \mathbf{x}_{\partial^2 x})_j| $$
    对于 $\frac{\partial^2 T}{\partial y^2}$ 的计算同理。
    \item \textbf{计算混合偏导数，如 $\frac{\partial^2 T}{\partial x \partial y}$}：
    可以分步进行。例如，先计算平滑的 $\mathbf{x}_{\partial y} = \frac{\partial T}{\partial y}$。然后，对于每一个固定的 $x$ 值，将 $\mathbf{x}_{\partial y}$ 沿着 $x$ 方向视为信号，计算其关于 $x$ 的平滑导数。令 $\mathbf{s} = \mathbf{x}_{\partial y}$ （在固定 $x$ 方向上），$\mathbf{s}'$ 为其简单差分。求解 $\mathbf{x}_{\partial xy} = \frac{\partial^2 T}{\partial x \partial y}$，其目标函数为：
    $$ E(\mathbf{x}_{\partial xy}) = \frac{1}{2} \| \mathbf{x}_{\partial xy} - \mathbf{s}' \|_2^2 + \lambda_{xy} \sum_{j} |(\mathbf{D}_x \mathbf{x}_{\partial xy})_j| $$
\end{itemize}
在所有这些计算中，正则化参数 $\lambda$ 的选择至关重要，需要根据具体数据和噪声情况进行调整。求解这些凸优化问题，如前所述，可以借助如CVX等软件包实现，可以通过https://github.com/cvxr/cvx获取.zip格式的完整 CVX 捆绑包。或者如代码中那样自行实现ADMM算法。
