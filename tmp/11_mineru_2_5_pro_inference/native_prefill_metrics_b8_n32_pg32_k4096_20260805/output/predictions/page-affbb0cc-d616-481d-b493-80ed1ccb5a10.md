第二章 行列式

18

4. 因此  \( \left|D_{3}\right| \leq 4 \leq (3 - 1)!(3 - 1) \)  成立.

2)假设 \(n\) 时成立， \(\left|D_{n + 1}\right|\leq \sum_{i = 1}^{n + 1}\left|D_n\right|\leq (n + 1)(n - 1)!(n - 1) <   n!\) 成立.综上，原命题成立.

\[
\mathbf {2 . 0 . 3 6} (\text {例} 2. 5 0). D = \left| \begin{array}{l l l l} 1 & \cos \varphi_ {1} & 2 \cos^ {2} \varphi_ {1} & 4 \cos^ {3} \varphi_ {1} \\ 1 & \cos \varphi_ {2} & 2 \cos^ {2} \varphi_ {2} & 4 \cos^ {3} \varphi_ {2} \\ 1 & \cos \varphi_ {3} & 2 \cos^ {2} \varphi_ {3} & 4 \cos^ {3} \varphi_ {3} \\ 1 & \cos \varphi_ {4} & 2 \cos^ {2} \varphi_ {4} & 4 \cos^ {3} \varphi_ {4} \end{array} \right| = 8 \prod_ {1 \leq j <   i \leq 4} (\cos \varphi_ {i} -
\]

\(\cos \varphi_{j})\) .因为 \(\cos n\varphi +i\sin n\varphi = e^{in\varphi} = (e^{i\varphi})^{n} = (\cos \varphi +i\sin \varphi)^{n},\cos n\varphi -\) \(i\sin n\varphi = e^{i(-n\varphi)} = (e^{i(-\varphi)})^{n} = (\cos \varphi -i\sin \varphi)^{n}.\) 所以 \(\cos n\varphi = \frac{1}{2} (e^{in\varphi}+\) \(e^{i(-n\varphi)}) = \frac{1}{2} ((\cos \varphi +i\sin \varphi)^{n} + (\cos \varphi -i\sin \varphi)^{n}) = 2^{n - 1}\cos^{n}\varphi +f(\cos \varphi),\) 其

中 \(f(x)\) 是次数小于 \(n\) 的多项式. 所以 \(D_{n} = \begin{vmatrix} 1 & \cos \varphi_{1} & \cos 2\varphi_{1} & \cdots & \cos (n - 1)\varphi_{1} \\ 1 & \cos \varphi_{2} & \cos 2\varphi_{2} & \cdots & \cos (n - 1)\varphi_{2} \\ 1 & \cos \varphi_{3} & \cos 2\varphi_{3} & \cdots & \cos (n - 1)\varphi_{3} \\ \vdots & \vdots & \vdots & & \vdots \\ 1 & \cos \varphi_{n} & \cos 2\varphi_{n} & \cdots & \cos (n - 1)\varphi_{n} \end{vmatrix} =\)

\[
\left| \begin{array}{cccc} 1 & \cos \varphi_ {1} & 2 \cos^ {2} \varphi_ {1} & \dots & 2 ^ {n - 2} \cos^ {n - 1} \varphi_ {1} \\ 1 & \cos \varphi_ {2} & 2 \cos^ {2} \varphi_ {2} & \dots & 2 ^ {n - 2} \cos^ {n - 1} \varphi_ {2} \\ 1 & \cos \varphi_ {3} & 2 \cos^ {2} \varphi_ {3} & \dots & 2 ^ {n - 2} \cos^ {n - 1} \varphi_ {3} \\ \vdots & \vdots & \vdots & & \vdots \\ 1 & \cos \varphi_ {n} & 2 \cos^ {2} \varphi_ {n} & \dots & 2 ^ {n - 2} \cos^ {n - 1} \varphi_ {n} \end{array} \right| = 2 ^ {\frac {(n - 1) (n - 2)}{2}} \prod_ {1 \leq j <   i \leq n} (\cos \varphi_ {i} - \cos \varphi_ {j}).
\]

2.0.37 (例2.63).  \( D_{n} = \begin{vmatrix} 1 & 0 & 0 & \cdots & 0 & 0 \\ 1 & 2 & 0 & \cdots & 0 & 0 \\ \vdots & \vdots & \vdots & & \vdots & \vdots \\ 1 & 2 & 2 & \cdots & 2 & 0 \\ 1 & 2 & 2 & \cdots & 2 & 2 \end{vmatrix} = 2^{n-1} \) . 设正项有 x 个, 负项

有y个.因为每一项的绝对值均为1,所以 \( \left\{\begin{aligned}x+y&=n!\\ x-y&=2^{n-1}\end{aligned}\right. \) ,所以正项有 \( \frac{1}{2}(n!+2^{n-1}) \) 项.

2.0.38 (例2.62). 对这 \(k\) 行用拉普拉斯展开, 因为 \(k > n - j\), 所以 \(k\) 阶子式必有一列全为0, 所以 \(k\) 阶子式为零, 所以 \(D = 0\).