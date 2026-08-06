第二章 行列式

27

\[
\begin{array}{l} = \prod_ {i = 1} ^ {n} \sin \alpha_ {i} \cos \alpha_ {i} \left| \begin{array}{c c c c c} 1 & - \tan \alpha_ {1} & - \tan \alpha_ {2} & \dots & - \tan \alpha_ {n} \\ \frac {1}{\tan \alpha_ {1}} & 1 & 1 & \dots & 1 \\ \frac {1}{\tan \alpha_ {2}} & 1 & 1 & \dots & 1 \\ \vdots & \vdots & \vdots & & \vdots \\ \frac {1}{\tan \alpha_ {n}} & 1 & 1 & \dots & 1 \end{array} \right| \\ = \left\{ \begin{array}{c c} 0 & , n \geq 3 \\ - \sin^ {2} \left(\alpha_ {1} - \alpha_ {2}\right) & , n = 2 \end{array} . \right. \\ \end{array}
\]

2.0.56 (例2.15).

\[
D _ {n + 1} = \left| \begin{array}{c c c c c c} a + x & - 1 & 0 & \dots & 0 & 0 \\ 0 & a + x & - 1 & \dots & 0 & 0 \\ 0 & 0 & a + x & \dots & 0 & 0 \\ \vdots & \vdots & \vdots & & \vdots & \vdots \\ 0 & 0 & 0 & \dots & a + x & - 1 \\ 0 & 0 & 0 & \dots & 0 & a \end{array} \right| = a (a + x) ^ {n}.
\]

2.0.57 (例2.19).

\[
\begin{array}{l} D = \left| \begin{array}{c c c c c} 1 & c _ {1} & c _ {2} & \dots & c _ {n} \\ 0 & a _ {1} + b _ {1} c _ {1} & a _ {2} + b _ {1} c _ {2} & \dots & a _ {n} + b _ {1} c _ {n} \\ 0 & a _ {1} + b _ {2} c _ {1} & a _ {2} + b _ {2} c _ {2} & \dots & a _ {n} + b _ {2} c _ {n} \\ \vdots & \vdots & \vdots & & \vdots \\ 0 & a _ {1} + b _ {n} c _ {1} & a _ {2} + b _ {n} c _ {2} & \dots & a _ {n} + b _ {n} c _ {n} \end{array} \right| \\ = \left| \begin{array}{c c c c c} 1 & c _ {1} & c _ {2} & \dots & c _ {n} \\ - b _ {1} & a _ {1} & a _ {2} & \dots & a _ {n} \\ - b _ {2} & a _ {1} & a _ {2} & \dots & a _ {n} \\ \vdots & \vdots & \vdots & & \vdots \\ - b _ {n} & a _ {1} & a _ {2} & \dots & a _ {n} \end{array} \right| = 0 \\ \end{array}
\]