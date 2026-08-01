 $$ \left|\sin x=\frac{\sin x}{1}=\frac{2\sin\frac{x}{2}\cos\frac{x}{2}}{\sin^{2}\frac{x}{2}+\cos^{2}\frac{x}{2}}=\frac{2\tan\frac{x}{2}}{1+\tan^{2}\frac{x}{2}}\right. $$ 

万能公式：

 $$ \begin{aligned}&\left\{\begin{aligned}\cos x&=\frac{\cos x}{1}=\frac{\cos^{2}\frac{x}{2}-\sin^{2}\frac{x}{2}}{\cos^{2}\frac{x}{2}+\sin^{2}\frac{x}{2}}=\frac{1-\tan^{2}\frac{x}{2}}{1+\tan^{2}\frac{x}{2}}\\ \tan x&=\frac{2\tan\frac{x}{2}}{1-\tan^{2}\frac{x}{2}}\end{aligned}\right.\end{aligned} $$ 

4. 含三角的分式，如果异名，还可考虑万能公式：

比如：求 $ f(x)=\frac{1+\sin x}{2-3\cos x}(x\in\mathbb{R}) $的值域：

(1)令 $ \tan\frac{x}{2}=t\Rightarrow\begin{cases}\sin x=\dfrac{2t}{1+t^2}\\ \cos x=\dfrac{1-t^2}{1+t^2}\neq\dfrac{2}{3}\end{cases}\Rightarrow\dfrac{1-\sin x}{2-3\cos x}=\dfrac{1-\dfrac{2t}{1+t^2}}{2-3\cdot\dfrac{1-t^2}{1+t^2}}=\dfrac{1+t^2-2}{5^2-1}t, $  $ \vec{5}t\neq1 $

(2)令 $ \dfrac{1+t^2-2t}{5t^2-1}=y\Rightarrow\begin{cases}(5y-1)t^2+2t-y-1=0\\ \text{判别式法}\end{cases} $

 $ \Rightarrow\begin{cases}5y-1=0\text{时},&t=\dfrac{3}{5}\text{符合题意}\\5y-1\neq0\text{时},&\Delta=4+4(5y-1)(y+1)\geq0\end{cases}\Rightarrow y\geq0\text{或}y\leq\dfrac{4}{5} $

5. 类似  $ f(x) = 2\sin x + \sin 2x $ 这种结构，也可万能公式后求导处理；

例题1.（重庆·高考真题）函数  $ f(x)=\frac{\sin x}{\sqrt{5+4\cos x}} $（ $ 0\leq x\leq2\pi $）的值域是（ ）

A.  $ [-\frac{1}{4},\frac{1}{4}] $ B.  $ [-\frac{1}{3},\frac{1}{3}] $ C.  $ [-\frac{1}{2},\frac{1}{2}] $ D.  $ [-\frac{2}{3},\frac{2}{3}] $

例题2.（全国·高考真题）已知函数 $ f(x)=2\sin x+\sin 2x $，则 $ f(x) $的最小值是___。