40

ROMAN SHVYDKOY

Since \( E > 4\bar{n} \), we have \( \mathrm{x} + \mathrm{v} \geqslant E / 4 \), and hence, from the above,

\[
| I | E \lesssim \int_ {I} (\mathrm{x} + \mathrm{v}) \mathrm{d} t \lesssim \sum_ {i} | I | ^ {\delta_ {i} ^ {\prime}} E ^ {\delta_ {i} ^ {\prime \prime}},
\]

which finishes the proof.

This finishes the proof of Theorem 5.1.

6. APPENDIX: INTERPOLATION IN WEIGHTED SOBOLEV SPACES

For any any \( p \in \mathbb{N} \) we denote for short \( \partial_x^p f = (\partial_{x_1}^p f, \ldots, \partial_{x_n}^p f) \), and \( |\partial_x^p f|^2 = \sum_i |\partial_{x_i}^p f|^2 \). Similar notation will be used in \( v \)-variable. It is clear that \( |\partial_x^p f|^2 \) represents a linear combination of \( p \)th order derivatives.

Let us first note the following simple interpolation inequality: for \( k, l \geqslant 0 \) and \( K, L > 0 \) such that \( \frac{k}{K} + \frac{l}{L} < 1 \) and any \( |\mathbf{k}| = k \), \( |\mathbf{l}| = l \), we have

\[
(131)
\]

\[
\int_ {\Omega \times \mathbb {R} ^ {n}} | \partial_ {x} ^ {\mathbf {k}} \partial_ {v} ^ {\mathbf {l}} f | ^ {2} \mathrm{d} v \mathrm{d} x \leqslant \left(\int_ {\Omega \times \mathbb {R} ^ {n}} | \partial_ {x} ^ {K} f | ^ {2} \mathrm{d} v \mathrm{d} x\right) ^ {\frac {k}{K}} \left(\int_ {\Omega \times \mathbb {R} ^ {n}} | \partial_ {v} ^ {L} f | ^ {2} \mathrm{d} v \mathrm{d} x\right) ^ {\frac {l}{L}} \left(\int_ {\Omega \times \mathbb {R} ^ {n}} | f | ^ {2} \mathrm{d} v \mathrm{d} x\right) ^ {1 - \frac {k}{K} - \frac {l}{L}}.
\]

Indeed, denoting by \(\mathrm{d}\xi\) the counting measure over \(\mathbb{Z}^n\),

\[
\begin{array}{l} \int_ {\Omega \times \mathbb {R} ^ {n}} | \partial_ {x} ^ {\mathbf {k}} \partial_ {v} ^ {\mathbf {l}} f | ^ {2} \mathrm{d} v \mathrm{d} x = \int_ {\mathbb {Z} ^ {n} \times \mathbb {R} ^ {n}} \Pi_ {i = 1} ^ {n} | \xi_ {i} | ^ {2 k _ {i}} | \eta_ {i} | ^ {2 l _ {i}} | \hat {f} | ^ {2} \mathrm{d} \eta \mathrm{d} \xi \leqslant \int_ {\mathbb {Z} ^ {n} \times \mathbb {R} ^ {n}} | \xi | ^ {2 k} | \eta | ^ {2 l} | \hat {f} | ^ {2} \mathrm{d} \eta \mathrm{d} \xi \\ \leqslant \int_ {\mathbb {Z} ^ {n} \times \mathbb {R} ^ {n}} | \xi | ^ {2 k} | \hat {f} | ^ {2 k / K} | \eta | ^ {2 l} | \hat {f} | ^ {2 l / L} | \hat {f} | ^ {2 (1 - k / K - l / K)} \mathrm{d} \eta \mathrm{d} \xi \\ \leqslant \left(\int_ {\mathbb {Z} ^ {n} \times \mathbb {R} ^ {n}} | \xi | ^ {2 K} | \hat {f} | ^ {2} \mathrm{d} \eta \mathrm{d} \xi\right) ^ {\frac {k}{K}} \left(\int_ {\mathbb {Z} ^ {n} \times \mathbb {R} ^ {n}} | \eta | ^ {2 L} | \hat {f} | ^ {2} \mathrm{d} \eta \mathrm{d} \xi\right) ^ {\frac {l}{L}} \left(\int_ {\mathbb {Z} ^ {n} \times \mathbb {R} ^ {n}} | f | ^ {2} \mathrm{d} \eta \mathrm{d} \xi\right) ^ {1 - \frac {k}{K} - \frac {l}{L}} \\ \lesssim \left(\sum_ {i} \int_ {\mathbb {Z} ^ {n} \times \mathbb {R} ^ {n}} | \xi_ {i} | ^ {2 K} | \hat {f} | ^ {2} \mathrm{d} \eta \mathrm{d} \xi\right) ^ {\frac {k}{K}} \left(\sum_ {i} \int_ {\mathbb {Z} ^ {n} \times \mathbb {R} ^ {n}} | \eta_ {i} | ^ {2 L} | \hat {f} | ^ {2} \mathrm{d} \eta \mathrm{d} \xi\right) ^ {\frac {l}{L}} \\ \times \left(\int_ {\mathbb {Z} ^ {n} \times \mathbb {R} ^ {n}} | f | ^ {2} \mathrm{d} \eta \mathrm{d} \xi\right) ^ {1 - \frac {k}{K} - \frac {l}{L}} \\ = \left(\int_ {\Omega \times \mathbb {R} ^ {n}} | \partial_ {x} ^ {K} f | ^ {2} \mathrm{d} v \mathrm{d} x\right) ^ {\frac {k}{K}} \left(\int_ {\Omega \times \mathbb {R} ^ {n}} | \partial_ {v} ^ {L} f | ^ {2} \mathrm{d} v \mathrm{d} x\right) ^ {\frac {l}{L}} \left(\int_ {\Omega \times \mathbb {R} ^ {n}} | f | ^ {2} \mathrm{d} v \mathrm{d} x\right) ^ {1 - \frac {k}{K} - \frac {l}{L}}. \\ \end{array}
\]

When a weight is involved that depends only on v,  \( \omega = \omega(v) \) , then the above interpolation extends trivially in the case when k = 0 by application of (131) in x-variable only, and the Hölder inequality in v:

\[
\begin{array}{l} \int_ {\Omega \times \mathbb {R} ^ {n}} | \partial_ {x} ^ {\mathbf {k}} f | ^ {2} \omega \mathrm{d} v \mathrm{d} x \leqslant \int_ {\mathbb {R} ^ {n}} \left(\int_ {\Omega} | \partial_ {x} ^ {K} f | ^ {2} \mathrm{d} x\right) ^ {\frac {k}{K}} \left(\int_ {\Omega} | f | ^ {2} \mathrm{d} x\right) ^ {1 - \frac {k}{K}} \omega \mathrm{d} v \tag {132} \\ \leqslant \left(\int_ {\Omega \times \mathbb {R} ^ {n}} \left| \partial_ {x} ^ {K} f \right| ^ {2} \omega \mathrm{d} v \mathrm{d} x\right) ^ {\frac {k}{K}} \left(\int_ {\Omega \times \mathbb {R} ^ {n}} \left| f \right| ^ {2} \omega \mathrm{d} v \mathrm{d} x\right) ^ {1 - \frac {k}{K}}. \\ \end{array}
\]

It is clear that the above inequalities extend to any fractional k, K as well if we replace the derivatives with Fourier multipliers  \( D_{x}^{k}f \)  with symbol  \( |\xi|^{k} \) . In fact the computations above already use that symbol even for integer order derivatives. It is important to keep in mind though that if k is integer, then since  \( \omega \)  is not involved in Fourier transform, we have

\[
\int_ {\Omega \times \mathbb {R} ^ {n}} | D _ {x} ^ {k} f | ^ {2} \omega \mathrm{d} v \mathrm{d} x \sim \int_ {\Omega \times \mathbb {R} ^ {n}} | \partial_ {x} ^ {k} f | ^ {2} \omega \mathrm{d} v \mathrm{d} x.
\]

In other words we can go back to the classical derivatives.

Lemma 6.1. Suppose \(\omega\) is a doubling weight,

\[
c \leqslant \frac {\omega (v ^ {\prime})}{\omega (v ^ {\prime \prime})} \leqslant C, \quad \frac {1}{2} \leqslant \frac {| v ^ {\prime} |}{| v ^ {\prime \prime} |} \leqslant 2 \tag {133}
\]

\[
\omega (v) \sim 1, \quad | v | \leqslant 1.
\]