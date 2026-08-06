FOKKER-PLANCK-ALIGNMENT EQUATIONS

19

We now test (80) against \( G' \left( \frac{(\theta f)_{\varepsilon_1}}{\theta + \varepsilon_2} \right) \), and integrate in time:

\[
\begin{array}{l} \int_ {\Omega \times \mathbb {R} ^ {n} \times \{t \}} G \left(\frac {(\theta f) _ {\varepsilon_ {1}}}{\theta + \varepsilon_ {2}}\right) \varphi \mathrm{d} v \mathrm{d} x - \int_ {\Omega \times \mathbb {R} ^ {n} \times \{0 \}} G \left(\frac {(\theta f) _ {\varepsilon_ {1}}}{\theta + \varepsilon_ {2}}\right) \varphi \mathrm{d} v \mathrm{d} x - \int_ {0} ^ {t} \int_ {\Omega \times \mathbb {R} ^ {n} \times \{t \}} G \left(\frac {(\theta f) _ {\varepsilon_ {1}}}{\theta + \varepsilon_ {2}}\right) \partial_ {t} \varphi \mathrm{d} v \mathrm{d} x \mathrm{d} s \\ = - \int_ {0} ^ {t} \int_ {\Omega \times \mathbb {R} ^ {n}} \frac {(\theta v \cdot \nabla_ {x} f) _ {\varepsilon_ {1}}}{\theta + \varepsilon_ {2}} G ^ {\prime} \left(\frac {(\theta f) _ {\varepsilon_ {1}}}{\theta + \varepsilon_ {2}}\right) \varphi \mathrm{d} v \mathrm{d} x \mathrm{d} s + \int_ {0} ^ {t} \int_ {\Omega \times \mathbb {R} ^ {n}} \frac {(\theta \mathrm{s} _ {\rho} \Delta_ {v} f) _ {\varepsilon_ {1}}}{\theta + \varepsilon_ {2}} G ^ {\prime} \left(\frac {(\theta f) _ {\varepsilon_ {1}}}{\theta + \varepsilon_ {2}}\right) \varphi \mathrm{d} v \mathrm{d} x \mathrm{d} s \\ + \int_ {0} ^ {t} \int_ {\Omega \times \mathbb {R} ^ {n}} \frac {\nabla_ {v} \cdot \left(\theta \mathrm{s} _ {\rho} v f\right) _ {\varepsilon_ {1}}}{\theta + \varepsilon_ {2}} G ^ {\prime} \left(\frac {(\theta f) _ {\varepsilon_ {1}}}{\theta + \varepsilon_ {2}}\right) \varphi \mathrm{d} v \mathrm{d} x \mathrm{d} s - \int_ {0} ^ {t} \int_ {\Omega \times \mathbb {R} ^ {n}} \frac {\nabla_ {v} \cdot \left(\theta \mathrm{w} _ {\rho} f\right) _ {\varepsilon_ {1}}}{\theta + \varepsilon_ {2}} G ^ {\prime} \left(\frac {(\theta f) _ {\varepsilon_ {1}}}{\theta + \varepsilon_ {2}}\right) \varphi \mathrm{d} v \mathrm{d} x \mathrm{d} s \\ + \int_ {0} ^ {t} \int_ {\Omega \times \mathbb {R} ^ {n}} \left[ \frac {(f \partial_ {t} \theta) _ {\varepsilon_ {1}}}{\theta + \varepsilon_ {2}} - \frac {(\theta f) _ {\varepsilon_ {1}} \partial_ {t} \theta}{[ \theta + \varepsilon_ {2} ] ^ {2}} \right] G ^ {\prime} \left(\frac {(\theta f) _ {\varepsilon_ {1}}}{\theta + \varepsilon_ {2}}\right) \varphi \mathrm{d} v \mathrm{d} x \mathrm{d} s := - I + I I + I I I - I V + V. \\ \end{array}
\]

Let us note that \(\theta\) satisfies the mollified continuity equation

\[
\partial_ {t} \theta = - (u \rho) * \nabla \chi_ {r _ {0}},
\]

and so,

\[
\| \partial_ {t} \theta \| _ {\infty} \lesssim \| u \| _ {L _ {\rho} ^ {2}} \lesssim C. \tag {81}
\]

In what follows we will be taking consecutive limits as \(\varepsilon_{1}\to 0\) and then \(\varepsilon_2\to 0\).

The left hand side converges to its natural limit since \( G \) is smooth and \( \frac{(\theta f)_{\varepsilon_1}}{\theta + \varepsilon_2} \to \frac{\theta f}{\theta + \varepsilon_2} \to f \) pointwise almost everywhere. So, we obtain

\[
\int_ {\Omega \times \mathbb {R} ^ {n} \times \{t \}} G (f) \varphi \mathrm{d} v \mathrm{d} x - \int_ {\Omega \times \mathbb {R} ^ {n} \times \{0 \}} G (f) \varphi \mathrm{d} v \mathrm{d} x - \int_ {0} ^ {t} \int_ {\Omega \times \mathbb {R} ^ {n} \times \{t \}} G (f) \partial_ {t} \varphi \mathrm{d} v \mathrm{d} x \mathrm{d} s.
\]

LIMIT OF \(I\). Let us go back to \(I\). Integrating by parts we have

\[
\begin{array}{l} I = - \int_ {0} ^ {t} \int_ {\Omega \times \mathbb {R} ^ {n}} \frac {(f v \cdot \nabla \theta) _ {\varepsilon_ {1}}}{\theta + \varepsilon_ {2}} G ^ {\prime} \left(\frac {(\theta f) _ {\varepsilon_ {1}}}{\theta + \varepsilon_ {2}}\right) \varphi \mathrm{d} v \mathrm{d} x \mathrm{d} s \\ + \int_ {0} ^ {t} \int_ {\Omega \times \mathbb {R} ^ {n}} \frac {(f v \theta) * \nabla_ {x} \chi_ {\varepsilon_ {1}}}{\theta + \varepsilon_ {2}} G ^ {\prime} \left(\frac {(\theta f) _ {\varepsilon_ {1}}}{\theta + \varepsilon_ {2}}\right) \varphi \mathrm{d} v \mathrm{d} x \mathrm{d} s := I _ {1} + I _ {2}. \\ \end{array}
\]

For the limit of \(I_1\), we have \(fv \cdot \nabla \theta \in L^1\), so, \((fv \cdot \nabla \theta)_{\varepsilon_1} \to fv \cdot \nabla \theta\) in \(L^1\), while \(G' \left( \frac{(\theta f)_{\varepsilon_1}}{\theta + \varepsilon_2} \right) \to G' \left( \frac{\theta f}{\theta + \varepsilon_2} \right)\) a.e. and is uniformly bounded. Consequently, by dominated convergence,

\[
I _ {1} \rightarrow - \int_ {0} ^ {t} \int_ {\Omega \times \mathbb {R} ^ {n}} \frac {f v \cdot \nabla \theta}{\theta + \varepsilon_ {2}} G ^ {\prime} \left(\frac {\theta f}{\theta + \varepsilon_ {2}}\right) \varphi \mathrm{d} v \mathrm{d} x \mathrm{d} s. \tag {82}
\]

We will leave it at that for now, and turn to \(I_{2}\). Observe

\[
\begin{array}{l} (f v \theta) * \nabla_ {x} \chi_ {\varepsilon_ {1}} = \int_ {\Omega \times \mathbb {R} ^ {n}} f (y, w) w \theta (y) \nabla_ {x} \chi_ {\varepsilon_ {1}} (x - y, v - w) \mathrm{d} w \mathrm{d} y \\ = \int_ {\Omega \times \mathbb {R} ^ {n}} f (y, w) (w - v) \theta (y) \nabla_ {x} \chi_ {\varepsilon_ {1}} (x - y, v - w) \mathrm{d} w \mathrm{d} y + v \cdot \nabla_ {x} (f \theta) _ {\varepsilon_ {1}} \\ \end{array}
\]

using that \(w\nabla_{x}\chi_{\varepsilon_{1}}(\cdot ,w)\) is odd in \(w\), we can insert \(f(y,v)\),

\[
\begin{array}{l} (f v \theta) * \nabla_ {x} \chi_ {\varepsilon_ {1}} = \int_ {\Omega \times \mathbb {R} ^ {n}} [ f (y, w) - f (y, v) ] (w - v) \theta (y) \nabla_ {x} \chi_ {\varepsilon_ {1}} (x - y, v - w) \mathrm{d} w \mathrm{d} y + v \cdot \nabla_ {x} (f \theta) _ {\varepsilon_ {1}} \tag {83} \\ = \int_ {0} ^ {1} \int_ {\Omega \times \mathbb {R} ^ {n}} \theta (x - y) \nabla_ {v} f (x - y, v + \theta w) w \otimes w \nabla_ {x} \chi_ {\varepsilon_ {1}} (y, w) \mathrm{d} w \mathrm{d} y \mathrm{d} \theta + v \cdot \nabla_ {x} (\theta f) _ {\varepsilon_ {1}}. \\ \end{array}
\]