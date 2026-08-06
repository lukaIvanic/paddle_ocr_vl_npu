For \(\beta_1'\), with (S.17) we have with probability at least \(1 - O(n^{-10})\)

\[
\begin{array}{l} \beta_ {1} ^ {\prime} = \left\| \mathbf {R} ^ {* ^ {\top}} \left(\mathbf {U} \mathbf {U} ^ {\top} \mathbf {U} ^ {*} - \mathbf {U} ^ {*}\right) \right\| _ {2, \infty} \\ \leq \sigma_ {1} ^ {*} \| \mathbf {V} ^ {*} \| _ {2, \infty} \left\| \mathbf {U} ^ {* \top} \mathbf {U} \mathbf {U} ^ {\top} \mathbf {U} ^ {*} - \mathbf {I} _ {K} \right\| \lesssim \frac {\kappa^ {*} \sigma^ {2} (M N + J)}{\sigma_ {K} ^ {*}} \| \mathbf {V} ^ {*} \| _ {2, \infty}. \tag {S.33} \\ \end{array}
\]

For \(\beta_2'\), introducing the "leave-one-block-out" alternative \(\mathbf{U}^{(j)}\) yields that

\[
\begin{array}{l} \beta_ {2} ^ {\prime} = \max _ {j \in [ J ]} \left\| \mathbf {E} _ {:, j} ^ {\top} \left(\mathbf {U} \mathbf {U} ^ {\top} \mathbf {U} ^ {*} - \mathbf {U} ^ {*}\right) \right\| _ {2} \\ \leq \underbrace {\max _ {j \in [ J ]} \left\| \mathbf {E} _ {: , j} ^ {\top} \left(\mathbf {U} ^ {(j)} \mathbf {U} ^ {(j) ^ {\top}} \mathbf {U} ^ {*} - \mathbf {U} ^ {*}\right) \right\|} _ {\gamma_ {1} ^ {\prime}} + \underbrace {\max _ {j \in [ J ]} \left\| \mathbf {E} _ {: , j} ^ {\top} \left(\mathbf {U} \mathbf {U} ^ {\top} \mathbf {U} ^ {*} - \mathbf {U} ^ {(j)} \mathbf {U} ^ {(j) ^ {\top}} \mathbf {U} ^ {*}\right) \right\|} _ {\gamma_ {2} ^ {\prime}}. \tag {S.34} \\ \end{array}
\]

Upper bounding \(\gamma_1'\) in (S.34). Utilizing Lemma S.4, one has at least \(1 - O(d^{-11})\) that,

\[
\begin{array}{l} \gamma_ {1} ^ {\prime} \leq \sigma \sqrt {\log d} \max _ {j \in [ J ]} \left\| \mathbf {U} ^ {(j)} \mathbf {U} ^ {(\boldsymbol {j}) ^ {\top}} \mathbf {U} ^ {*} - \mathbf {U} ^ {*} \right\| _ {F} + B \log d \max _ {j \in [ J ]} \left\| \mathbf {U} ^ {(j)} \mathbf {U} ^ {(\boldsymbol {j}) ^ {\top}} \mathbf {U} ^ {*} - \mathbf {U} ^ {*} \right\| _ {2, \infty} \\ \leq \sigma \sqrt {\log d} \| \mathbf {U} \mathbf {U} ^ {\top} \mathbf {U} ^ {*} - \mathbf {U} ^ {*} \| _ {F} + \sigma \sqrt {\log d} \max _ {j \in [ J ]} \| \mathbf {U} ^ {(j)} \mathbf {U} ^ {(j) \top} \mathbf {U} ^ {*} - \mathbf {U} \mathbf {U} ^ {\top} \mathbf {U} ^ {*} \| _ {F} \\ + B \log d \| \mathbf {U} \mathbf {U} ^ {\top} \mathbf {U} ^ {*} - \mathbf {U} ^ {*} \| _ {2, \infty} + B \log d \max _ {j \in [ J ]} \| \mathbf {U} ^ {(j)} \mathbf {U} ^ {(j) \top} \mathbf {U} ^ {*} - \mathbf {U} \mathbf {U} ^ {\top} \mathbf {U} ^ {*} \| _ {2, \infty} \\ \leq \sigma \sqrt {\log d} \left\| \mathbf {U} \mathbf {U} ^ {\top} \mathbf {U} ^ {*} - \mathbf {U} ^ {*} \right\| _ {F} + B \log d \left\| \mathbf {U} \mathbf {U} ^ {\top} \mathbf {U} ^ {*} - \mathbf {U} ^ {*} \right\| _ {2, \infty} \\ + \left(\sigma \sqrt {\log d} + B \log d\right) \max _ {j \in [ J ]} \left\| \mathbf {U} ^ {(j)} \mathbf {U} ^ {(j) ^ {\top}} - \mathbf {U} \mathbf {U} ^ {\top} \right\| _ {F} \\ \lesssim \sigma \sqrt {\log d} \left\| \mathbf {U} \mathbf {U} ^ {\top} \mathbf {U} ^ {*} - \mathbf {U} ^ {*} \right\| _ {F} + B \log d \left\| \mathbf {U} \mathbf {U} ^ {\top} \mathbf {U} ^ {*} - \mathbf {U} ^ {*} \right\| _ {2, \infty} \\ + B \log d \left\| \mathbf {U} ^ {(j)} \mathbf {U} ^ {(j) ^ {\top}} - \mathbf {U} \mathbf {U} ^ {\top} \right\| _ {F}, \tag {S.35} \\ \end{array}
\]

where the last line holds since \(\sigma \sqrt{\log d} \ll B \log d\).

To bound \(\max_{j\in [J]}\left\| \mathbf{U}^{(j)}\mathbf{U}^{(j)^\top} - \mathbf{U}\mathbf{U}^\top \right\| _F\) in the RHS above, we apply Wedin's theorem with (S.31) and obtain

\[
\begin{array}{l} \max \left\{\left\| \mathbf {U} \mathbf {U} ^ {\top} - \mathbf {U} ^ {(j)} \mathbf {U} ^ {(j) ^ {\top}} \right\| _ {F}, \left\| \mathbf {V} \mathbf {V} ^ {\top} - \mathbf {V} ^ {(j)} \mathbf {V} ^ {(j) ^ {\top}} \right\| _ {F} \right\} \\ \lesssim \frac {\max \left\{\left\| \mathcal {P} _ {: , S _ {l _ {j}}} (\mathbf {E}) ^ {\top} \mathbf {U} ^ {(j)} \right\| _ {F} , \left\| \mathcal {P} _ {: , S _ {l _ {j}}} (\mathbf {E}) \mathbf {V} ^ {(j)} \right\| _ {F} \right\}}{\sigma_ {K} ^ {*}}. \tag {S.36} \\ \end{array}
\]

51