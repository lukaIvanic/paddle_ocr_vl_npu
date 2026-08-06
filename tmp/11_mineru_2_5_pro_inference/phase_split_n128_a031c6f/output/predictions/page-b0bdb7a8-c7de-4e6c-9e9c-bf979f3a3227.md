UNIVERSAL \(\overline{\partial}\) SOLUTION ON STRONGLY PSEUDOCONVEX

9

Proof. By compactness of \( b\Omega \) we can take a finitely many \( \zeta_1, \ldots, \zeta_M \) such that \( b\Omega \subset \bigcup_{\nu=1}^{M} \Phi_{\zeta_\nu}(U_{\zeta_\nu}) \). Let \( \chi_0 \in C_c^\infty(\Omega) \) and \( \chi_\nu \in C_c^\infty(\Phi_{\zeta_\nu}(U_{\zeta_\nu})) \) (\( 1 \leq \nu \leq M \)) be such that \( \sum_{\nu=0}^\infty \chi_\nu(z) \equiv 1 \) for all \( z \in \Omega \). We take \( \lambda_0 \in C_c^\infty(\Omega) \) and \( \lambda_\nu \in C_c^\infty(\Phi_{\zeta_\nu}(U_{\zeta_\nu})) \) for \( 1 \leq \nu \leq M \), such that for each \( 0 \leq \nu \leq M \), \( \lambda_\nu \equiv 1 \) in an open neighborhood of \( \mathrm{supp}\, \chi_\nu \). Therefore \( \chi_\nu = \lambda_\nu \chi_\nu \) and \( \mathrm{supp}\, \chi_\nu \cap \mathrm{supp}(\overline{\partial}\lambda_\nu) = \varnothing \) for \( 0 \leq \nu \leq M \).

We need to apply Proposition 2.3. Set

\[
\mathscr {X} _ {q} ^ {k} := H ^ {k t _ {0}, 2} (\Omega ; \wedge^ {0, q}), \qquad 0 \leq q \leq n, \qquad k \in \mathbb {Z}.
\]

Therefore by Lemma A.13, we have \(\mathcal{X}_q^{-\infty} = \mathcal{S}'(\Omega; \wedge^{0,q})\) and \(\mathcal{X}_q^{+\infty} = \mathcal{C}^\infty(\Omega; \wedge^{0,q})\). Set \(D_q := \overline{\partial}\) on \((0,q)\)-forms. Immediately (B) holds. By (2.4) we get (E).

We define operators \(P'\), \(H_q'\) and \(R_q'\) on \(\Omega\) by the following:

\[
P ^ {\prime} f := \sum_ {\nu = 1} ^ {M} \Phi_ {\zeta_ {\nu} *} \big ((\lambda_ {\nu} \circ \Phi_ {\zeta_ {\nu}}) \cdot \mathcal {P} ^ {\zeta_ {\nu}} [ \Phi_ {\zeta_ {\nu}} ^ {*} (\chi_ {\nu} f) ] \big) = \sum_ {\nu = 1} ^ {M} \lambda_ {\nu} \cdot \Phi_ {\zeta_ {\nu} *} \circ \mathcal {P} ^ {\zeta_ {\nu}} \circ \Phi_ {\zeta_ {\nu}} ^ {*} [ \chi_ {\nu} f ]; \tag {2.11}
\]

\[
H _ {q} ^ {\prime} f := \lambda_ {0} \cdot \mathcal {B} _ {q} [ \chi_ {0} f ] + \sum_ {\nu = 1} ^ {M} \lambda_ {\nu} \cdot \Phi_ {\zeta_ {\nu} *} \circ \mathcal {H} _ {q} ^ {\zeta_ {\nu}} \circ \Phi_ {\zeta_ {\nu}} ^ {*} [ \chi_ {\nu} f ]. \tag {2.12}
\]

Here \(\mathcal{B}_q\) are the Bochner-Martinelli integral operators from (A.23). Recall (see Lemmas A.29 and A.30) the boundedness \(\mathcal{B}_q:H_c^{s,2}(\Omega ;\wedge^{0,q})\to H^{s + 1,2}(\Omega ;\wedge^{0,q - 1})\) for all \(s\in \mathbb{R}\) and the formula \(g = \overline{\partial}\mathcal{B}_qg + \mathcal{B}_{q + 1}\overline{\partial} g\) for \((0,q)\) forms \(g\) that has compact support.

We define  \( (R_{q})_{q=0}^{n} \)  by  \( R_{0}f := f - P'f - H_{1}'\overline{\partial}f \)  and  \( R_{q}f := f - \overline{\partial}H_{q}'f - H_{q+1}'\overline{\partial}f \)  for  \( 1 \leq q \leq n \) . Therefore (D) holds since we define  \( R_{q} \)  by this way.

To verify (C), we adapt the standard convention for forms of mixed degree from Convention 1.3: for form \( f = \sum_{q=0}^{n} f_q \) where \( f_q \) has degree \( (0, q) \), we use \( P'f = P'f_0 \) and \( H'f = \sum_{q=1}^{n} H'_q f_q \) and \( Rf = \sum_{q=0}^{n} R_q f_q \). Therefore, \( Rf = f - P'f - \overline{\partial} H'f - H' \overline{\partial} f \), which means \( \overline{\partial} R - R \overline{\partial} = \overline{\partial} P' \). Therefore (recall Remark 2.4 and we set \( R_{n+1} = 0 \)),

\[
R _ {q + 1} \overline {{\partial}} = \overline {{\partial}} R _ {q} \quad \text {for} 1 \leq q \leq n, \qquad \text {and} \qquad R _ {1} \overline {{\partial}} _ {0} - \overline {{\partial}} R _ {0} = \overline {{\partial}} P ^ {\prime}.
\]

Note that by (2.11) and (2.3) we have

\[
\overline {{\partial}} P ^ {\prime} f = \sum_ {\nu = 1} ^ {M} \overline {{\partial}} \lambda_ {\nu} \wedge \Phi_ {\zeta_ {\nu} *} \mathcal {P} ^ {\zeta_ {\nu}} \left[ \Phi_ {\zeta_ {\nu}} ^ {*} (\chi_ {\nu} f) \right] = \Phi_ {\zeta_ {\nu} *} \circ \left(\left(\Phi_ {\zeta_ {\nu}} ^ {*} \overline {{\partial}} \lambda_ {\nu}\right) \wedge \mathcal {P} ^ {\zeta_ {\nu}} \circ \left[ \left(\Phi_ {\zeta_ {\nu}} ^ {*} \chi_ {\nu}\right) \cdot \left(\Phi_ {\zeta_ {\nu}} ^ {*} f\right) \right]\right).
\]

Since \(\overline{\partial}\lambda_{\nu}\) and \(\chi_{\nu}\) have disjoint supports, so do \(\Phi_{\zeta_{\nu}}^{*}\overline{\partial}\lambda_{\nu}\) and \(\Phi_{\zeta_{\nu}}^{*}\chi_{\nu}\). By the assumption (c) we get \(\overline{\partial} P^{\prime}:\mathcal{S}^{\prime}\to \mathcal{C}^{\infty}\). This completes the verification of (C).

To see \( R_{q}: H^{s,2} \to H^{s + t_{0},2} \) is bounded, i.e. to verify (A) we compute the explicit expression.

\[
\begin{array}{l} R f = f - P ^ {\prime} f - \overline {{\partial}} H ^ {\prime} f - H ^ {\prime} \overline {{\partial}} f \\ = \lambda_ {0} \chi_ {0} f - \overline {{\partial}} (\lambda_ {0} \mathcal {B} [ \chi_ {0} f ]) - \lambda_ {0} \mathcal {B} [ \chi_ {0} \overline {{\partial}} f ] \\ + \sum_ {\nu = 1} ^ {M} \left\{\lambda_ {\nu} \chi_ {\nu} f - \lambda_ {\nu} \Phi_ {\zeta_ {\nu} *} \mathcal {P} ^ {\zeta_ {\nu}} \Phi_ {\zeta_ {\nu}} ^ {*} [ \chi_ {\nu} f ] - \overline {{\partial}} (\lambda_ {\nu} \Phi_ {\zeta_ {\nu} *} \mathcal {H} ^ {\zeta_ {\nu}} \Phi_ {\zeta_ {\nu}} ^ {*} [ \chi_ {\nu} f ]) - \lambda_ {\nu} \Phi_ {\zeta_ {\nu} *} \mathcal {H} ^ {\zeta_ {\nu}} \Phi_ {\zeta_ {\nu}} ^ {*} [ \chi_ {\nu} \overline {{\partial}} f ] \right\} \\ = \lambda_ {0} \cdot (\mathrm{id} - \overline {{\partial}} \mathcal {B} - \mathcal {B} \overline {{\partial}}) [ \chi_ {0} f ] - \overline {{\partial}} \lambda_ {0} \wedge \mathcal {B} [ \chi_ {0} f ] + \lambda_ {0} \cdot \mathcal {B} [ \overline {{\partial}} \chi_ {0} \wedge f ] + \sum_ {\nu = 1} ^ {M} \lambda_ {\nu} \cdot \Phi_ {\zeta_ {\nu} *} \mathcal {H} ^ {\zeta_ {\nu}} \Phi_ {\zeta_ {\nu}} ^ {*} [ \overline {{\partial}} \chi_ {\nu} \wedge f ] \\ + \sum_ {\nu = 1} ^ {M} \left\{\lambda_ {\nu} \cdot \Phi_ {\zeta_ {\nu} *} \left\{\mathrm{id} - \mathcal {P} ^ {\zeta_ {\nu}} - \overline {{\partial}} \mathcal {H} ^ {\zeta_ {\nu}} - \mathcal {H} ^ {\zeta_ {\nu}} \overline {{\partial}} \right\} \Phi_ {\zeta_ {\nu}} ^ {*} [ \chi_ {\nu} f ] - \overline {{\partial}} \lambda_ {\nu} \wedge \Phi_ {\zeta_ {\nu} *} \mathcal {H} ^ {\zeta_ {\nu}} \Phi_ {\zeta_ {\nu}} ^ {*} [ \chi_ {\nu} f ] \right\} \\ = \lambda_ {0} \cdot \mathcal {B} [ \overline {{\partial}} \chi_ {0} \wedge f ] - \overline {{\partial}} \lambda_ {0} \wedge \mathcal {B} [ \chi_ {0} f ] + \sum_ {\nu = 1} ^ {M} \lambda_ {\nu} \cdot \Phi_ {\zeta_ {\nu} *} \circ \mathcal {R} ^ {\zeta_ {\nu}} \circ \Phi_ {\zeta_ {\nu}} ^ {*} [ \chi_ {\nu} f ] \\ + \sum_ {\nu = 1} ^ {M} \Big \{- \overline {{\partial}} \lambda_ {\nu} \wedge \Phi_ {\zeta_ {\nu} *} \circ \mathcal {H} ^ {\zeta_ {\nu}} \circ \Phi_ {\zeta_ {\nu}} ^ {*} [ \chi_ {\nu} f ] + \lambda_ {\nu} \cdot \Phi_ {\zeta_ {\nu} *} \circ \mathcal {H} ^ {\zeta_ {\nu}} \circ \Phi_ {\zeta_ {\nu}} ^ {*} [ \overline {{\partial}} \chi_ {\nu} \wedge f ] \Big \}. \\ \end{array}
\]