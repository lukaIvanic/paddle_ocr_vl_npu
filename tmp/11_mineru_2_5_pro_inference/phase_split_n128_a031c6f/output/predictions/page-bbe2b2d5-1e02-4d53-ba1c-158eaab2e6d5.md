62

RYOTA MIKAMI

\((\epsilon = *,!)\), where \(S_{L'} \in \Lambda\) is the polyhedron such that \(\mathrm{rel.int} S_{L'} \supset \mathrm{rel.int} S_{L',r(L')}^{\prime}\), the morphism \(j_{S_{L'}}\): \(\mathrm{rel.int} S_{L'} \to X\) is the inclusion,

\[
a ^ {\prime} (S _ {L ^ {\prime}, r (L ^ {\prime})} ^ {\prime} = 0, S _ {L ^ {\prime}} = a _ {S _ {L ^ {\prime}, r (L ^ {\prime})} ^ {\prime}} ^ {\prime}) \in \mathbb {Z} ^ {\Lambda \cup \Lambda^ {\prime}}
\]

is the element whose \( S_{L',r(L')} \)-component is 0, \( S_{L'} \)-component is \( a_{S_{L',r(L')}}' \), and the other components are same as \( a' \), and

\[
a ^ {\prime} (S _ {L ^ {\prime}, r (L ^ {\prime})} ^ {\prime} = a _ {S _ {L ^ {\prime}, r (L ^ {\prime})} ^ {\prime}} ^ {\prime} + \mathrm{codim} _ {S _ {L ^ {\prime}}} S _ {L ^ {\prime}, r (L ^ {\prime})} ^ {\prime}) \in \mathbb {Z} ^ {\Lambda \cup \Lambda^ {\prime}}
\]

is the element whose \( S_{L',r(L')} \)-component is \( a_{S_{L',r(L')}'}' + \mathrm{codim}_{S_{L'}} S_{L',r(L')}' \), and the other components are same as \( a' \).

Proof. In the same way as Lemma 5.8, there is a decomposition

\[
\mathrm{Fil}_{*}^{\Lambda_{\supseteq S_{L^{\prime}}}}\left.\mathscr{F}_{X_{\mathrm{sm}}}^{p,w}\right|_{X_{\mathrm{sm}}\cap \bigcap_{i = 1}^{r(L^{\prime})}W_{S^{\prime}_{L^{\prime},i}}^{\prime}}\cong \bigoplus_{\substack{b\in \mathbb{Z}^{\left\{S^{\prime}_{L^{\prime},i}\right\}_{i = 1}^{r(L^{\prime})}\cup \left\{S_{L^{\prime}}\right\}}}}\mathrm{Fil}_{*}^{\Lambda_{\supseteq S_{L^{\prime}}}}\left.\mathrm{gr}_{b}\left.\mathscr{F}_{X_{\mathrm{sm}}}^{p,w}\right|_{X_{\mathrm{sm}}\cap \bigcap_{i = 1}^{r(L^{\prime})}W_{S^{\prime}_{L^{\prime},i}}^{\prime}}\right.,
\]

where direct summands of the right-hand side are defined similarly to above Lemma 5.8. Then we have a decomposition of \( IC_{\mathrm{Trop,sheaf},X}^{d - p,*} \big|_{\bigcap_{i = 1}^{r(L')} W_{S_{L',i}'}'} \) similar to Corollary 5.24. Then cohomology groups in the assertion are direct sums of cohomology groups of direct summands, and the assertion immediately follows from direct computation. (Note that we can take \( W_{S_{L',r(L')}'}' \subset W_{S_{L'}} \), since \( a' \) is adapted to \( L' \), we have \( a_{S_{L'}}' \geq 0 \), and we have \( \sigma_{S_{L'}}' = \sigma_{S_{L',r(L')}'}' \).)

Remark 6.14. By construction,  \( \operatorname{Fil}_{*}^{\Lambda\cup\Lambda'} IC_{\operatorname{Trop,sheaf}}^{d-p,*} \)  satisfies analogs of Axiom  \( A_{p} \)  (1) and (2) at each  \( S_{k} \in \Lambda_{sing} \)  for an adapted pair  \( L' = (S_{L',1}' \subsetneq \cdots \subsetneq S_{L',r(L')-1}' \subsetneq S_{k}) \)  ( \( S_{L',i}' \in \Lambda' \) ) and  \( a^{\circ} \in Z^{\Lambda\cup\Lambda'} \) . By decompositions in proof of Lemma 6.13 and Lemma 5.21, it also satisfies an analog of Axiom  \( A_{p}(2)' \) .

Proposition 6.15. We have a natural LG-quasi-isomorphism

\[
\mathrm{For} _ {\Lambda \setminus \Lambda^ {\prime}} \mathrm{Fil} _ {*} ^ {\Lambda \cup \Lambda^ {\prime}} I C _ {\mathrm{Trop,sheaf}} ^ {d - p, *} \cong \mathrm{Fil} _ {*} ^ {\Lambda^ {\prime}} I C _ {\mathrm{Trop,sheaf}, X ^ {\prime}} ^ {d - p, *}
\]

in \(D_{c,[-\mathrm{gr}(X'),0],LG}^{b}(X,\mathrm{grMod}\mathbb{Q}[T_{S'}]_{S'\in \Lambda '})\).

Proof. We have a natural LG-isomorphism

\[
\mathrm{For} _ {\Lambda \setminus \Lambda^ {\prime}} \mathrm{Fil} _ {*} ^ {\Lambda \cup \Lambda^ {\prime}} \mathcal {F} _ {X _ {\mathrm{sm}}} ^ {d - p, w} | _ {X _ {\mathrm{sm}} ^ {\prime}} \cong \mathrm{Fil} _ {*} ^ {\Lambda^ {\prime}} \mathcal {F} _ {X _ {\mathrm{sm}} ^ {\prime}} ^ {d - p, w ^ {\prime}}
\]

on \(X_{\mathrm{sm}}^{\prime}\). By Proposition 5.17, it suffices to show that \(\mathrm{For}_{\Lambda \setminus \Lambda^{\prime}}\mathrm{Fil}_{*}^{\Lambda \cup \Lambda^{\prime}}IC_{\mathrm{Trop,sheaf}}^{d - p,*}\) satisfies Axiom \(A_{p}\) (1) and \((2)^{\prime}\) at each \(S^{\prime}\in \Lambda_{\mathrm{sing}}^{\prime}\). This follows from Remark 5.20, Lemma 6.13, and Remark 6.14 directly. (Note that for \(S\in \Lambda\) and \(S^{\prime}\in \Lambda^{\prime}\) with rel.int \(S^{\prime}\subset\) rel.int \(S\), we have \(m_{p,S} + \mathrm{codim}_S S^{\prime} = m_{p,S^{\prime}}\).)