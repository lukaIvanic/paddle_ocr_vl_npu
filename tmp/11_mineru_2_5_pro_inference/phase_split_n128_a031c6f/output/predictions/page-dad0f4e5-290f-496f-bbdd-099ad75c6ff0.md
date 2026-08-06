HOMEOMORPHISMS OF THE PSEUDOARC

23

hence \( p \sqcap_{G_m} q \), as \( \sqsupseteq_n^m \) is \( \sqcap \)-preserving. Conversely, if \( p \sqcap_{G_m} q \) then we have \( r \in G_{m+1} \) with \( p, q > r \), as \( \sqsupseteq_{m+1}^m \) is edge-witnessing, showing that \( \sqcap_{G_m} \subseteq \wedge_m \). Thus \( \mathbb{P} \) is a \( \mathbf{K} \)-poset that is \( \mathbf{K} \)-subabsorbing, by construction.

3.3. Compatibility. Throughout the rest of this section we assume that

(1) K is a fixed subcategory of the graph category G.

(2) \(\mathbb{P}\) is a K-regular K-subabsorbing K-poset.

In particular, \(\mathbb{P}\) is regular so \(\mathbb{S}\mathbb{P}\) is Hausdorff, by [BBV23, Corollary 2.40]. As \(\mathbf{K}\)-morphisms are co-injective, each level \(\mathbb{P}_n\) is a minimal cap, by [BBV23, Proposition 1.21], which thus yields a minimal open cover \(\{p_{\mathbb{S}} : p \in \mathbb{P}_n\}\) of \(\mathbb{S}\mathbb{P}\), by [BBV23, Proposition 2.8]. In particular, \(p_{\mathbb{S}} \neq \emptyset\), for all \(p \in \mathbb{P}\), i.e. \(\mathbb{P}\) is also a prime \(\omega\)-poset.

We are interested in continuous maps on \(\mathbb{S}\mathbb{P}\) that are compatible with the given subcategory \(\mathbf{K}\). First, for any finite \(\sqsupset \subseteq \mathbb{P} \times \mathbb{P}\), let \(\sqsupset^{\mathbf{C}}\) denote the basic open subset of \(\mathbf{C}_{\mathbb{S}\mathbb{P}}^{\mathbb{S}\mathbb{P}}\) corresponding to the open subset \(\sqsupset^{\mathbf{S}}\) of strong refiners defined in (2.3), i.e.

\[
\sqsupset^ {\mathbf {C}} = \left\{\phi \in \mathbf {C} _ {\mathbb {S P}} ^ {\mathbb {S P}}: \sqsupset \subseteq \sqsupset_ {\phi} \right\} = \left\{\phi \in \mathbf {C} _ {\mathbb {S P}} ^ {\mathbb {S P}}: \forall p, q (q \sqsupset p \Rightarrow q _ {\mathsf {S}} \supseteq \phi [ \overline {{p _ {\mathsf {S}}}} ]) \right\}.
\]

Definition 3.14. The \(\mathbf{K}\)-compatible functions on \(\mathbb{SP}\) are given by

\[
\mathbf {C} _ {\mathbf {K}} = \bigcap_ {m \in \omega} \bigcup_ {n \geq m} \bigcup_ {\sqsupset \in \mathbf {K} _ {n} ^ {m}} \sqsupset^ {\mathbf {C}} = \{\phi \in \mathbf {C} _ {\mathbb {S P}} ^ {\mathbb {S P}}: \forall m \in \omega \exists \sqsupset \in \mathbf {K} _ {n} ^ {m} (\sqsupset \subseteq \sqsupset_ {\phi}) \}.
\]

Corollary 2.32 tells us that \(\mathbf{C}_{\mathbf{K}}\) has another basis consisting of open sets \(\leftarrow_{\mathbf{K}}\), for relations \(\leftarrow \subseteq \mathbb{P}_n \times \mathbb{P}_n\) defined on levels of \(\mathbb{P}\), where

\[
\leftarrow_ {\mathbf {K}} = \leftarrow_ {\mathbf {C}} \cap \mathbf {C} _ {\mathbf {K}} = \left\{\phi \in \mathbf {C} _ {\mathbf {K}}: \phi \subseteq \leftarrow_ {\mathsf {S}} \right\}.
\]

However, many of these basic sets can be empty and our interest in \(\mathbf{K}\)-like relations stems from the fact that these are the only ones for which \(\leftarrow_{\mathsf{S}}\) can contain any \(\phi \in \mathbf{C}_{\mathbf{K}}\). We can even revert to a relation below \(\leftarrow\) which not just \(\mathbf{K}\)-like but widely \(\mathbf{K}\)-subfactorisable (note every widely \(\mathbf{K}\)-subfactorisable \(\leftarrow\) is \(\mathbf{K}\)-like, by Lemma 3.8, as \(\mathbb{P}\) is \(\mathbf{K}\)-subabsorbing).

Let us denote the widely \(\mathbf{K}\)-subfactorisable relations on any \(G \in \mathbf{K}\) by

\[
G ^ {\mathbf {K}} = \{\leftarrow \subseteq G \times G: \forall > \in \mathbf {K} _ {H} ^ {G} \left(<   \circ \leftarrow \circ > \text {subfactors into some} \sqsupset , \exists \in \mathbf {K} _ {I} ^ {H}\right) \}.
\]

Lemma 3.15. For any \(m, m' \in \omega\), \(\leftarrow \subseteq \mathbb{P}_m \times \mathbb{P}_m\), \(\leftarrow'\subseteq \mathbb{P}_{m'} \times \mathbb{P}_{m'}\) and \(\phi \in \leftarrow_{\mathbf{K}} \cap \leftarrow_{\mathbf{K}}'\), we have some \(n > \max(m, m')\) and \(\leftarrow \subseteq \mathbb{P}_n^{\mathbf{K}}\) with \(\phi \in \leftarrow_{\mathbf{K}}\), \(\leftarrow \leq \leftarrow\) and \(\leftarrow \leq \leftarrow'\)

Proof. By Proposition 2.31, we have \(k > m\) with \(\leftarrow_{k}^{\phi} \triangleleft_{k} \leftarrow\) and \(\leftarrow_{k}^{\phi} \triangleleft_{k} \leftarrow'\). By Proposition 3.11, we have \(\mathbb{P}\)-subamalgamable \(\Rightarrow \in \mathbf{K}_l^k\). As \(\phi \in \mathbf{C}_{\mathbf{K}}\), we have \(\sqsupset \in \mathbf{K}_n^l\) with \(\sqsupset \subseteq \sqsupset_{\phi}\). By Propositions 2.19 and 2.36, \(\phi \subseteq (\sqsupset \circ \leq_n^l)_{\mathsf{S}}\) and \(\sqsupset \circ \leq_n^l \subseteq \leftarrow_l^{\phi}\). Now set

\[
\ll^ {\prime} = \ll \circ > \circ \sqsupset \circ \leq_ {n} ^ {l} \circ <   \circ > \quad \text {and} \quad \ll = \leq_ {l} ^ {k} \circ > \circ \sqsupset \circ \leq_ {n} ^ {l} \circ <   \circ \geq_ {l} ^ {k}.
\]

As \(\supset \circ \leq_{n}^{l} \subseteq \leftarrow'\), it follows that \(\leftarrow'\) is also a \(\mathbf{K}\)-like relation satisfying \(\phi \subseteq \leftarrow_{\mathfrak{S}}'\). The same then applies to the even larger relation \(\leftarrow\), which is also widely \(\mathbf{K}\)-subfactorisable, by Lemma 3.3. As \(\supset \circ \leq_{n}^{l} \subseteq \leftarrow_{l}^{\phi} \subseteq \leq_{l}^{k} \circ \leftarrow_{k}^{\phi} \circ \geq_{l}^{k}\) and \(\leftarrow_{k}^{\phi} \triangleleft_{k} \leftarrow\),

\[
\leftarrow \subseteq \leq_ {l} ^ {k} \circ \geq_ {l} ^ {k} \circ \leq_ {l} ^ {k} \circ \leftarrow_ {k} ^ {\phi} \circ \geq_ {l} ^ {k} \circ \leq_ {l} ^ {k} \circ \geq_ {l} ^ {k} \subseteq \leq_ {l} ^ {k} \circ \wedge_ {k} \circ \leftarrow_ {k} ^ {\phi} \circ \wedge_ {k} \circ \geq_ {l} ^ {k} \subseteq \leq_ {l} ^ {m} \circ \leftarrow \circ \geq_ {l} ^ {m}.
\]

This shows that \(\leftarrow \leq \leftarrow\) and, likewise, \(\leftarrow \leq \leftarrow'\) as well.

Corollary 3.16. The open sets \(\{\leftarrow_{\mathbf{K}}:n\in \omega\) and \(\leftarrow \in \mathbb{P}_n^{\mathbf{K}}\}\) form a basis for \(\mathbf{C}_{\mathbf{K}}\).

Proof. This is immediate from Lemma 3.15 once we note \(\leftarrow \leq \leftarrow\) implies \(\leftarrow_{\mathbf{K}} \subseteq \leftarrow_{\mathbf{K}}\).

□