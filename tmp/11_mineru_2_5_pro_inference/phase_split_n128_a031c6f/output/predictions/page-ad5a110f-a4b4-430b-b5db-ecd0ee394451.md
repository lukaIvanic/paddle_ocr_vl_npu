KOLMOGOROV EQUATIONS FOR 2D SCBF EQUATIONS

29

\[
\begin{array}{l} \leq C \lambda_ {1} ^ {\delta (2 - 8 \delta)} \int_ {\{\| \boldsymbol {x} \| _ {\mathbb {V}} \geq \varepsilon^ {- 1} \}} e ^ {\kappa \| \boldsymbol {x} \| _ {\mathbb {H}} ^ {2}} \| \mathrm{A} ^ {\delta} \boldsymbol {x} \| _ {\mathbb {H}} ^ {8 \delta} \frac {\| \mathrm{A} ^ {\delta + \frac {1}{2}} \boldsymbol {x} \| _ {\mathbb {H}} ^ {2}}{\| \boldsymbol {x} \| _ {\mathbb {V}} ^ {8 \delta - 2}} \eta (\mathrm{d} \boldsymbol {x}) \\ \leq C \lambda_ {1} ^ {\delta (2 - 8 \delta)} \varepsilon^ {8 \delta - 2} \int_ {\mathbb {H}} e ^ {\kappa \| \boldsymbol {x} \| _ {\mathbb {H}} ^ {2}} \| \mathrm{A} ^ {\delta} \boldsymbol {x} \| _ {\mathbb {H}} ^ {8 \delta} \| \mathrm{A} ^ {\delta + \frac {1}{2}} \boldsymbol {x} \| _ {\mathbb {H}} ^ {2} \eta (\mathrm{d} \boldsymbol {x}) \\ \leq C \varepsilon^ {8 \delta - 2}, \tag {6.18} \\ \end{array}
\]

provided \(\frac{1}{4} < \delta < \frac{1}{2}\), and thus (6.16) follows by an application of the Dominated convergence theorem. Let us now calculate by using (5.3) and (6.12)

\[
\begin{array}{l} \int_ {\mathbb {H}} | (\mathcal {C} _ {\varepsilon} (\boldsymbol {x}) - \mathcal {C} (\boldsymbol {x}), \mathrm{D} _ {\boldsymbol {x}} \varphi_ {\varepsilon} (\boldsymbol {x})) | ^ {2} \eta (\mathrm{d} \boldsymbol {x}) \\ = \int_ {\{\| \boldsymbol {x} \| _ {\mathbb {V}} \geq \varepsilon^ {- 1} \}} \left| \left(\mathcal {C} _ {\varepsilon} (\boldsymbol {x}) - \mathcal {C} (\boldsymbol {x}), \mathrm{D} _ {\boldsymbol {x}} \varphi_ {\varepsilon} (\boldsymbol {x})\right) \right| ^ {2} \eta (\mathrm{d} \boldsymbol {x}) \\ \leq \int_ {\{\| \boldsymbol {x} \| _ {\mathbb {V}} \geq \varepsilon^ {- 1} \}} \left| \frac {1 - \varepsilon^ {r + 1} \| \boldsymbol {x} \| _ {\mathbb {V}} ^ {r + 1}}{\varepsilon^ {r + 1} \| \boldsymbol {x} \| _ {\mathbb {V}} ^ {r + 1}} \right| \| \mathcal {C} (\boldsymbol {x}) \| _ {\mathbb {H}} ^ {2} \| \mathrm{D} _ {\boldsymbol {x}} \varphi_ {\varepsilon} (\boldsymbol {x}) \| _ {\mathbb {H}} ^ {2} \eta (\mathrm{d} \boldsymbol {x}) \\ \leq C e ^ {- \delta t} \| f \| _ {1} ^ {2} \int_ {\{\| \boldsymbol {x} \| _ {\mathbb {V}} \geq \varepsilon^ {- 1} \}} e ^ {\kappa \| \boldsymbol {x} \| _ {\mathbb {H}} ^ {2}} \| \mathcal {C} (\boldsymbol {x}) \| _ {\mathbb {H}} ^ {2} \eta (\mathrm{d} \boldsymbol {x}), \\ \end{array}
\]

so that (6.15) follows if

\[
\lim _ {\varepsilon \rightarrow 0} \int_ {\{\| \boldsymbol {x} \| _ {\mathbb {V}} \geq \varepsilon^ {- 1} \}} e ^ {\kappa \| \boldsymbol {x} \| _ {\mathbb {H}} ^ {2}} \| \mathcal {C} (\boldsymbol {x}) \| _ {\mathbb {H}} ^ {2} \eta (\mathrm{d} \boldsymbol {x}) = 0. \tag {6.19}
\]

We calculate for \( r = 2 \) by using (5.4), Ladyzhenskaya's and Poincaré's inequalities

\[
\| \mathcal {C} (\boldsymbol {x}) \| _ {\mathbb {H}} \leq \| \boldsymbol {x} \| _ {\widetilde {\mathbb {L}} ^ {4}} ^ {2} \leq \sqrt {2} \| \boldsymbol {x} \| _ {\mathbb {H}} \| \boldsymbol {x} \| _ {\mathbb {V}} \leq \frac {\sqrt {2}}{\lambda_ {1}} \| \boldsymbol {x} \| _ {\mathbb {V}} ^ {2} \leq \frac {\sqrt {2}}{\lambda_ {1}} \| \mathrm{A} ^ {\delta} \boldsymbol {x} \| _ {\mathbb {H}} ^ {4 \delta} \| \mathrm{A} ^ {\delta + \frac {1}{2}} \boldsymbol {x} \| _ {\mathbb {H}} ^ {2 (1 - 2 \delta)}.
\]

Then proceeding in a similar way as we did in (6.17), one can conclude (6.19) provided \(\frac{1}{4} < \delta < \frac{1}{2}\). Along with Lemma 5.1, estimates (5.35), (6.18) and (2.5), it follows for \(r = 3\) that

\[
\begin{array}{l} \int_ {\{\| \boldsymbol {x} \| _ {\mathbb {V}} \geq \varepsilon^ {- 1} \}} e ^ {\kappa \| \boldsymbol {x} \| _ {\mathbb {H}} ^ {2}} \| \mathcal {C} (\boldsymbol {x}) \| _ {\mathbb {H}} ^ {2} \eta (\mathrm{d} \boldsymbol {x}) \\ \leq C \int_ {\{\| \boldsymbol {x} \| _ {\mathbb {V}} \geq \varepsilon^ {- 1} \}} e ^ {\kappa \| \boldsymbol {x} \| _ {\mathbb {H}} ^ {2}} \| \boldsymbol {x} \| _ {\mathbb {V}} ^ {4} \| \boldsymbol {x} \| _ {\mathbb {H}} ^ {2} \eta (\mathrm{d} \boldsymbol {x}) \\ \leq \frac {C}{\lambda_ {1} ^ {2 \delta}} \int_ {\{\| \boldsymbol {x} \| _ {\mathbb {V}} \geq \varepsilon^ {- 1} \}} e ^ {\kappa \| \boldsymbol {x} \| _ {\mathbb {H}} ^ {2}} \| \mathrm{A} ^ {\delta} \boldsymbol {x} \| _ {\mathbb {H}} ^ {8 \delta} \| \mathrm{A} ^ {\delta + \frac {1}{2}} \boldsymbol {x} \| _ {\mathbb {H}} ^ {4 - 8 \delta} \| \mathrm{A} ^ {\delta} \boldsymbol {x} \| _ {\mathbb {H}} ^ {2} \eta (\mathrm{d} \boldsymbol {x}) \\ \leq \frac {C}{\lambda_ {1} ^ {2 \delta}} \int_ {\{\| \boldsymbol {x} \| _ {\mathbb {V}} \geq \varepsilon^ {- 1} \}} e ^ {\kappa \| \boldsymbol {x} \| _ {\mathbb {H}} ^ {2}} \| \mathrm{A} ^ {\delta} \boldsymbol {x} \| _ {\mathbb {H}} ^ {8 \delta + 2} \frac {\| \mathrm{A} ^ {\delta + \frac {1}{2}} \boldsymbol {x} \| _ {\mathbb {H}} ^ {2}}{\| \mathrm{A} ^ {\delta + \frac {1}{2}} \boldsymbol {x} \| _ {\mathbb {H}} ^ {8 \delta - 2}} \eta (\mathrm{d} \boldsymbol {x}) \\ \leq C \lambda_ {1} ^ {\delta (2 - 8 \delta)} \int_ {\{\| \boldsymbol {x} \| _ {\mathbb {V}} \geq \varepsilon^ {- 1} \}} e ^ {\kappa \| \boldsymbol {x} \| _ {\mathbb {H}} ^ {2}} \| \mathrm{A} ^ {\delta} \boldsymbol {x} \| _ {\mathbb {H}} ^ {8 \delta + 2} \frac {\| \mathrm{A} ^ {\delta + \frac {1}{2}} \boldsymbol {x} \| _ {\mathbb {H}} ^ {2}}{\| \boldsymbol {x} \| _ {\mathbb {V}} ^ {8 \delta - 2}} \eta (\mathrm{d} \boldsymbol {x}) \\ \leq C \lambda_ {1} ^ {\delta (2 - 8 \delta)} \varepsilon^ {8 \delta - 2} \int_ {\mathbb {H}} e ^ {\kappa \| \boldsymbol {x} \| _ {\mathbb {H}} ^ {2}} \| \mathrm{A} ^ {\delta} \boldsymbol {x} \| _ {\mathbb {H}} ^ {8 \delta + 2} \| \mathrm{A} ^ {\delta + \frac {1}{2}} \boldsymbol {x} \| _ {\mathbb {H}} ^ {2} \eta (\mathrm{d} \boldsymbol {x}) \\ \leq C \varepsilon^ {8 \delta - 2}, \\ \end{array}
\]