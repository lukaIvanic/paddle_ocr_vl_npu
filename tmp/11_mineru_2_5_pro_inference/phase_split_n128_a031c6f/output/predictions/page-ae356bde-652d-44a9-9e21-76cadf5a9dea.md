27

This model in [35] is equivalent to our construction when the sign is flipped. Notice that

\[
U _ {i} = Z _ {8 i + 1} Z _ {8 i + 3} Z _ {8 i + 5} Z _ {8 i + 7}
\]

is a symmetric operator, we have

\[
\left(- \sum_ {I} \tilde {P} _ {I, I + 1}\right) \prod_ {i = 1} ^ {[ L / 8 ]} U _ {i} = \left\{ \begin{array}{l l} \prod_ {i = 1} ^ {[ L / 8 ]} U _ {i} \left(- \sum_ {I} P _ {I, I + 1}\right) & L \equiv 0 \mod 8 \\ \prod_ {i = 1} ^ {[ L / 8 ]} U _ {i} \left(- \sum_ {I} P _ {I, I + 1} - \bar {P} _ {L / 4 - 1, L / 4} - \bar {P} _ {L / 4, L / 4 + 1}\right) & L \equiv 4 \mod 8 \end{array} \right.
\]

where

\[
P _ {I, I + 1} = \frac {1}{4} (1 + Z _ {4 I - 3} X _ {4 I - 1} Z _ {4 I + 1}) (1 + Z _ {4 I - 1} X _ {4 I + 1} Z _ {4 I + 3}),
\]

\[
\bar {P} _ {L / 4 - 1, L / 4} = \frac {1}{4} (1 + Z _ {L - 7} X _ {L - 5} Z _ {L - 3}) (1 - Z _ {L - 5} X _ {L - 3} Z _ {L - 1}),
\]

\[
\bar {P} _ {L / 4, 1} = \frac {1}{4} (1 - Z _ {L - 3} X _ {L - 1} Z _ {1}) (1 + Z _ {L - 1} X _ {1} Z _ {3}).
\]

Thus the local projector is \( P_{I,I + 1} \) with exceptions on the boundary, which correspond to local symmetric defects. Since a phase is at the thermodynamic limit, the phase realized in [35] is equivalent to

\[
H = - \sum_ {I} P _ {I, I + 1}.
\]

The component of \(P_{I,I + 1}\) in each sector can be expanded as

\[
\begin{array}{l} P _ {e} = \frac {1}{4} \begin{array}{c c c c c c} & e & e & s & s & r ^ {2} & r ^ {2} & s r ^ {2} & s r ^ {2} \\ \hline & e & e & 1 & 1 & - 1 & 1 \\ & s & s & 1 & 1 & - 1 & 1 \\ & r ^ {2} & r ^ {2} & - 1 & - 1 & 1 & - 1 \\ & s r ^ {2} & s r ^ {2} & 1 & 1 & - 1 & 1 \end{array} , \\ P _ {s} = \frac {1}{4} \begin{array}{c c c c c} & e s & s e & s r ^ {2} r ^ {2} & r ^ {2} s r ^ {2} \\ \hline e s & 1 & 1 & - 1 & 1 \\ s e & 1 & 1 & - 1 & 1 \\ s r ^ {2} r ^ {2} & - 1 & - 1 & 1 & - 1 \\ r ^ {2} s r ^ {2} & 1 & 1 & - 1 & 1 \end{array} , \\ P _ {r ^ {2}} = \frac {1}{4} \begin{array}{c c c c c} & e r ^ {2} & r ^ {2} e & s s r ^ {2} & s r ^ {2} s \\ \hline e r ^ {2} & 1 & 1 & 1 & - 1 \\ r ^ {2} e & 1 & 1 & 1 & - 1 \\ s s r ^ {2} & 1 & 1 & 1 & - 1 \\ s r ^ {2} s & - 1 & - 1 & - 1 & 1 \end{array} , \\ P _ {s r ^ {2}} = \frac {1}{4} \begin{array}{c c c c c} & e s r ^ {2} & s r ^ {2} e & s r ^ {2} & r ^ {2} s \\ \hline e s r ^ {2} & 1 & 1 & 1 & - 1 \\ s r ^ {2} e & 1 & 1 & 1 & - 1 \\ s r ^ {2} & 1 & 1 & 1 & - 1 \\ r ^ {2} s & - 1 & - 1 & - 1 & 1 \end{array} \\ \end{array}
\]

Define

\[
m _ {e} = \frac {\left| \begin{array}{c c c c c} e & e & s & s & r ^ {2} & r ^ {2} & s r ^ {2} & s r ^ {2} \\ \hline e & 1 & 1 & - 1 & 1 \end{array} \right|}{s} \text {,} \quad m _ {s} = \frac {\left| \begin{array}{c c c c c} e & s & s & e & s r ^ {2} & r ^ {2} & r ^ {2} & s r ^ {2} \\ \hline 1 & 1 & - 1 & 1 \end{array} \right|}{s},
\]

\[
m _ {r ^ {2}} = \frac {1}{r ^ {2}} \left| \begin{array}{c c c c} e r ^ {2} & r ^ {2} e & s s r ^ {2} & s r ^ {2} s \\ 1 & 1 & 1 & - 1 \end{array} \right., \quad m _ {s r ^ {2}} = \frac {1}{s r ^ {2}} \left| \begin{array}{c c c c} e s r ^ {2} & s r ^ {2} e & s r ^ {2} & r ^ {2} s \\ 1 & 1 & 1 & - 1 \end{array} \right..
\]

Then

\[
P = \frac {1}{4} m ^ {\dagger} m.
\]

We can then extract the cocycle gauge by \( m \):

\[
\omega (r ^ {2}, r ^ {2}) = \omega (s r ^ {2}, r ^ {2}) = \omega (r ^ {2}, s) = \omega (s r ^ {2}, s) = - 1, \quad \omega (\mathrm{others}) = 1.
\]

Thus the Hamiltonian of [35], for either \( L = 0 \mod 8 \) or \( L = 4 \mod 8 \), matches our model locally

\[
H = - \frac {1}{4} \sum_ {i} m _ {i, i + 1} ^ {\dagger} m _ {i, i + 1}.
\]

And the MPO \(A_{\sigma}\) acts on ground state as \(\mathrm{Tr}(A_{\sigma})|e\rangle = 2|e\rangle\).

The third model is constructed similarly. After applying the CZ gate, the ground state is stabilized by the generator

\[
- X _ {2 n - 1} = 1, \quad - Z _ {2 n - 2} X _ {2 n} Z _ {2 n + 2} = 1.
\]