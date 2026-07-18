# Exp31 hidden-demand reward-only selector

- Profile: `formal`; seeds: 30.
- Scale decision: **formal-complete**.
- Claim classification: **support**.
- Reward-only minus train-fixed: +0.0472.
- Reward-only minus matched random: +0.0479.
- Oracle gain retained: 0.473.
- Reliability crossover: +0.3961.
- Associative minus query-shuffled: +0.3467.
- Memory accuracy/pressure Spearman: -0.971.

The primary score includes the forced-exploration prefix. The local
selector never receives true reliability, task descriptors, unexecuted
rewards, or a candidate-utility vector. The oracle is an explicitly
labelled train-map upper bound. This panel isolates controller
identifiability; it does not establish high-rank E/I carrier dynamics
or real-data validity.
