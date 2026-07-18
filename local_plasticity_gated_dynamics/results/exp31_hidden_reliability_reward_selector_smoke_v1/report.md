# Exp31 hidden-demand reward-only selector

- Profile: `smoke`; seeds: 5.
- Scale decision: **scale-authorized**.
- Claim classification: **inconclusive**.
- Reward-only minus train-fixed: +0.0443.
- Reward-only minus matched random: +0.0444.
- Oracle gain retained: 0.453.
- Reliability crossover: +0.4169.
- Associative minus query-shuffled: +0.3497.
- Memory accuracy/pressure Spearman: -0.982.

The primary score includes the forced-exploration prefix. The local
selector never receives true reliability, task descriptors, unexecuted
rewards, or a candidate-utility vector. The oracle is an explicitly
labelled train-map upper bound. This panel isolates controller
identifiability; it does not establish high-rank E/I carrier dynamics
or real-data validity.
