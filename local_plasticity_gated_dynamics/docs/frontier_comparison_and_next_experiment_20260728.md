# Frontier comparison and next-experiment decision (2026-07-28)

## Decision

The project has retained the abstract question of whether a low-dimensional
signal can control a high-dimensional carrier, but the active evidence does
not support the original physical-low-rank or biological E/I mechanism chain.
The next experiment must therefore localize the remaining synthetic failure
before adding carrier scale or neural claims.

The highest-value question is not whether another gate can be added. It is:

> When the event and Q/R signals are separately made correct, can the existing
> continuation/reset and Kalman-gain actuators convert them into early and
> late held-out utility?

This is narrower than the abandoned Exp42 entry-gated plan. It is a causal
diagnostic on new tapes, not a confirmation or a claim upgrade.

## Exp43 outcome update

The prospective 2x2 audit has now completed on 8/8 development seeds with no
failure and no access to reserved formal seeds. Oracle-both improves learned
NLL by +0.090067 in 8/8 seeds. Oracle event improves the next-eight post-jump
NLL by +0.031488, and oracle Q/R improves overall NLL by +0.076979, both in
8/8. The existing actuator can therefore use correct signals.

The deployable gate nevertheless fails. Total variance minus learned is only
+0.013021 NLL with 6/8 positive seeds and 5/8 positive latent-MSE seeds; cell
`110` remains negative. In that cell the learned controller underestimates Q
and overestimates R, while oracle Q/R restores +0.125209 NLL in 8/8 seeds.
The development conclusion is consequently a localization result: both
inference paths have headroom, Q/R is the dominant aggregate bottleneck, and
event inference matters after jumps. It is not a controller confirmation.

## Exp44 outcome update

The external behavior gate now fails on 223/223 participants. Factorized NLL
gain is +0.001044 over fixed and +0.006105 over total uncertainty, but both
participant-bootstrap intervals cross zero; both registered MSE clauses also
fail. The selected symmetric local-EM candidate has nearly constant gain and
its Q effect has the wrong sign (-0.000702). In an outcome-exposed descriptive
replay, human blockwise gains retain the expected Q (+0.042389) and R
(+0.055021) effects. Thus the external task contains the target computation,
but the current causal controller does not recover it with held-out utility.
Experiment 2 and POPGym remain locked.

## What the existing experiments establish

| Evidence | What is established | What is not established |
|---|---|---|
| Exp08 | Synapse-wise masking, Dale projection, and normalization can preserve high physical rank despite low-dimensional credit | Low-dimensional feedback does not imply low-rank physical connectivity |
| Exp09/21 | A leakage-safe hidden belief can modulate a frozen high-rank E/I receiver and alter its effective trajectories | Participating recurrent plasticity, homeostatic necessity, or a biological MD implementation |
| Exp24/26/29/31/32 | Constructed tasks can favor different actuator motifs, and sparse reward can select among fixed motifs in bounded settings | General real-task superiority or a learned participating high-rank carrier |
| Exp35--38 | Prefix accumulation can help stable video streams; fixed long memory harms switching; registered change controllers fail | Generic adaptive forgetting or fast release |
| Exp39 | A local three-coordinate filter selected on isolated factors improves average NLL and latent MSE on unseen joint synthetic cells | Uniform cell-wise benefit, clean h/Q/R recovery, or fast post-switch release |
| Exp40 | Semi-Markov structure improves hidden-block decoding on exposed IBL behavior | Incremental held-out choice utility or a neural mechanism |
| Exp41 | Lag covariance contains enough information to order matched Q/R regimes | Timely predictive utility: the estimator loses early and overall to online EM and total variance |
| Exp43 | The fixed actuator converts correct event and Q/R signals into held-out headroom | The all-learned controller does not uniformly beat total variance/IMM; cell `110` still fails |

The repeated negative boundary is therefore specific: useful stable evidence
accumulation exists, but the learned controller has not released or redirected
state quickly enough at changes.

## Comparison with current primary work

Piray and Daw show why the Q/R distinction is computationally meaningful:
volatility should raise learning rate, whereas observation stochasticity
should lower it. Their human task also shows that the distinction can be
extracted from temporal covariance, with participants as the statistical unit
([Nature Communications 2024](https://www.nature.com/articles/s41467-024-53459-z)).
Exp41 agrees with the identifiability part but fails the utility part. This
makes the public human dataset a stronger next behavioral bridge than another
visible-category video stream, but only after the synthetic actuator ceiling
is localized.

Selective state-space models already make state propagation input dependent;
Mamba explicitly learns when to propagate or forget
([Gu and Dao 2023](https://arxiv.org/abs/2312.00752)). Gated DeltaNet further
shows that rapid erasure and targeted delta writing are complementary
([ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4904fad153f6434a7bcf04465d4be2cc-Abstract-Conference.html)).
Consequently, generic gating, selective forgetting, or a write/erase split is
not a novelty claim. A defensible contribution would require a falsifiable,
causal mapping from distinct uncertainty sources to distinct actions, without
online backpropagation.

Bayesian online changepoint detection already supplies a principled run-length
posterior ([Adams and MacKay 2007](https://arxiv.org/abs/0710.3742)), and robust
generalized-Bayes variants address outliers and scale
([Altamirano et al. 2023](https://arxiv.org/abs/2302.04759)). These are required
fast-event baselines or ceilings, not names for the project's novelty.

Efficient online recurrent learning is also an active comparator class:
Recurrent Trace Units make RTRL practical for restricted linear recurrent
architectures and perform strongly in partially observable control
([NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1e616bde0438cb10cb6adf076ae7d336-Abstract-Conference.html)).
Any later local-learning claim must therefore compare both performance and
state/update cost against an online-gradient ceiling, while keeping that
ceiling outside the local mechanism.

Recent E/I work gives a plausible later mechanism but does not rescue the
current frozen carrier. Co-dependent excitatory and inhibitory plasticity can
produce stable, input-sensitive recurrent dynamics through local neighboring
interactions
([Agnes and Vogels 2024](https://www.nature.com/articles/s41593-024-01597-4)).
This motivates a future participating-carrier test only after downstream
utility exists; it is not evidence that the present controller uses that
mechanism.

Finally, IBL establishes an important real-data target: uncued 0.2/0.8 priors
switch in variable 20--100-trial blocks, subjective priors explain behavior,
and prior-related activity is widespread
([Nature 2025](https://www.nature.com/articles/s41586-025-09226-1)). The paper
also demonstrates why movement and embodiment controls are mandatory. Exp40's
decoding result is therefore a necessary but insufficient first layer.

## Architectural correction

Exp39 already contains more of the proposed fast/slow architecture than the
verbal diagnosis implied. Its audited jump step computes:

1. an immediate posterior probability of reset versus continuation;
2. a continuation Kalman gain controlled by Q/R;
3. a reset posterior; and
4. a mixture of the two states.

Exp41 changes the slow Q/R estimator but deliberately reuses this actuator.
Thus another implementation of "fast event posterior + slow Q/R + separate
write/release" would be semantically redundant. Exp43 supplied the missing
exchange audit: the actuator has headroom, while both learned inference paths
remain limiting.

## Priority ladder

1. **Archive Exp43 without a formal run.** Its development gate localizes
   opportunity but the deployable conjunction fails; reserved seeds remain
   untouched.
2. **Archive the completed Piray--Daw test.** Its participant-held gate fails
   despite a valid empirical Q/R behavior signal. Do not run the same-tape
   Experiment-2 cohort or characterize the result as partial support.
3. **Change inference, not the actuator.** Any successor must be a new
   prospective contract on untouched outcomes. A causal participant-specific
   calibration layer may be tested against fixed-person, total-uncertainty,
   and hierarchical behavioral baselines, but it cannot rescue Exp44.
   Pre-register cell-`110` recovery,
   post-change utility, calibration, and update-cost endpoints. Keep the
   current write/release algebra fixed so improvement remains identifiable.
4. **Keep POPGym and real neural scaling conditional on behavioral utility.**
   Cross-fit
   subjective state by animal/session before neural encoding; regress movement
   and pose; use animal/session bootstrap.
5. **Add participating E/I plasticity last.** Match functional update budgets
   and test effective-dynamics closure, stability, and behavior. Do not revive
   physical matrix rank as the endpoint.

## Stop conditions

- If the oracle-both actuator has no meaningful NLL and latent-MSE headroom,
  the current task/actuator pairing is not worth scaling.
- If oracle event helps but learned event does not, change inference is the
  bottleneck; do not tune Q/R windows.
- If oracle Q/R helps but learned Q/R does not, slow inference is the
  bottleneck; use the human volatility/stochasticity task before IBL.
- If both one-axis oracles help but the all-learned controller does not beat
  total variance and seen-mode IMM, the contribution remains a failure
  localization result, not a new controller.
- No synthetic result licenses a neural or E/I mechanism claim.

Exp43 reaches exactly the fourth condition. The next experiment must therefore
change the data/inference bridge and use untouched participants; it must not
be presented as a post-hoc rescue on seeds `43000--43007`.
