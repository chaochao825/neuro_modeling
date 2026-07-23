# Real-task frontier and scale plan (2026-07-24)

## Decision

The next evidence-bearing task is causal personalized video recognition on
ORBIT, not another synthetic actuator endpoint and not a solver-dominated ARC,
maze, or Sudoku score. ORBIT supplies raw observations, few-shot clean-video
support, cluttered query videos, official user-disjoint splits, and a protocol
that permits only the current and preceding query frames. It therefore tests
the intended combination of small-sample personalization, temporal state, and
unseen-person generalization end to end.

The choice is task-first. A participating Dale E/I carrier is scientifically
useful only after a controller improves real held-out behavior; adding one to a
task-space failure would increase mechanism complexity without rescuing the
identifiability problem.

## Relationship to prior work

- The [ORBIT benchmark](https://openaccess.thecvf.com/content/ICCV2021/html/Massiceti_ORBIT_A_Real-World_Few-Shot_Dataset_for_Teachable_Object_Recognition_ICCV_2021_paper.html)
  contains 3,822 videos of 486 personalized objects recorded by 77 blind or
  low-vision participants. Its [official implementation](https://github.com/microsoft/ORBIT-Dataset)
  freezes 50 tasks for each of 17 test users, samples up to 200 valid frames per
  clutter video, and allows current and past frames but not future frames or
  query labels.
- [LITE](https://proceedings.neurips.cc/paper/2021/hash/cc1aa436277138f61cda703991069eaf-Abstract.html)
  makes large-image episodic meta-learning memory efficient. It is the right
  trained-representation comparison, but it does not answer whether a small
  online controller can choose among reusable computations after the encoder
  is frozen.
- The official repository reports 67.48% test frame accuracy for cosine
  ProtoNet with EfficientNet-B0 and 75.38% with ViT-B/32 under LITE-trained
  encoders. These are external reference points, not directly interchangeable
  with the current ImageNet-frozen feature audit.
- [Strong parameter-efficient few-shot baselines](https://ojs.aaai.org/index.php/AAAI/article/view/28978)
  motivate eventual adapter/FiLM comparisons. They do not justify adding
  gradient-based test personalization to the local-learning main condition.
- [Temporal test-time adaptation with state-space models](https://openreview.net/forum?id=HFETOmUtrV)
  supports the broader idea that label-free filtering of time-varying feature
  statistics can help under temporal shift. Exp34 is narrower: it uses
  within-video prediction persistence and resets at every video boundary.
- [Test-time training as an RNN](https://arxiv.org/abs/2407.04620),
  [Titans](https://arxiv.org/abs/2501.00663), and
  [Gated DeltaNet](https://arxiv.org/abs/2412.06464) show how online parameter
  updates or fast weights can act as recurrent memory. Their large
  end-to-end-trained sequence models are architectural context, not evidence
  for local biological credit or appropriate strong baselines for the present
  frozen-encoder audit.

## Falsifiable benchmark ladder

1. **Mechanism-matched audit.** Share the exact frozen EfficientNet-B0
   embeddings, support/query samples, task order, and actuator predictions
   across prototype, diagonal gain, delta memory, temporal accumulation,
   validation-selected fixed, causal consensus, memoryless, delayed, and
   oracle conditions. Charge the full actuator-bank event cost to consensus.
2. **Untouched public test.** Select the fixed comparator on all six validation
   users and evaluate the frozen gate on 50 tasks for every one of the 17 test
   users. Average algorithmic seeds within user; use user-level bootstrap and
   exact paired sign flips with Holm correction.
3. **Competitive representation check.** Only if step 2 supports, repeat the
   paired controller audit with the official ViT-B/32 or CLIP feature path and
   compare against official LITE ProtoNet, CNAPs, and fine-tuning numbers. This
   separates controller value from encoder value.
4. **Compute-aware controller.** The current gate evaluates all four motifs.
   A sparse early-exit or confidence-triggered gate must match accuracy at a
   registered event/MAC budget before any efficiency claim is allowed.
5. **Mechanistic carrier.** Only after real task-space utility survives steps
   2--4 should the selected motifs be realized in a stable high-rank Dale E/I
   carrier and audited for closure, normal perturbation decay, saturation, and
   matched control budget.
6. **Neural validation.** Multi-session neural claims remain separate and
   fail-closed until the canonical CompositionalTasks or IBL bundle satisfies
   the animal/session-level data contract.

## Novelty boundary

The plausible contribution is not a new universal RNN or a claim of ORBIT
state of the art. It is a causal, query-label-free, backpropagation-free belief
controller that reuses heterogeneous few-shot computations and is tested on
unseen people with causal ablations. A positive test result would support this
bounded capability. A negative result would oppose the registered consensus
rule and end this scale path without being hidden by synthetic or matrix-rank
diagnostics.
