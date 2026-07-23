# Literature Snapshot: Few-Shot Video Personalization and Reusable Dynamics

Verified 2026-07-24 from primary papers, venue pages, and official repositories.

## Closest task line

- ORBIT defines user-disjoint few-shot personalization from labelled clean videos to clutter-video frame recognition. Its updated official protocol samples 50 tasks per test user, uses frame accuracy as the main endpoint, and allows only the current and preceding query frames.
- LITE makes large-image episodic meta-learning practical and reports strong ORBIT results for ProtoNet/CNAPs-style learners.
- The ORBIT 2022 challenge winner improves ProtoNet through episode-level embedding adaptation, uniform temporal sampling, augmentation, and invalid-frame detection. It reports a 5.39-point ablation gain over its reimplemented ProtoNet and 71.69% frame accuracy.
- Later parameter-efficient few-shot studies show that simple LayerNorm tuning and attention scaling are strong controls. Consequently, weak linear-head or nearest-neighbour comparisons are insufficient for a superiority claim.

## Closest mechanism line

- Gated DeltaNet combines adaptive forgetting with delta-rule writes; Gated DeltaNet-2 separates erase and write channels. These works support fast-weight memory as a serious motif, but their evidence is primarily language, retrieval, and long-context modelling.
- TTT treats the recurrent hidden state as a model updated by self-supervised learning at test time; Titans similarly studies learned test-time memory. They motivate adaptive state updates, not ORBIT actuator selection.
- Mamba-3 strengthens state tracking through a more expressive state-space recurrence, complex states, and multi-input/multi-output updates. It is a relevant temporal baseline family, not evidence for a biological or local controller.
- Driscoll et al. identify reusable dynamical motifs in multitask recurrent networks and show faster transfer when required motifs were previously learned. This directly motivates motif reuse while leaving the selection and local-credit mechanism open.
- Dendritic E/I task-switching models and predictive-alignment learning show plausible circuit gating and local recurrent shaping, respectively. Neither establishes that a participating E/I carrier improves ORBIT behavior.

## Bounded gap

This snapshot supports a narrow question: can a small causal controller learn when to invoke already useful mechanisms on held-out users? It does not justify claims that the controller invents new motifs, that E/I constraints improve recognition, or that fixed-state memory replaces general attention.

## Primary sources

- https://github.com/microsoft/ORBIT-Dataset
- https://openaccess.thecvf.com/content/ICCV2021/html/Massiceti_ORBIT_A_Real-World_Few-Shot_Dataset_for_Teachable_Object_Recognition_ICCV_2021_paper.html
- https://proceedings.neurips.cc/paper_files/paper/2021/hash/cc1aa436277138f61cda703991069eaf-Abstract.html
- https://arxiv.org/abs/2210.00174
- https://ojs.aaai.org/index.php/AAAI/article/view/28978
- https://arxiv.org/abs/2412.06464
- https://arxiv.org/abs/2605.22791
- https://arxiv.org/abs/2407.04620
- https://arxiv.org/abs/2501.00663
- https://arxiv.org/abs/2603.15569
- https://www.nature.com/articles/s41593-024-01668-6
- https://www.nature.com/articles/s41467-024-50501-y
- https://www.nature.com/articles/s41467-025-61309-9
