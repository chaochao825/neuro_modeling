# Topic Brief

## Topic

Test whether a low-dimensional controller can causally select reusable input-routing, fast-associative-memory, and temporal-dynamics motifs for few-shot personalized object recognition in real egocentric video.

## Scope

The primary task is the official ORBIT clutter-video evaluation (CLU-VE): labelled clean support videos personalize a recognizer for a held-out user, and the recognizer predicts the user's objects frame by frame from clutter videos using only the current and preceding frames. Train, validation, and test users remain disjoint. A secondary ORBIT-derived stress track varies support shots, observable frame quality, temporal gaps, and occlusion duration without changing the primary benchmark.

The main comparison keeps the visual encoder, sampled frames, support examples, query order, and personalization budget paired across methods. The proposed contribution is the controller and reusable motif interface, not a new visual foundation model. Query labels, future frames, quality annotations, and true latent demand are unavailable to deployable methods.

High-rank Dale-constrained E/I execution is a mechanistic secondary test. It may support the computational interpretation only if it preserves held-out behavior while exhibiting low reduced-dynamics closure error and decaying normal perturbations. Low physical matrix rank, biological realism of the frozen image encoder, and replacement of general-purpose attention/SSM models are explicitly out of scope.

## Audience

Researchers in few-shot and continual learning, efficient sequence models, computational neuroscience, and adaptive on-device vision.

## Constraints

- Venue: empirical ML/computational-neuroscience paper; venue selected only after the primary ORBIT evidence is available.
- Page target: 8--10 main pages plus reproducibility and mechanism supplement.
- Deadline: none registered.
- Special requirements: Python 3.11; fixed seeds; user is the statistical unit; train-only preprocessing and controller fitting; official ORBIT user splits; causal query inference; every failed condition retained; support/oppose/inconclusive conclusions; BPTT allowed only for explicit offline baselines or shared-motif pretraining, never for the local controller.

## Key Terms

ORBIT, personalized few-shot recognition, causal video inference, actuator matching, fast weights, gated delta memory, dynamical motifs, local credit assignment, E/I carrier, out-of-distribution personalization.
