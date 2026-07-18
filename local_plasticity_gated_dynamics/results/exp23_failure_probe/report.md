# Exp23 failure probe

This is a fixed reanalysis of the immutable 30-seed formal-v2 archive;
no seed, scale or condition was rerun or selected.

- Current local matched gain: -0.00126; natural gain: -0.00101.
- Delayed local matched gain: -0.01133; natural gain: +0.00056.
- Delayed BPTT natural gain: +0.01796, 95% seed-bootstrap CI [+0.01162, +0.02444].
- Delayed exact-forward natural gain: +0.00194, 95% seed-bootstrap CI [+0.00056, +0.00338].
- Delayed local median matched/natural axis ratio: 83.4x.
- Current local task-loss change: -0.01575 while balanced accuracy did not improve.

Formal eligibility reverified: True; registered oppose component reverified: True.
The original `oppose` result remains valid only when both audits above
are true, and only for the registered drive-gain
axis, local rule and state-displacement protocol.  It should not be
generalized to all e-prop or local plasticity.  In this post-hoc
diagnostic, the tested natural BPTT mean CI upper bound remained below
the local rule's registered 0.03 MCID, suggesting limited headroom for
this axis, optimizer and protocol rather than proving an impossibility.
Post-training state-displacement matching strongly amplified
the weakest delayed local direction.  A future probe must use an ex-ante
direction-by-scale curve and separately match cumulative plasticity L1/L2.
