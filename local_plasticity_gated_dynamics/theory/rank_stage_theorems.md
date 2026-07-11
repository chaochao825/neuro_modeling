# Rank-stage theorems for low-dimensional credit on a high-rank substrate

This note fixes the objects measured by the P1 audit. Matrix rank, credit
dimension, activity dimension, and dynamical order are not interchangeable.

## 1. A sparse mask can turn a rank-one proposal into a high-rank matrix

Let $M\in\{0,1\}^{n_{post}\times n_{pre}}$,
$u\in\mathbb R^{n_{post}}$, and $v\in\mathbb R^{n_{pre}}$. Entrywise,

$$
  (M\odot uv^\top)_{ij}=M_{ij}u_i v_j
  =(\operatorname{diag}(u)M\operatorname{diag}(v))_{ij}.
$$

Therefore

$$
  M\odot uv^\top=\operatorname{diag}(u)M\operatorname{diag}(v),
$$

and the two matrices necessarily have identical rank. If every entry of
$u$ and $v$ is nonzero, both diagonal matrices are invertible, giving

$$
  \operatorname{rank}(M\odot uv^\top)=\operatorname{rank}(M).
$$

Thus a rank-one unmasked Hebbian proposal can become almost full rank after a
generic sparse connectivity mask. If factors contain zeros, the result is the
rank of the mask restricted to active rows and columns.

The exact theorem and a finite-precision SVD decision are reported separately.
Very small but nonzero factors keep the exact diagonal maps invertible while
making them ill conditioned; the audit therefore records both diagonal
condition numbers and whether equality of rank is numerically resolved at the
configured threshold.

This identity applies directly to the Hebbian term. In the implementation,

$$
  \Delta W_{raw}=\Delta W_{Hebbian}+\Delta W_{decay}.
$$

When weight decay is nonzero, `raw_update` is not generally one outer product.
P1 therefore reports Hebbian, decay, and combined raw ranks separately.

## 2. Dale projection and fan-in normalization are separate stages

Dale projection acts on the candidate effective weights, not on the sign of
the update itself:

$$
  \Delta W_{Dale}=P_{Dale}(W+M\odot\Delta W_{raw})-W.
$$

It is a piecewise-linear map. Away from zero-weight boundaries its derivative
is an active-coordinate mask; when a connected weight lies exactly on the
Dale boundary, a two-sided linear tangent is not defined. Experiments must
record boundary events rather than silently treating the projection as a
globally linear rank-preserving operation.

Fan-in normalization rescales each postsynaptic row to its fixed L1 target.
Within a fixed sign orthant, for one row $w$, target $t$, and
$c=\lVert w\rVert_1>0$,

$$
  N(w)=\frac{t}{c}w,
$$

with directional derivative

$$
  dN_w[\delta w]
  =\frac{t}{c}\left(
    \delta w-w\frac{\operatorname{sign}(w)^\top\delta w}{c}
  \right).
$$

The normalization correction is recorded independently as

$$
  \Delta W_{norm}=N(W+\Delta W_{Dale})-(W+\Delta W_{Dale}),
$$

and the physical update is

$$
  \Delta W_{total}=\Delta W_{Dale}+\Delta W_{norm}.
$$

Even when the credit map is low dimensional, either correction can have high
matrix rank because it acts on a high-rank E/I substrate.

## 3. Fixed-state plastic tangent dimension

Let $B\in\mathbb R^{n_{post}\times d}$ be the feedback basis,
$z\in\mathbb R^d$ the third-factor channels, $g=\phi'(x)$ the fixed-state
derivative, and $e\in\mathbb R^{n_{pre}}$ the eligibility trace. For channel
$a$, the
masked additive weight-space direction is

$$
  D_a=M\odot[(g\odot B_{:a})e^\top].
$$

The instantaneous credit tangent dimension is

$$
  d_{tan}=\operatorname{rank}
  \begin{bmatrix}
    \operatorname{vec}(D_1)&\cdots&\operatorname{vec}(D_d)
  \end{bmatrix}
  \le d.
$$

It is computed from the small Gram matrix

$$
  G_{ab}=\langle D_a,D_b\rangle_F
$$

without materializing an $n_{post}n_{pre}\times d$ Jacobian. The same
definition supports a synaptic scale matrix. For multiplicative plasticity the
local linear scale is the current weight matrix, yielding directions

$$
  D_a=W\odot M\odot[(g\odot B_{:a})e^\top].
$$

Each realized $D_a$ may have high matrix rank while their controllable span
still has dimension at most $d$. This is the primary revised hypothesis.

Because Gram formation squares the condition number, tangent rank uses a
singular-value noise floor of at least
$\sqrt{\epsilon_{mach}d}\,s_{max}$ in addition to the configured absolute and
relative tolerances.

At different states, $g_t$, eligibility $e_t$, and the Dale active set can
change. Consequently the union of tangent spaces or the cumulative update
span across time can exceed the instantaneous feedback dimension. P1 must not
call that a violation of low-dimensional instantaneous credit assignment.

## 4. Three parameterizations

1. **Direct additive:**
   $\Delta W=\eta M\odot uv^\top$, followed by candidate-weight Dale
   projection.
2. **Sign-preserving multiplicative:**
   $W'=W\odot\exp(\eta M\odot uv^\top)$. This is an additive low-rank step in
   log-magnitude coordinates, preserves fixed sparse/Dale signs, and generally
   produces a high-rank physical $\Delta W=W'-W$. If a maximum log step is
   needed, the implementation applies one global scalar to the masked control;
   it does not elementwise clip and rotate the control direction.
3. **Full per-synapse:** every connected synapse receives an independent third
   factor. Its credit-space dimension is the number of connected edges and it
   serves as a high-dimensional upper-bound control, not the local
   low-dimensional main model.

All comparisons must report both control-coordinate and physical-weight
costs/ranks.

## 5. Dynamical metrics

Jacobian outliers are defined relative to a paired bulk Jacobian evaluated at
the same state, gain, derivative, and time constants, with the task-plastic
component removed. P1 counts target eigenvalues to the right of a preregistered
quantile of the paired bulk real-part spectrum. This count is distinct from
the number of eigenvalues with positive real part.

The unadjusted Hankel output is explicitly named **raw numerical rank** of a
future-by-past cross moment; in noisy finite samples it is not a system-order
estimate. Past/future windows are constructed wholly inside each trial. An
optional training-fold parallel-analysis threshold permutes future windows
only within the same trial and compares each mode with a preregistered quantile
of the corresponding null singular value. Centering, scaling, rank tolerances,
and this noise floor
are fit or fixed using training trials only.

## 6. Feedback-angle feasibility

If alignment is defined by principal angles between subspaces, an
$N$-dimensional `full` feedback subspace is the whole ambient space and
cannot have a nonzero principal angle to the task subspace. Nonzero
`full x angle` cells must be retained as invalid, or the protocol must instead
preregister a different operator-angle definition.
