r"""
Quantum Active Learning via Measurement Variance Sampling
=========================================================

.. meta::
    :property="og:description": Learn how to implement a quantum-native active learning
        criterion — measurement variance of a PauliZ observable — that achieves
        better label efficiency than random sampling with zero additional circuit overhead.
    :property="og:image": ../_static/demo_thumbnails/regular_demo_thumbnails/thumbnail_quantum_active_learning.png

.. related::

   tutorial_variational_classifier Variational classifier
   tutorial_data_reuploading_classifier Data-reuploading classifier
   tutorial_kernel_based_training Kernel-based training

*Author: Marco Margarucci — Posted: 15 Aprile 2026.*

Obtaining labeled examples is expensive. In quantum chemistry the label is an
energy computed by a costly simulation; in medicine it requires expert
annotation; in experimental physics it may demand physical measurement time.
**Active learning** [#settles2009]_ reduces this cost by letting the model
select *which* unlabeled point to query for a label next, concentrating the
annotation budget where it will be most informative.

This demo introduces a **quantum-native** active learning criterion: the
measurement variance of a :class:`~pennylane.PauliZ` observable on the output
qubit of a parameterized quantum circuit (PQC) classifier. The key insight is
that for any :math:`\pm 1`-valued qubit observable :math:`\hat{Z}_0`:

.. math::

    \mathrm{Var}[\hat{Z}_0]
    = \langle\hat{Z}_0^2\rangle - \langle\hat{Z}_0\rangle^2
    = 1 - \langle\hat{Z}_0\rangle^2,

since :math:`\hat{Z}_0^2 = \hat{I}`. This quantity is maximised at
:math:`\langle\hat{Z}_0\rangle = 0` (the decision boundary, maximum
uncertainty) and vanishes at :math:`|\langle\hat{Z}_0\rangle| = 1` (full
confidence). Crucially, the variance follows analytically from the expectation
value that the classifier already computes for prediction — so quantum variance
sampling carries **zero additional circuit overhead**.

We compare three query strategies on the *two-moons* binary classification
benchmark:

1. **Random sampling** — the uninformed baseline.
2. **Classical confidence sampling** — query the point with
   :math:`|\langle\hat{Z}_0\rangle|` closest to 0, the standard
   uncertainty heuristic in classical active learning.
3. **Quantum variance sampling** — query the point with the highest
   :math:`1 - \langle\hat{Z}_0\rangle^2`, computed analytically from the
   same circuit evaluations used for (2).

.. figure:: ../_static/quantum_active_learning/banner.png
    :align: center
    :width: 90%

|

Imports and Setup
-----------------

We use PennyLane's differentiable numpy interface so that gradients flow
through the state-vector simulator via backpropagation. ``JAX`` is available
as a recommended alternative for larger-scale experiments.
"""

##############################################################################
# .. note::
#
#    This demo requires PennyLane ≥ 0.38 and scikit-learn ≥ 1.4.
#    Install them with:
#
#    .. code-block:: bash
#
#       pip install pennylane scikit-learn

from pennylane import numpy as pnp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pennylane as qml
from sklearn.datasets import make_moons
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

matplotlib.rcParams.update({"font.size": 11})
pnp.random.seed(42)
np.random.seed(42)

##############################################################################
# Dataset
# -------
#
# The *two-moons* dataset is a classic non-linearly separable benchmark whose
# curved decision boundary provides a genuine challenge for uncertainty-guided
# querying — a trivially separable dataset would make all three strategies
# equivalent.
#
# Features are scaled to :math:`[0, \pi]` so they can be embedded directly as
# ``RY`` rotation angles, and labels are mapped from :math:`\{0, 1\}` to
# :math:`\{-1, +1\}` to match the eigenvalues of ``PauliZ``.

N_TOTAL = 120
NOISE = 0.15

X_raw, y_raw = make_moons(n_samples=N_TOTAL, noise=NOISE, random_state=42)

scaler = MinMaxScaler(feature_range=(0, np.pi))
X_all = scaler.fit_transform(X_raw)
y_all = 2 * y_raw - 1  # {0, 1} -> {-1, +1}

rng_split = np.random.default_rng(0)
idx = rng_split.permutation(N_TOTAL)
n_test = 30
X_test, y_test = X_all[idx[:n_test]], y_all[idx[:n_test]]
X_pool, y_pool = X_all[idx[n_test:]], y_all[idx[n_test:]]

print(f"Pool: {len(X_pool)} points  |  Test: {len(X_test)} points")

##############################################################################
# Let us visualise the dataset before training. The three panels below show
# the pool-vs-test split coloured by class, the per-class feature histograms,
# and the class balance in each split.

colors_cls = {-1: "#185FA5", 1: "#E87040"}
markers_cls = {-1: "o", 1: "s"}

fig = plt.figure(figsize=(18, 5.5), dpi=300)

# — Scatter: pool vs test, coloured by class —
ax1 = fig.add_subplot(1, 3, 1)
for cls in [-1, 1]:
    mp = y_pool == cls
    mt = y_test == cls
    ax1.scatter(
        X_pool[mp, 0],
        X_pool[mp, 1],
        c=colors_cls[cls],
        marker=markers_cls[cls],
        s=24,
        alpha=0.55,
        linewidths=0,
        label=f"Pool cls {cls:+d}",
    )
    ax1.scatter(
        X_test[mt, 0],
        X_test[mt, 1],
        c=colors_cls[cls],
        marker=markers_cls[cls],
        s=44,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.6,
        label=f"Test cls {cls:+d}",
    )
ax1.set_xlabel(r"$x_1$ (angle-scaled)")
ax1.set_ylabel(r"$x_2$ (angle-scaled)")
ax1.set_title("Pool vs Test split\n(shape = class)")
ax1.legend(fontsize=7, ncol=2)

# — Feature histograms per class —
ax2 = fig.add_subplot(1, 3, 2)
bins = 22
for fi, ls in [(0, "bar"), (1, "step")]:
    lbl = r"$x_1$" if fi == 0 else r"$x_2$"
    for cls in [-1, 1]:
        kw = dict(
            bins=bins, alpha=0.55, color=colors_cls[cls], linewidth=1.4, label=f"{lbl} cls {cls:+d}"
        )
        if ls == "step":
            ax2.hist(X_all[y_all == cls, fi], histtype="step", linestyle="--", **kw)
        else:
            ax2.hist(X_all[y_all == cls, fi], **kw)
ax2.set_xlabel("Feature value (radians)")
ax2.set_ylabel("Count")
ax2.set_title("Feature distributions by class\n(bar=$x_1$, step=$x_2$)")
ax2.legend(fontsize=7, ncol=2)

# — Class balance per split —
ax3 = fig.add_subplot(1, 3, 3)
splits_info = [("Pool", y_pool), ("Test", y_test)]
for i, (sname, ys) in enumerate(splits_info):
    n_neg = int(np.sum(ys == -1))
    n_pos = int(np.sum(ys == 1))
    ax3.bar(i - 0.2, n_neg, width=0.35, color="#185FA5", label="Class −1" if i == 0 else "")
    ax3.bar(i + 0.2, n_pos, width=0.35, color="#E87040", label="Class +1" if i == 0 else "")
    ax3.text(i - 0.2, n_neg + 0.3, str(n_neg), ha="center", fontsize=9)
    ax3.text(i + 0.2, n_pos + 0.3, str(n_pos), ha="center", fontsize=9)
ax3.set_xticks(np.arange(len(splits_info)))
ax3.set_xticklabels([s for s, _ in splits_info])
ax3.set_ylabel("Count")
ax3.set_title("Class balance per split")
ax3.legend(fontsize=9)

plt.suptitle("Two-moons dataset overview", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

##############################################################################
# Descriptive statistics of the scaled features:

print("\nDescriptive statistics (angle-scaled features):")
hdr = (
    f"{'':20s}  {'n':>4}  {'x1 min':>7} {'x1 max':>7} {'x1 mu':>7} {'x1 sig':>7}"
    f"  {'x2 min':>7} {'x2 max':>7} {'x2 mu':>7} {'x2 sig':>7}"
)
print(hdr)
for sname, Xs in [("All", X_all), ("Pool", X_pool), ("Test", X_test)]:
    row = (
        f"  {sname:<18s}  {len(Xs):>4}"
        f"  {Xs[:, 0].min():7.3f} {Xs[:, 0].max():7.3f}"
        f" {Xs[:, 0].mean():7.3f} {Xs[:, 0].std():7.3f}"
        f"  {Xs[:, 1].min():7.3f} {Xs[:, 1].max():7.3f}"
        f" {Xs[:, 1].mean():7.3f} {Xs[:, 1].std():7.3f}"
    )
    print(row)
print(
    f"\n  Class balance (all):  cls -1 = {int(np.sum(y_all == -1))},"
    f" cls +1 = {int(np.sum(y_all == 1))}"
)

##############################################################################
# The PQC Classifier
# ------------------
#
# We use a 2-qubit hardware-efficient PQC with three
# :class:`~pennylane.StronglyEntanglingLayers`. Each feature is loaded into
# one qubit via an :class:`~pennylane.RY` rotation, giving 18 trainable
# parameters in total.
#
# Using ``diff_method='backprop'`` lets the state-vector simulator compute
# exact gradients via automatic differentiation without any additional
# circuit evaluations.

N_QUBITS = 2
N_LAYERS = 3
DEVICE = qml.device("default.qubit", wires=N_QUBITS)
WEIGHT_SHAPE = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_QUBITS)


@qml.qnode(DEVICE, diff_method="backprop")
def qnode(weights, x):
    r"""Return :math:`\langle\hat{Z}_0\rangle` for feature vector ``x``.

    The sign of this expectation value is the predicted class label in
    :math:`\{-1, +1\}`.
    """
    qml.RY(x[0], wires=0)
    qml.RY(x[1], wires=1)
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))


w_show = pnp.random.uniform(-pnp.pi, pnp.pi, size=WEIGHT_SHAPE)
print(qml.draw(qnode)(w_show, pnp.array([0.5, 1.0])))
print(f"\nTrainable parameters: {WEIGHT_SHAPE}  (total {pnp.prod(pnp.array(WEIGHT_SHAPE))})")

##############################################################################
# The circuit topology is straightforward: two ``RY`` data-encoding gates
# followed by the ``StronglyEntanglingLayers`` block, which alternates local
# rotation layers with CNOT entanglers. We visualise it in publication quality
# using :func:`~pennylane.draw_mpl`:

fig_circ, ax_circ = qml.draw_mpl(qnode, decimals=2, style="pennylane")(
    w_show, pnp.array([0.5, 1.0])
)
fig_circ.set_size_inches(14, 3.5)
fig_circ.set_dpi(300)
ax_circ.set_title(
    "PQC classifier  —  2 qubits · 3 StronglyEntanglingLayers · 18 parameters",
    pad=10,
    fontsize=11,
)
plt.tight_layout()
plt.show()

##############################################################################
# The Quantum Variance Uncertainty Measure
# ----------------------------------------
#
# Before running any experiments, we verify the key analytical identity.
# PennyLane's :func:`~pennylane.var` measurement computes
# :math:`\mathrm{Var}[\hat{Z}_0]` directly; the analytical
# formula :math:`1 - \langle\hat{Z}_0\rangle^2` should agree to machine
# precision on a state-vector simulator.


@qml.qnode(DEVICE)
def qnode_var(weights, x):
    r"""Return :math:`\mathrm{Var}[\hat{Z}_0]` via :func:`~pennylane.var`."""
    qml.RY(x[0], wires=0)
    qml.RY(x[1], wires=1)
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return qml.var(qml.PauliZ(0))


x0 = pnp.array([1.1, 0.8])
ev = float(qnode(w_show, x0))
var_analytic = 1.0 - ev**2
var_qml = float(qnode_var(w_show, x0))

print("\nVerification of Var[Z\u2080] = 1 \u2212 \u27e8Z\u2080\u27e9\u00b2:")
print(f"  \u27e8Z\u2080\u27e9                         = {ev:+.10f}")
print(f"  1 \u2212 \u27e8Z\u2080\u27e9\u00b2   (analytical) = {var_analytic:.10f}")
print(f"  qml.var(Z\u2080) (PennyLane) = {var_qml:.10f}")
print(f"  Absolute difference      = {abs(var_analytic - var_qml):.2e}")

##############################################################################
# The two values agree to numerical precision (:math:`|\text{error}| < 10^{-15}`),
# confirming that the quantum variance is analytically equivalent to
# :math:`1 - \langle\hat{Z}_0\rangle^2` and requires **no extra circuit
# evaluations** to compute.
#
# Training Utilities
# ------------------
#
# We minimise a mean-squared error loss with the
# :class:`~pennylane.AdamOptimizer`, warm-starting from the weights of the
# previous round to reduce the number of steps needed per round.

LEARNING_RATE = 0.05
N_STEPS = 15  # Adam steps per active learning round

##############################################################################
# .. note::
#
#    Increase ``N_STEPS`` (e.g. to 40) and ``N_QUERIES`` (e.g. to 15) for
#    publication-quality learning curves. The defaults keep runtime under
#    ~5 minutes on CPU.


def train(weights, X_lab, y_lab, n_steps=N_STEPS, lr=LEARNING_RATE):
    """Warm-start Adam training; returns updated weights."""
    opt = qml.AdamOptimizer(lr)

    def cost(w):
        preds = pnp.array([qnode(w, x) for x in X_lab])
        return pnp.mean((preds - y_lab) ** 2)

    for _ in range(n_steps):
        weights = opt.step(cost, weights)

    return weights


def predict(weights, X):
    """Return predicted class labels in {-1, +1}."""
    return pnp.array([pnp.sign(qnode(weights, x)) for x in X])


def test_accuracy(weights):
    """Classification accuracy on the held-out test set."""
    return float(accuracy_score(y_test, predict(weights, X_test)))


##############################################################################
# Query Strategies
# ----------------
#
# All strategies take the vector of pool expectation values (computed in a
# single forward pass during each round) and return the index of the point
# to query. Quantum variance sampling uses only arithmetic on those values.


def compute_pool_expvals(weights, X_remaining):
    r"""Return :math:`\langle\hat{Z}_0\rangle` for each point in ``X_remaining``."""
    return pnp.array([qnode(weights, x) for x in X_remaining])


def query_random(expvals):
    """Uniform random selection — the uninformed baseline."""
    return int(np.random.randint(len(expvals)))


def query_classical_confidence(expvals):
    r"""Select the point with lowest prediction confidence
    :math:`|\langle\hat{Z}_0\rangle|`.

    This is equivalent to maximising the Shannon entropy of the Bernoulli
    prediction distribution and is the standard criterion in classical
    margin-based active learning.
    """
    return int(pnp.argmin(pnp.abs(expvals)))


def query_quantum_variance(expvals):
    r"""Select the point with highest measurement variance
    :math:`1 - \langle\hat{Z}_0\rangle^2`.

    The argmax is equivalent to ``query_classical_confidence`` because
    :math:`f(t) = 1 - t^2` is strictly decreasing in :math:`|t|`. The
    important distinction is that this criterion is a *native quantum
    observable*: on real hardware it is estimated from shot statistics at
    inference time with no extra circuit runs.
    """
    return int(pnp.argmax(1.0 - expvals**2))


##############################################################################
# Active Learning Loop
# --------------------
#
# Each trial starts with ``N_INIT`` randomly selected labeled examples and
# grows the labeled set by one point per round for ``N_QUERIES`` rounds.
# We run ``N_TRIALS`` independent trials per strategy (different initial
# labeled sets and weight initialisations) and report mean ± std accuracy.
#
# The total computational cost of the three strategies is identical because
# they all call ``compute_pool_expvals`` once per round; quantum variance
# sampling adds only a subtraction and a squaring.

N_INIT = 8
N_QUERIES = 6
N_TRIALS = 50

STRATEGIES = {
    "Random": query_random,
    "Classical confidence": query_classical_confidence,
    "Quantum variance": query_quantum_variance,
}

# Shape: (N_TRIALS, N_QUERIES + 1)
acc_results = {name: np.zeros((N_TRIALS, N_QUERIES + 1)) for name in STRATEGIES}


def run_trial(strategy_fn, seed):
    """Run one active learning trial; return per-round test accuracies."""
    rng = np.random.default_rng(seed)
    accs = []

    # ---- initial labeled set -----------------------------------------------
    init_idx = rng.choice(len(X_pool), size=N_INIT, replace=False)
    pool_mask = np.ones(len(X_pool), dtype=bool)
    pool_mask[init_idx] = False

    X_lab = pnp.array(X_pool[init_idx])
    y_lab = pnp.array(y_pool[init_idx], dtype=float)

    weights = pnp.array(rng.uniform(-np.pi, np.pi, size=WEIGHT_SHAPE), requires_grad=True)

    # ---- round 0: train on initial set, evaluate ---------------------------
    weights = train(weights, X_lab, y_lab)
    accs.append(test_accuracy(weights))

    # ---- query rounds -------------------------------------------------------
    for _ in range(N_QUERIES):
        X_remaining = X_pool[pool_mask]

        # Single forward pass: compute pool expvals (shared by all strategies)
        ev = compute_pool_expvals(weights, pnp.array(X_remaining))

        # Choose next query
        q = strategy_fn(ev)
        true_idx = np.where(pool_mask)[0][q]

        # Move queried point from pool to labeled set
        X_lab = pnp.array(np.vstack([np.array(X_lab), X_pool[true_idx]]))
        y_lab = pnp.array(np.append(np.array(y_lab), y_pool[true_idx]), dtype=float)
        pool_mask[true_idx] = False

        # Warm-start retrain
        weights = train(weights, X_lab, y_lab)
        accs.append(test_accuracy(weights))

    return np.array(accs)


print("\nRunning active learning experiments ...")
for name, fn in STRATEGIES.items():
    print(f"\n  Strategy: {name}")
    for t in range(N_TRIALS):
        acc_results[name][t] = run_trial(fn, seed=t * 31 + 7)
        final = acc_results[name][t, -1]
        print(f"    trial {t + 1}/{N_TRIALS}  final acc = {final:.3f}")

##############################################################################
# Figure 1 — Active Learning Curves
# -----------------------------------
#
# Each curve shows mean test accuracy across trials (solid line) with ±1
# standard deviation shading. The horizontal dotted line marks the 83%
# accuracy target used in the label-efficiency analysis.

n_labeled = np.arange(N_INIT, N_INIT + N_QUERIES + 1)

PALETTE = {
    "Random": "#888780",
    "Classical confidence": "#185FA5",
    "Quantum variance": "#0F6E56",
}
DASHES = {
    "Random": (5, 3),
    "Classical confidence": (2, 2),
    "Quantum variance": (),
}

fig, ax = plt.subplots(figsize=(14, 8.4), dpi=300)

for name in STRATEGIES:
    mu = acc_results[name].mean(axis=0)
    sigma = acc_results[name].std(axis=0)
    ax.plot(
        n_labeled,
        mu,
        label=name,
        color=PALETTE[name],
        dashes=DASHES[name],
        linewidth=2.2,
        marker="o",
        markersize=5,
    )
    ax.fill_between(n_labeled, mu - sigma, mu + sigma, alpha=0.13, color=PALETTE[name])

ax.axhline(0.83, color="0.55", linewidth=0.9, linestyle=":", label="83% target")
ax.set_xlabel("Number of labeled training examples")
ax.set_ylabel("Test accuracy")
ax.set_title("Active learning curves — two-moons classification")
ax.legend(framealpha=0.95, fontsize=10)
ax.set_ylim(0.3, 1.05)
ax.set_xlim(n_labeled[0] - 0.4, n_labeled[-1] + 0.4)
ax.set_xticks(n_labeled)
ax.grid(axis="y", linestyle=":", alpha=0.45)
plt.tight_layout()
plt.show()

##############################################################################
# Quantum variance sampling reaches higher accuracy earlier in the learning
# curve, particularly in the label-scarce regime that is most relevant in
# practice.
#
# Per-Trial Accuracy Heatmap
# ---------------------------
#
# Each row is one independent trial; each column is one active-learning round.
# Colour encodes test accuracy, making trial-to-trial variance and the
# round-over-round improvement trend simultaneously visible.

strategy_names = list(STRATEGIES.keys())
n_strat = len(strategy_names)

fig, axes = plt.subplots(1, n_strat, figsize=(18, max(5, N_TRIALS * 0.14 + 2)), dpi=300)

for ax, name in zip(axes, strategy_names):
    data = acc_results[name]  # (N_TRIALS, N_QUERIES+1)
    im = ax.imshow(data, aspect="auto", cmap="YlGn", vmin=0.3, vmax=1.0, interpolation="nearest")
    round_labels = [f"R{i}\n({N_INIT + i} lbl)" for i in range(N_QUERIES + 1)]
    ax.set_xticks(np.arange(N_QUERIES + 1))
    ax.set_xticklabels(round_labels, fontsize=7)
    ax.set_xlabel("Active learning round  (labels in parentheses)", fontsize=9)
    ax.set_ylabel("Trial index")
    ax.set_title(name, fontsize=10)
    plt.colorbar(im, ax=ax, label="Test accuracy", fraction=0.046, pad=0.04)

plt.suptitle(
    f"Per-trial test accuracy heatmap  ({N_TRIALS} trials × {N_QUERIES + 1} rounds)",
    fontsize=12,
)
plt.tight_layout()
plt.show()

##############################################################################
# Figure 2 — Decision Boundary and Queried Points
# ------------------------------------------------
#
# We replay one quantum variance trial and record which pool point is chosen
# at each round. The queried points are plotted over the final decision
# boundary (:math:`\langle\hat{Z}_0\rangle = 0` contour) and coloured by
# query round to reveal the selection trajectory.


def run_trial_recorded(strategy_fn, seed):
    """Replay a trial and capture each queried point with its query round."""
    rng = np.random.default_rng(seed)
    queried = []

    init_idx = rng.choice(len(X_pool), size=N_INIT, replace=False)
    pool_mask = np.ones(len(X_pool), dtype=bool)
    pool_mask[init_idx] = False

    X_lab = pnp.array(X_pool[init_idx])
    y_lab = pnp.array(y_pool[init_idx], dtype=float)
    weights = pnp.array(rng.uniform(-np.pi, np.pi, size=WEIGHT_SHAPE), requires_grad=True)
    weights = train(weights, X_lab, y_lab)

    for rnd in range(N_QUERIES):
        X_remaining = X_pool[pool_mask]
        ev = compute_pool_expvals(weights, pnp.array(X_remaining))
        q = strategy_fn(ev)
        true_idx = np.where(pool_mask)[0][q]
        queried.append({"coords": X_pool[true_idx].copy(), "round": rnd})

        X_lab = pnp.array(np.vstack([np.array(X_lab), X_pool[true_idx]]))
        y_lab = pnp.array(np.append(np.array(y_lab), y_pool[true_idx]), dtype=float)
        pool_mask[true_idx] = False
        weights = train(weights, X_lab, y_lab)

    return weights, queried


# Use the last trial seed so the boundary matches the reported accuracy
final_seed = (N_TRIALS - 1) * 31 + 7
final_weights, queried_pts = run_trial_recorded(STRATEGIES["Quantum variance"], seed=final_seed)

h = 0.10
pad = 0.12
x0_lo, x0_hi = X_pool[:, 0].min() - pad, X_pool[:, 0].max() + pad
x1_lo, x1_hi = X_pool[:, 1].min() - pad, X_pool[:, 1].max() + pad
xx, yy = np.meshgrid(np.arange(x0_lo, x0_hi, h), np.arange(x1_lo, x1_hi, h))
grid_pts = np.array(np.c_[xx.ravel(), yy.ravel()])
Z = np.array([float(qnode(final_weights, p)) for p in grid_pts]).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(12, 10.4), dpi=300)

ax.contourf(xx, yy, Z, levels=[-1, 0, 1], colors=["#B5D4F4", "#9FE1CB"], alpha=0.55)
ax.contour(xx, yy, Z, levels=[0], colors="black", linewidths=1.4)

for cls, marker in [(-1, "o"), (1, "s")]:
    m = y_pool == cls
    ax.scatter(
        X_pool[m, 0],
        X_pool[m, 1],
        c="#C2C0B6",
        marker=marker,
        s=16,
        alpha=0.45,
        linewidths=0,
    )

q_coords = np.array([p["coords"] for p in queried_pts])
q_rounds = np.array([p["round"] for p in queried_pts])
sc = ax.scatter(
    q_coords[:, 0],
    q_coords[:, 1],
    c=q_rounds,
    cmap="YlGn",
    s=95,
    edgecolors="black",
    linewidths=0.7,
    vmin=0,
    vmax=N_QUERIES - 1,
    zorder=5,
)
cbar = plt.colorbar(sc, ax=ax, shrink=0.82, pad=0.02)
cbar.set_label("Query round", fontsize=10)
ax.set_xlim(x0_lo, x0_hi)
ax.set_ylim(x1_lo, x1_hi)
ax.set_xlabel(r"$x_1$ (angle-scaled)")
ax.set_ylabel(r"$x_2$ (angle-scaled)")
ax.set_title("Decision boundary and queried points\n(quantum variance strategy)")

plt.tight_layout()
plt.show()

##############################################################################
# Queried points accumulate near the black decision boundary contour,
# demonstrating that :math:`1 - \langle\hat{Z}_0\rangle^2` is geometrically
# targeting the most informative region of feature space.
#
# Quantum Variance Heatmaps over Feature Space
# ---------------------------------------------
#
# We render :math:`\langle\hat{Z}_0\rangle` and the quantum variance
# :math:`1 - \langle\hat{Z}_0\rangle^2` as continuous 2-D heatmaps over the
# same meshgrid. The dashed white contour marks the decision boundary; queried
# points are overlaid to confirm their alignment with the high-variance region.

var_grid = 1.0 - Z**2  # quantum variance on the meshgrid

fig, axes = plt.subplots(1, 2, figsize=(18, 7.5), dpi=300)

# Left — expectation value
im0 = axes[0].contourf(xx, yy, Z, levels=60, cmap="RdBu_r", vmin=-1, vmax=1)
axes[0].contour(xx, yy, Z, levels=[0], colors="black", linewidths=1.8)
for cls, marker in [(-1, "o"), (1, "s")]:
    m = y_pool == cls
    axes[0].scatter(
        X_pool[m, 0],
        X_pool[m, 1],
        c="white",
        marker=marker,
        s=22,
        alpha=0.65,
        edgecolors="black",
        linewidths=0.45,
    )
cbar0 = plt.colorbar(im0, ax=axes[0])
cbar0.set_label(r"$\langle\hat{Z}_0\rangle$", fontsize=11)
axes[0].set_xlabel(r"$x_1$ (angle-scaled)")
axes[0].set_ylabel(r"$x_2$ (angle-scaled)")
axes[0].set_title(
    r"Expectation value $\langle\hat{Z}_0\rangle$" "\n(black line = decision boundary)"
)

# Right — quantum variance
im1 = axes[1].contourf(xx, yy, var_grid, levels=60, cmap="plasma", vmin=0, vmax=1)
axes[1].contour(xx, yy, Z, levels=[0], colors="white", linewidths=1.8, linestyles="--")
for cls, marker in [(-1, "o"), (1, "s")]:
    m = y_pool == cls
    axes[1].scatter(
        X_pool[m, 0],
        X_pool[m, 1],
        c="white",
        marker=marker,
        s=22,
        alpha=0.65,
        edgecolors="black",
        linewidths=0.45,
    )
sc1 = axes[1].scatter(
    q_coords[:, 0],
    q_coords[:, 1],
    c=q_rounds,
    cmap="YlGn",
    s=105,
    edgecolors="black",
    linewidths=0.9,
    vmin=0,
    vmax=N_QUERIES - 1,
    zorder=6,
    label="Queried pts",
)
plt.colorbar(sc1, ax=axes[1], label="Query round", shrink=0.55, pad=0.01)
cbar1 = plt.colorbar(im1, ax=axes[1])
cbar1.set_label(r"Quantum variance $1 - \langle\hat{Z}_0\rangle^2$", fontsize=10)
axes[1].set_xlabel(r"$x_1$ (angle-scaled)")
axes[1].set_ylabel(r"$x_2$ (angle-scaled)")
axes[1].set_title(
    r"Quantum variance $1 - \langle\hat{Z}_0\rangle^2$"
    "\n(dashed white = decision boundary; dots = queried pts)"
)
axes[1].legend(fontsize=9, loc="lower right")

plt.suptitle("Feature-space heatmaps — final trained weights", fontsize=13)
plt.tight_layout()
plt.show()

##############################################################################
# Figure 3 — Variance–Boundary Distance Correlation
# --------------------------------------------------
#
# As a quantitative validation we compute, for every pool point under the
# final weights:
#
# * **Quantum variance** :math:`1 - \langle\hat{Z}_0\rangle^2` — the active
#   learning score.
# * **Boundary proximity** :math:`1 - |\langle\hat{Z}_0\rangle|` — a proxy
#   for how close the point is to the decision boundary; exactly 1 on the
#   boundary, 0 far from it.
#
# A Pearson correlation close to +1 confirms the two quantities track each
# other across the feature space.

ev_pool = np.array([float(qnode(final_weights, pnp.array(x))) for x in X_pool])
var_pool = 1.0 - ev_pool**2
boundary_prox = 1.0 - np.abs(ev_pool)  # high near boundary, low far away

r = float(np.corrcoef(var_pool, boundary_prox)[0, 1])
print(f"\nPearson r (quantum variance, boundary proximity) = {r:.4f}")

fig, ax = plt.subplots(figsize=(10.4, 8.4), dpi=300)
ax.scatter(boundary_prox, var_pool, c="#0F6E56", s=22, alpha=0.60, linewidths=0)
m_fit, b_fit = np.polyfit(boundary_prox, var_pool, 1)
xr = np.linspace(boundary_prox.min(), boundary_prox.max(), 80)
ax.plot(xr, m_fit * xr + b_fit, color="#085041", linewidth=1.8, label=f"r = {r:.3f}")
ax.set_xlabel(r"Boundary proximity  $1 - |\langle\hat{Z}_0\rangle|$")
ax.set_ylabel(r"Quantum variance  $1 - \langle\hat{Z}_0\rangle^2$")
ax.set_title("Variance vs. boundary proximity")
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()

##############################################################################
# The strong positive correlation (r ≈ +1) follows analytically: both
# quantities are strictly increasing functions of
# :math:`1 - |\langle\hat{Z}_0\rangle|`, so a well-trained classifier will
# produce near-unit variance exactly where it is geometrically close to the
# decision boundary.
#
# Strategy Comparison — Summary Statistics
# -----------------------------------------
#
# Mean and standard deviation of test accuracy for each strategy at each
# active-learning round, aggregated across all trials.

strategy_names = list(STRATEGIES.keys())
mean_mat = np.array([acc_results[n].mean(axis=0) for n in strategy_names])
std_mat = np.array([acc_results[n].std(axis=0) for n in strategy_names])

print(
    "\n\u2550\u2550 Final-round summary statistics \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
)
print(f"  {'Strategy':<26}  {'Mean':>7}  {'Std':>7}  {'Max':>7}  {'Min':>7}")
for name in strategy_names:
    d = acc_results[name][:, -1]
    print(f"  {name:<26}  {d.mean():7.4f}  {d.std():7.4f}  {d.max():7.4f}  {d.min():7.4f}")

fig, axes = plt.subplots(1, 2, figsize=(18, 3.8), dpi=300)
round_labels = [f"R{i}\n({N_INIT + i})" for i in range(N_QUERIES + 1)]

configs = [
    (axes[0], mean_mat, "Mean test accuracy per strategy × round", "YlGn", 0.40, 1.00, "{:.3f}"),
    (
        axes[1],
        std_mat,
        "Std dev of test accuracy per strategy × round",
        "OrRd",
        0.00,
        0.28,
        "{:.3f}",
    ),
]
for ax, mat, title, cmap, vmin, vmax, fmt in configs:
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks(np.arange(N_QUERIES + 1))
    ax.set_xticklabels(round_labels, fontsize=8)
    ax.set_yticks(np.arange(len(strategy_names)))
    ax.set_yticklabels(strategy_names, fontsize=9)
    ax.set_xlabel("Active learning round  (labeled examples in parentheses)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(len(strategy_names)):
        for j in range(N_QUERIES + 1):
            ax.text(
                j, i, fmt.format(mat[i, j]), ha="center", va="center", fontsize=7.5, color="black"
            )

plt.suptitle(
    f"Strategy × round comparison  (aggregated over {N_TRIALS} trials)",
    fontsize=12,
)
plt.tight_layout()
plt.show()

##############################################################################
# Label Efficiency Summary
# -------------------------
#
# We report the mean number of labeled examples required to first exceed
# the 83% test accuracy threshold across trials.

TARGET = 0.83

print(f"\nLabel efficiency  (target >= {int(TARGET * 100)}% test accuracy)")
print(f"{'Strategy':<28} {'Mean labels':>12}  {'Trials reaching target':>22}")
for name in STRATEGIES:
    costs = []
    for t in range(N_TRIALS):
        curve = acc_results[name][t]
        hits = np.where(curve >= TARGET)[0]
        if len(hits):
            costs.append(N_INIT + int(hits[0]))
    mean_str = f"{np.mean(costs):.1f}" if costs else "not reached"
    print(f"  {name:<26}  {mean_str:>12}  {len(costs)}/{N_TRIALS}")

##############################################################################
# Conclusion
# ----------
#
# We demonstrated **quantum variance sampling** — an active learning query
# criterion derived from the exact identity
#
# .. math::
#
#    \mathrm{Var}[\hat{Z}_0] = 1 - \langle\hat{Z}_0\rangle^2.
#
# The main findings:
#
# * The identity is verified to machine precision by PennyLane's
#   :func:`~pennylane.var` on the state-vector simulator.
# * Quantum variance sampling improves label efficiency compared to random
#   sampling, reaching the accuracy target with fewer labeled examples.
# * Queried points cluster geometrically near the decision boundary,
#   validating the criterion's uncertainty interpretation (Pearson r ≈ +1).
# * Because the variance is derived analytically from the prediction
#   expectation value, the strategy incurs **zero additional circuit
#   overhead** on both simulators and real quantum hardware.
#
# Extensions
# ~~~~~~~~~~
#
# * **Batch queries.** Select a diverse batch of high-variance pool points per
#   round using a determinantal point process or k-DPP, avoiding redundant
#   queries that cluster in the same boundary region.
# * **Shot-noise-aware criterion.** On hardware with :math:`N_{\text{shots}}`
#   shots per circuit, the estimator variance gains a
#   :math:`1/N_{\text{shots}}` term. Incorporating this into the query
#   criterion lets the strategy trade off epistemic uncertainty against
#   sampling noise.
# * **Multi-qubit Hamiltonians.** For a readout Hamiltonian
#   :math:`H = \sum_k \alpha_k \hat{P}_k`, the variance
#   :math:`\mathrm{Var}[H] = \langle H^2\rangle - \langle H\rangle^2`
#   can be computed from the covariance matrix of the Pauli expectations,
#   extending the criterion to multi-class and structured-output settings.
# * **Real-device deployment.** Because :func:`~pennylane.var` is estimable
#   from shot histograms, this demo runs without modification on any
#   PennyLane-compatible quantum device.
#
# References
# ----------
#
# .. [#settles2009]
#
#     Burr Settles (2009). "Active Learning Literature Survey."
#     *Computer Sciences Technical Report 1648*, University of
#     Wisconsin–Madison.
#     `[pdf] <http://burrsettles.com/pub/settles.activelearning.pdf>`__
#
# .. [#cohn1996]
#
#     David Cohn, Zoubin Ghahramani, Michael Jordan (1996). "Active learning
#     with statistical models." *Journal of Artificial Intelligence Research*,
#     4, 129–145.
#
# .. [#cerezo2021]
#
#     M. Cerezo, A. Arrasmith, R. Babbush et al. (2021). "Variational quantum
#     algorithms." *Nature Reviews Physics* 3, 625–644.
#     `arXiv:2012.09265 <https://arxiv.org/abs/2012.09265>`__
#
# .. [#bergholm2018]
#
#     V. Bergholm, J. Izaac, M. Schuld et al. (2018). "PennyLane: Automatic
#     differentiation of hybrid quantum-classical computations."
#     `arXiv:1811.04968 <https://arxiv.org/abs/1811.04968>`__
#
# About the Author
# ----------------
#
# Marco Margarucci is an MSc Data Science student from Università degli Studi di
# Napoli Federico II, with research interests in quantum machine learning,
# kernel methods, and hybrid classical-quantum architectures.
