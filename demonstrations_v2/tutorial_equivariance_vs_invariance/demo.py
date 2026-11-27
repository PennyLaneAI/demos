r"""Equivariant circuit or invariant model? Where the advantage of symmetric QNNs comes from
==========================================================================================

In this demo we separate the two ingredients of a symmetric quantum classifier, an equivariant circuit and an
invariant readout, and show that it is the invariance of the whole model that protects it from learning a shortcut.
It follows the paper `Invariance can prevent overfitting and shortcut learning in Quantum Neural Networks
<https://arxiv.org/abs/2611.14368>`__.

Many image classification tasks have labels that do not change when the image is rotated or reflected: a galaxy on
the sky or a blood cell on a microscope slide belongs to the same class in any orientation. *Geometric quantum
machine learning* builds this knowledge into the model, and the usual recipe has two ingredients [#Larocca2022]_
[#Meyer2023]_:

1. an **equivariant circuit** :math:`U(\theta)`, which commutes with the action of the symmetry group on the qubits;
2. an **invariant readout**: an observable :math:`O` that the group maps into itself.

Together they make the model *invariant*: it returns the same output for an image and for any of its rotated or
reflected copies. But the two ingredients are not the same thing, and an equivariant circuit followed by a generic
readout gives a model that is **not** invariant. So which one is responsible for the benefits of symmetric models?

In this demo we answer this question in a controlled setting, following Ref. [#DAmbruoso2026]_. We build four
eight-qubit image classifiers with the same encoding, the same gates, the same number of parameters and the same
dynamical Lie algebra, and vary independently the equivariance of the circuit and the invariance of the model. Then we
train them on a task with a *shortcut*: a small watermark whose position reveals the class in the training images.
We will see that the invariant models cannot learn the shortcut, while the non-invariant ones learn it and fail when
the watermark moves, **even when their circuit is equivariant**.

Images as states of two registers
---------------------------------

We work with :math:`16\times 16` grey-scale images :math:`x_{ij}` and amplitude-encode them on eight qubits:

.. math:: |\psi(x)\rangle = \frac{1}{\lVert x \rVert}\sum_{i,j=0}^{15} x_{ij}\, |i\rangle_\text{row} |j\rangle_\text{col},

with the row index :math:`i` in binary on wires 0–3 and the column index :math:`j` on wires 4–7. The symmetry group
of the square, the dihedral group :math:`D_4`, has eight elements (four rotations, four reflections) and is generated
by three reflections. With this encoding each of them is a layer of simple gates [#Chang2023]_:

* flipping the image upside down, :math:`i \mapsto 15 - i`, flips all the bits of the row index: a layer of
  :math:`X` gates on wires 0–3;
* flipping it left to right is a layer of :math:`X` gates on wires 4–7;
* transposing it, :math:`x_{ij}\mapsto x_{ji}`, swaps the two registers: four SWAP gates.

Let's check this numerically.
"""

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)  # fixed seed for the random images and angles of the checks below

SIDE, N_QUBITS, HALF = 16, 8, 4
ROWS, COLS = range(0, HALF), range(HALF, N_QUBITS)

# the three generators of D4 as functions on images (..., 16, 16) and as gates on the qubits
image_ops = {
    "flip rows": lambda img: torch.flip(img, dims=[-2]),
    "flip columns": lambda img: torch.flip(img, dims=[-1]),
    "transpose": lambda img: img.transpose(-2, -1),
}


def qubit_op(name):
    if name == "flip rows":
        for w in ROWS:
            qml.X(w)
    elif name == "flip columns":
        for w in COLS:
            qml.X(w)
    else:
        for w in ROWS:
            qml.SWAP(wires=[w, w + HALF])


dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev)
def transformed_state(image, name):
    qml.StatePrep(image.reshape(-1), wires=range(N_QUBITS), normalize=True)
    qubit_op(name)
    return qml.state()


image = torch.rand(SIDE, SIDE)
for name, op in image_ops.items():
    expected = op(image).reshape(-1) / image.norm()
    print(
        f"{name:13s}: gates reproduce the image transformation:",
        np.allclose(transformed_state(image, name), expected),
    )

######################################################################
# All eight elements of :math:`D_4` are products of these three gate layers, so a model is invariant under
# :math:`D_4` if it is invariant under each of them. We will use the matrices :math:`R_g` of the three generators to
# check equivariance and invariance below.

R = {name: qml.matrix(qubit_op, wire_order=range(N_QUBITS))(name) for name in image_ops}

######################################################################
# Two circuits with the same gates
# --------------------------------
#
# Our circuits are adapted from Ref. [#Chang2023]_. One layer applies five *blocks*. A block acts on two bits
# :math:`(a, b)` of the row index and on the same two bits :math:`(a', b') = (a+4, b+4)` of the column index:
#
# .. math::
#
#     B(\theta) = e^{-i\frac{\theta_3}{2} Y_a Y_b Y_{a'} Y_{b'}}\,
#                 e^{-i\frac{\theta_2}{2}(X_b + X_{b'})}\, e^{-i\frac{\theta_1}{2}(X_a + X_{a'})}.
#
# Each generator is symmetric in the two registers, so it commutes with the SWAPs; the :math:`X` terms commute with
# the :math:`X` layers, and :math:`Y_aY_b` picks up two minus signs under them, which cancel. This circuit is
# therefore **equivariant**: :math:`U(\theta) R_g = R_g U(\theta)` for every :math:`g` and every :math:`\theta`.
#
# To get a **non-equivariant** circuit with the same structure, we cycle the Pauli axes of the column register,
# :math:`X\to Y` and :math:`Y\to Z`:
#
# .. math::
#
#     B'(\theta) = e^{-i\frac{\theta_3}{2} Y_a Y_b Z_{a'} Z_{b'}}\,
#                  e^{-i\frac{\theta_2}{2}(X_b + Y_{b'})}\, e^{-i\frac{\theta_1}{2}(X_a + Y_{a'})}.
#
# The four "fine" blocks on the bit pairs (0, 1), (2, 3), (1, 2), (3, 0) share three angles, and a "coarse" block on
# (0, 2) has three more, so each layer has six trainable angles in both circuits.

FINE_BLOCKS = ((0, 1), (2, 3), (1, 2), (3, 0))
COARSE_BLOCK = (0, 2)


def layer_gates(equivariant):
    """The gates of one layer as (Pauli word, wires, index of the angle in the layer)."""
    rot = "X" if equivariant else "Y"  # single-qubit rotations on the column register
    ent = "Y" if equivariant else "Z"  # column factors of the four-qubit rotation
    gates = []
    for stage, blocks in ((0, FINE_BLOCKS), (1, (COARSE_BLOCK,))):
        for a, b in blocks:
            for k, q in enumerate((a, b)):
                gates += [("X", [q], 3 * stage + k), (rot, [q + HALF], 3 * stage + k)]
            gates.append(("YY" + 2 * ent, [a, b, a + HALF, b + HALF], 3 * stage + 2))
    return gates


def circuit(params, equivariant):
    """params has shape (layers, 6)."""
    for layer in params:
        for word, wires, k in layer_gates(equivariant):
            if len(wires) == 1:
                (qml.RX if word == "X" else qml.RY)(layer[k], wires=wires)
            else:
                qml.PauliRot(layer[k], word, wires=wires)


######################################################################
# Let's verify equivariance for random angles.

params = torch.rand(2, 6) * 2 * np.pi
for equivariant in (True, False):
    U = qml.matrix(circuit, wire_order=range(N_QUBITS))(params.numpy(), equivariant)
    commutes = all(np.allclose(U @ Rg, Rg @ U) for Rg in R.values())
    print(
        f"{'equivariant' if equivariant else 'non-equivariant':16s} circuit commutes with D4: {commutes}"
    )

######################################################################
# Same circuits, same expressivity
# --------------------------------
#
# If we want to attribute a difference in performance to symmetry, the two circuits should not differ in anything
# else. They have the same gates (up to a relabelling of the Pauli axes on four qubits), the same structure and the
# same parameters. They are also equally *expressive*. The set of unitaries that a circuit can reach is fixed by its
# **dynamical Lie algebra** (DLA) [#Ragone2024]_, the Lie algebra spanned by the gate generators and all their nested
# commutators, which one can compute with :func:`~pennylane.lie_closure`. On eight qubits this takes a while, but
# here we do not need it: the cyclic change of axes :math:`X\to Y\to Z\to X` is a fixed rotation by
# :math:`2\pi/3` about the axis :math:`(1,1,1)/\sqrt 3` of the Bloch sphere. Applying it to the four column qubits,
# :math:`V`, maps every generator of the equivariant circuit to the corresponding generator of the non-equivariant
# one. Hence :math:`U'(\theta) = V U(\theta) V^\dagger` for every :math:`\theta`, and the two DLAs are unitarily
# equivalent. Let's check this.


def axis_cycle():
    """V: X -> Y -> Z -> X on each qubit of the column register."""
    for w in COLS:
        qml.exp(qml.X(w) + qml.Y(w) + qml.Z(w), -1j * np.pi / (3 * np.sqrt(3)))


V = qml.matrix(axis_cycle, wire_order=range(N_QUBITS))()
U_eq = qml.matrix(circuit, wire_order=range(N_QUBITS))(params.numpy(), True)
U_neq = qml.matrix(circuit, wire_order=range(N_QUBITS))(params.numpy(), False)
print("U' = V U V^dagger:", np.allclose(U_neq, V @ U_eq @ V.conj().T))

######################################################################
# The DLA of both circuits has dimension 569, out of the :math:`4^8 - 1 = 65535` of
# :math:`\mathfrak{su}(256)` [#DAmbruoso2026]_. The two circuits reach equally rich sets of unitaries; what differs
# is how these unitaries sit relative to the symmetry of the images. Note that :math:`V` does not commute with the
# swap of the registers, which is why it can turn an equivariant circuit into a non-equivariant one.
#
# Four models
# -----------
#
# The output of a model is the expectation value of an observable, :math:`m(x) = \langle\psi(x)|U^\dagger O
# U|\psi(x)\rangle`, and the predicted class is the sign of :math:`m(x)`. We consider two readouts:
#
# * :math:`O = (X_0 + X_4)/2`, which commutes with all :math:`R_g`: **invariant**;
# * :math:`O_\text{ns} = (X_0 + Z_0 + Z_4)/(1+\sqrt 2)`, which no element of :math:`D_4` other than the identity
#   maps into itself (the :math:`X` layers flip the sign of :math:`Z_0` or :math:`Z_4`, the swap maps :math:`X_0` to
#   :math:`X_4`). It is normalized like :math:`O`, so the outputs of every model lie in :math:`[-1, 1]`.
#
# With these we build four models:
#
# ==================  ===============  ============================  ==================
# model               circuit          readout                       invariant model?
# ==================  ===============  ============================  ==================
# Equiv               equivariant      :math:`O`                     yes
# NonEquiv            non-equivariant  :math:`O`                     no
# NonEquiv-Twirled    non-equivariant  :math:`O`, averaged over      yes
#                                      the 8 transformed inputs
# Equiv-NoSym         equivariant      :math:`O_\text{ns}`           no
# ==================  ===============  ============================  ==================
#
# The *twirled* model obtains invariance in a different way: it runs the non-equivariant circuit on the eight
# transformed copies :math:`gx` of the image and averages the outputs. This always gives an invariant model, at the
# price of eight times as many circuit evaluations.
#
# Comparing Equiv, NonEquiv-Twirled and NonEquiv tells us how invariance is obtained; comparing Equiv and Equiv-NoSym
# tells us what happens to the *same* equivariant circuit with and without an invariant readout.

O = 0.5 * (qml.X(0) + qml.X(HALF))
O_NS = (qml.X(0) + qml.Z(0) + qml.Z(HALF)) / (1 + np.sqrt(2))

MODELS = {
    "Equiv": dict(equivariant=True, readout=O, twirl=False),
    "NonEquiv": dict(equivariant=False, readout=O, twirl=False),
    "NonEquiv-Twirled": dict(equivariant=False, readout=O, twirl=True),
    "Equiv-NoSym": dict(equivariant=True, readout=O_NS, twirl=False),
}


def d4(imgs, g):
    """Element g = 0, ..., 7 of D4 on images (..., 16, 16): g mod 4 quarter turns, then a reflection if g >= 4."""
    out = torch.rot90(imgs, g % 4, dims=(-2, -1))
    return torch.flip(out, dims=[-1]) if g >= 4 else out


def make_model(name):
    spec = MODELS[name]

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def qnode(states, params):
        qml.StatePrep(states, wires=range(N_QUBITS), normalize=True)
        circuit(params, spec["equivariant"])
        return qml.expval(spec["readout"])

    def model(images, params):
        """images (batch, 16, 16) -> outputs (batch,)."""
        if not spec["twirl"]:
            return qnode(images.reshape(len(images), -1), params)
        copies = torch.cat([d4(images, g) for g in range(8)]).reshape(8 * len(images), -1)
        return qnode(copies, params).reshape(8, len(images)).mean(dim=0)

    return model


models = {name: make_model(name) for name in MODELS}

######################################################################
# Let's check which models are invariant: we feed a random image and its eight transformed copies to each model, with
# random angles, and look at the spread of the outputs.

image = torch.rand(SIDE, SIDE)
copies = torch.stack([d4(image, g) for g in range(8)])
for name, model in models.items():
    out = model(copies, params).detach()
    print(f"{name:17s} max |m(gx) - m(x)| = {(out - out[0]).abs().max():.1e}")

######################################################################
# As expected, Equiv and NonEquiv-Twirled give the same output on all eight copies, while NonEquiv and Equiv-NoSym do
# not. Note that Equiv-NoSym has an equivariant circuit: equivariance of the circuit alone does not make the model
# invariant.
#
# A task with a shortcut
# ----------------------
#
# We now train the four models on a synthetic task: a filled square (class 0) or a thin plus-shaped cross (class 1) of
# random size, position and intensity, on a noisy background. Both shapes are symmetric, and their positions are
# uniform over the frame, so the label is invariant under :math:`D_4`.
#
# To probe what the models learn, we add a **shortcut** to the training images: a :math:`2\times 2` watermark near a
# corner, whose position depends on the class. The watermark of class 1 is the watermark of class 0 rotated by 90
# degrees. It always carries 20% of the squared norm of the image, so a model that reads it can classify the training
# set perfectly without looking at the shape. We evaluate the trained models on three test sets of fresh images:
#
# * **clean**: no watermark, i.e., the true rule;
# * **cued**: with the watermark of the true class, as in training;
# * **cued, moved**: the cued images, each transformed by a random element of :math:`D_4` (the watermark moves with
#   the image). The shape still determines the label, but the position of the watermark no longer does.


def make_shapes(n, seed, noise=0.05):
    g = torch.Generator().manual_seed(seed)
    labels = torch.arange(n) % 2
    labels = labels[torch.randperm(n, generator=g)]
    imgs = (torch.randn(n, SIDE, SIDE, generator=g) * noise).abs()
    rint = lambda lo, hi: int(torch.randint(lo, hi + 1, (1,), generator=g))
    for i in range(n):
        v = 0.5 + 0.5 * torch.rand(1, generator=g).item()
        if labels[i] == 0:  # filled square of side 8-14
            side = rint(8, 14)
            r, c = rint(0, SIDE - side), rint(0, SIDE - side)
            imgs[i, r : r + side, c : c + side] = v
        else:  # cross one pixel thick, with odd span 9-15
            span = rint(9, 15)
            span += (span - 1) % 2
            r, c = rint(0, SIDE - span), rint(0, SIDE - span)
            o = (span - 1) // 2
            imgs[i, r : r + span, c + o] = v
            imgs[i, r + o, c : c + span] = v
    return imgs / imgs.flatten(1).norm(dim=1)[:, None, None], labels


WATERMARK_SHARE = 0.2
base_mask = torch.zeros(SIDE, SIDE, dtype=torch.bool)
base_mask[0:2, 2:4] = True  # not on a symmetry axis of the square
masks = [d4(base_mask, g) for g in (0, 1)]  # class 0, class 1 (rotated by 90 degrees)


def add_watermark(imgs, labels):
    imgs = imgs.clone()
    for i in range(len(imgs)):
        m = masks[int(labels[i])]
        rest = (imgs[i] ** 2).sum() - (imgs[i][m] ** 2).sum()
        imgs[i][m] = torch.sqrt(WATERMARK_SHARE * rest / (m.sum() * (1 - WATERMARK_SHARE)))
    return imgs / imgs.flatten(1).norm(dim=1)[:, None, None]


def random_pose(imgs, seed):
    gs = torch.randint(8, (len(imgs),), generator=torch.Generator().manual_seed(seed))
    return torch.stack([d4(img, int(g)) for img, g in zip(imgs, gs)])


x_test, y_test = make_shapes(400, seed=12345)
x_cued = add_watermark(x_test, y_test)
test_sets = {"clean": x_test, "cued": x_cued, "cued, moved": random_pose(x_cued, seed=1)}

fig, axes = plt.subplots(1, 4, figsize=(9, 2.6))
examples = [
    (x_test, 0, "class 0, clean"),
    (x_test, 1, "class 1, clean"),
    (x_cued, 0, "class 0, cued"),
    (test_sets["cued, moved"], 0, "class 0, cued, moved"),
]
for ax, (xs, cls, title) in zip(axes, examples):
    ax.imshow(xs[(y_test == cls).nonzero()[0, 0]], cmap="gray_r")
    ax.set_title(title, fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.show()

######################################################################
# Why can't an invariant model use the watermark? The two class watermarks lie in the same :math:`D_4` orbit: the
# watermark of class 1 is a rotated copy of the watermark of class 0. An invariant model gives the same output to an
# image and its rotated copy, so it cannot tell the two watermarks apart by their position. It *has* to look at the
# shape. A non-invariant model has no such constraint.
#
# Training
# --------
#
# We train every model on only :math:`N = 20` images, with two layers (12 angles), using the logistic loss
# :math:`\log(1 + e^{-\beta t\, m(x)})` with :math:`t = \pm 1` and :math:`\beta = 20` and the Adam optimizer, starting
# from angles close to zero. We repeat the training with three random training sets for each model.

N_TRAIN, LAYERS, EPOCHS, BATCH, BETA, SEEDS = 20, 2, 70, 2, 20.0, (0, 1, 2)


def loss_fn(outputs, labels):
    return torch.nn.functional.softplus(-BETA * (2.0 * labels - 1.0) * outputs).mean()


def train(model, x, y, seed):
    gen = torch.Generator().manual_seed(seed)
    params = (torch.rand(LAYERS, 6, generator=gen) * 0.2 - 0.1).requires_grad_()
    opt = torch.optim.Adam([params], lr=0.05, betas=(0.5, 0.999))
    for _ in range(EPOCHS):
        for batch in torch.randperm(len(x), generator=gen).split(BATCH):
            opt.zero_grad()
            loss_fn(model(x[batch], params), y[batch]).backward()
            opt.step()
    return params.detach()


@torch.no_grad()
def accuracy(model, x, y, params):
    return ((model(x, params) > 0).long() == y).double().mean().item()


results = {name: {test: [] for test in ["train", *test_sets]} for name in models}
for seed in SEEDS:
    x_train, y_train = make_shapes(N_TRAIN, seed=seed)
    x_train = add_watermark(x_train, y_train)
    for name, model in models.items():
        params = train(model, x_train, y_train, seed)
        results[name]["train"].append(accuracy(model, x_train, y_train, params))
        for test, xs in test_sets.items():
            results[name][test].append(accuracy(model, xs, y_test, params))

print(f"{'model':17s}" + "".join(f"{test:>13s}" for test in ["train", *test_sets]))
for name, res in results.items():
    print(f"{name:17s}" + "".join(f"{np.mean(v):13.2f}" for v in res.values()))

######################################################################
# Let's plot the test accuracies.

fig, ax = plt.subplots(figsize=(7, 3.5))
width = 0.2
for k, (name, res) in enumerate(results.items()):
    means = [np.mean(res[test]) for test in test_sets]
    errs = [np.std(res[test]) for test in test_sets]
    ax.bar(np.arange(3) + (k - 1.5) * width, means, width, yerr=errs, capsize=3, label=name)
ax.axhline(0.5, color="gray", lw=0.8, ls="--")
ax.set_xticks(range(3), list(test_sets))
ax.set_ylabel("test accuracy")
ax.set_ylim(0.4, 1.0)
ax.legend(fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
plt.tight_layout()
plt.show()

######################################################################
# All four models fit the training set almost perfectly, and all of them do well on the cued test images, where the
# watermark is where they saw it in training. The difference appears when the watermark moves:
#
# * The two **invariant** models (Equiv and NonEquiv-Twirled) have exactly the same accuracy on the cued and on the
#   moved images. This is guaranteed by their invariance: the moved images are transformed copies of the cued ones.
# * The two **non-invariant** models (NonEquiv and Equiv-NoSym) lose accuracy when the watermark moves; Equiv-NoSym
#   drops to chance. On the clean images Equiv-NoSym is also at chance: it learned almost nothing but the watermark.
# * **Equiv-NoSym behaves like NonEquiv, not like Equiv**, although it has the same equivariant circuit as Equiv. The
#   readout alone decides whether the model can use the shortcut.
#
# Removing the watermark also changes the images: it carried a fifth of their squared norm. For an invariant model
# this shifts the output by about the same amount for both classes, which moves some test images across the decision
# threshold :math:`m(x) = 0`; this is why the invariant models also lose some accuracy on the clean test set
# [#DAmbruoso2026]_. With three training sets of 20 images the numbers fluctuate by a few percent; the paper uses
# more training sets and larger test sets.
#
# Conclusion
# ----------
#
# We compared quantum classifiers that differ *only* in their symmetry: same encoding, gates, parameters and
# dynamical Lie algebra. The benefit of building in the symmetry of the task came from the **invariance of the
# model**, not from the equivariance of the circuit: the equivariant circuit with a non-invariant readout learned the
# shortcut just like the non-equivariant one, while the twirled model, whose circuit is not equivariant, was immune to
# it.
#
# Equivariance remains a convenient way to get invariance: Equiv needs one circuit evaluation per image, the twirled
# model needs eight, one per element of the group, and for larger groups twirling quickly becomes impractical. In
# Ref. [#DAmbruoso2026]_ the same picture holds on MNIST digits in random orientations and on two scientific datasets,
# galaxies and red blood cells: invariant models trained on a few tens of images also generalize better than
# non-invariant ones, and the gain in accuracy persists under a realistic noise model. The code of every experiment
# of the paper, the results of every run and the scripts that rebuild its figures are available at
# `github.com/giuseppedambruoso/eqnn <https://github.com/giuseppedambruoso/eqnn>`__.
#
# A practical take-away when designing symmetric QML models: check the invariance of the *whole model*, readout
# included, as we did above with a few lines of PennyLane.
#
# References
# ----------
#
# .. [#Larocca2022]
#
#     M. Larocca, F. Sauvage, F. M. Sbahi, G. Verdon, P. J. Coles, M. Cerezo,
#     "Group-invariant quantum machine learning",
#     `PRX Quantum 3, 030341 (2022) <https://doi.org/10.1103/PRXQuantum.3.030341>`__.
#
# .. [#Meyer2023]
#
#     J. J. Meyer, M. Mularski, E. Gil-Fuster, A. A. Mele, F. Arzani, A. Wilms, J. Eisert,
#     "Exploiting symmetry in variational quantum machine learning",
#     `PRX Quantum 4, 010328 (2023) <https://doi.org/10.1103/PRXQuantum.4.010328>`__.
#
# .. [#Chang2023]
#
#     S. Y. Chang, M. Grossi, B. Le Saux, S. Vallecorsa,
#     "Approximately equivariant quantum neural network for p4m group symmetries in images",
#     `IEEE QCE 2023 <https://arxiv.org/abs/2310.02323>`__.
#
# .. [#Ragone2024]
#
#     M. Ragone, B. N. Bakalov, F. Sauvage, A. F. Kemper, C. Ortiz Marrero, M. Larocca, M. Cerezo,
#     "A Lie algebraic theory of barren plateaus for deep parameterized quantum circuits",
#     `Nature Communications 15, 7172 (2024) <https://doi.org/10.1038/s41467-024-49909-3>`__.
#
# .. [#DAmbruoso2026]
#
#     G. D'Ambruoso, G. Gramegna, D. Pomarico, G. Magnifico, C. Lupo,
#     "Invariance can prevent overfitting and shortcut learning in Quantum Neural Networks",
#     `arXiv:2611.14368 <https://arxiv.org/abs/2611.14368>`__ (2026). Code, data and results: `github.com/giuseppedambruoso/eqnn <https://github.com/giuseppedambruoso/eqnn>`__.
#
