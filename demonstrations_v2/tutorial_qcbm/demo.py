r"""Quantum Circuit Born Machines
=============================

Unsupervised generative modelling emerges as a promising application for achieving practical quantum advantage
on classical data due to its high complexity relative to supervised machine learning tasks. This makes them
a suitable candidate for leveraging the potential of near-term quantum computers. A popular quantum generative
model known as Quantum Circuit Born Machines (QCBMs) has shown impressive results in modelling distributions
across various datasets, including both toy and real-world datasets, and you will learn about them in this demo.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_large_QuantumCircuitBornMachines_2024-05-13.png
    :align: center
    :width: 50%
    :target: javascript:void(0)


Generative modelling with QCBMs
-------------

Quantum Circuit Born machines (QCBMs) show promise in unsupervised generative modelling, aiming to learn and
represent classical dataset probability distributions through pure quantum states [#Liu]_ [#Ben]_. They are popular
due to their high expressive power [#Du]_. Born machines leverage the probabilistic interpretation of
quantum wavefunctions, representing probability distributions with pure quantum states instead of
thermal distributions like Boltzmann machines. This allows Born machines to directly generate
samples through projective measurements on qubits, offering a faster alternative to the Gibbs
sampling approach [#Ack]_.

For a dataset :math:`\mathcal{D} = \{x\}` with independent and identically distributed samples from
an unknown target distribution :math:`\pi(x),` QCBM is used to generate samples closely resembling
the target distribution. QCBM transforms the input product state :math:`|\textbf{0} \rangle` to a
parameterized quantum state :math:`|\psi_\boldsymbol{\theta}\rangle.` Measuring this output state in the
computational basis yields a sample of bits :math:`x \sim p_\theta(x).`

.. math::

   p_\boldsymbol{\theta}(x) = |\langle x | \psi_\boldsymbol{\theta} \rangle|^2.

The objective is to align the model probability distribution :math:`p_\boldsymbol{\theta}` with the target
distribution :math:`\pi.`

In this tutorial, following [#Liu]_, we will implement a gradient-based algorithm for QCBM using
PennyLane. We describe the model and learning algorithm followed by its application to the
:math:`3 \times 3` Bars and Stripes dataset and double Gaussian peaks.

To train the QCBM, we use the squared maximum mean discrepancy (MMD) as the loss function

.. math::

    \mathcal{L}(\boldsymbol{\theta}) = \left\|\sum_{x} p_\boldsymbol{\theta}(x) \phi(x)- \sum_{x} \pi(x) \phi(x)  \right\|^2,


where :math:`\phi(x)` maps :math:`x` to a larger feature space. Using a kernel
:math:`K(x,y) = \phi(x)^T\phi(y)` allows us to work in a lower-dimensional space. We use the Radial
basis function (RBF) kernel for this purpose, which is defined as:

.. math::

   K(x,y) = \frac{1}{c}\sum_{i=1}^c \exp \left( \frac{|x-y|^2}{2\sigma_i^2} \right).

Here, :math:`\sigma_i` is the bandwidth parameter controlling the Gaussian kernel's width.
:math:`\mathcal{L}` approaches to zero if and only if :math:`p_\boldsymbol{\theta}` approaches :math:`\pi` [#Gret]_.
To learn more about kernel methods, check out this :doc:`demo <demos/tutorial_kernels_module>`.

We can now write the loss function in terms of :math:`K(x,y)` as

.. math::

    \mathcal{L} = \underset{x, y \sim p_\boldsymbol{\theta}}{\mathbb{E}}[{K(x,y)}]-2\underset{x\sim p_\boldsymbol{\theta},y\sim \pi}{\mathbb{E}}[K(x,y)]+\underset{x, y \sim \pi}{\mathbb{E}}[K(x, y)]
"""

######################################################################
# Armed with these ingredients, we can write code for the QCBM and the loss function.
# We first define the ``MMD`` class for computing the squared MMD loss with radial basis
# function kernel. Upon initialization, it calculates the kernel function.
#

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class MMD:

    def __init__(self, scales, space):
        gammas = 1 / (2 * (scales**2))
        sq_dists = jnp.abs(space[:, None] - space[None, :]) ** 2
        self.K = sum(jnp.exp(-gamma * sq_dists) for gamma in gammas) / len(scales)
        self.scales = scales

    def k_expval(self, px, py):
        # Kernel expectation value
        return px @ self.K @ py

    def __call__(self, px, py):
        pxy = px - py
        return self.k_expval(pxy, pxy)


######################################################################
# Defining classes helps in caching the kernel instead of calculating it everytime to find the
# expectation value. Next up, the ``QCBM`` holds the definition for the quantum circuit born machine
# and the objective function to minimize.
#

from functools import partial


class QCBM:

    def __init__(self, circ, mmd, py):
        self.circ = circ
        self.mmd = mmd
        self.py = py  # target distribution π(x)

    @partial(jax.jit, static_argnums=0)
    def mmd_loss(self, params):
        px = self.circ(params)
        return self.mmd(px, self.py), px


######################################################################
# Learning the Bars and Stripes data distribution
# -----------------------------------------------
#
# We train the QCBM on the Bars and stripes dataset. The dataset has binary black and white images of
# size :math:`n \times n` pixels. We consider :math:`n=3` for this tutorial, which will give 14 valid
# configurations. The dataset is represented by flattened bitstrings. The quantum circuit will use 9
# qubits in total.
#

######################################################################
# Data generation
# ~~~~~~~~~~~~~~~
#

import numpy as np


def get_bars_and_stripes(n):
    bitstrings = [list(np.binary_repr(i, n))[::-1] for i in range(2**n)]
    bitstrings = np.array(bitstrings, dtype=int)

    stripes = bitstrings.copy()
    stripes = np.repeat(stripes, n, 0)
    stripes = stripes.reshape(2**n, n * n)

    bars = bitstrings.copy()
    bars = bars.reshape(2**n * n, 1)
    bars = np.repeat(bars, n, 1)
    bars = bars.reshape(2**n, n * n)
    return np.vstack((stripes[0 : stripes.shape[0] - 1], bars[1 : bars.shape[0]]))


n = 3
size = n**2
data = get_bars_and_stripes(n)
print(data.shape)

######################################################################
# The dataset has 9 features per data point. A visualization of one data point is shown below. Each
# data point represents a flattened bitstring.
#

import matplotlib.pyplot as plt

sample = data[1].reshape(n, n)

plt.figure(figsize=(2, 2))
plt.imshow(sample, cmap="gray", vmin=0, vmax=1)
plt.grid(color="gray", linewidth=2)
plt.xticks([])
plt.yticks([])

for i in range(n):
    for j in range(n):
        text = plt.text(
            i,
            j,
            sample[j][i],
            ha="center",
            va="center",
            color="gray",
            fontsize=12,
        )

print(f"\nSample bitstring: {''.join(np.array(sample.flatten(), dtype='str'))}")

######################################################################
# We can plot the full dataset of :math:`3 \times 3` images.
#

plt.figure(figsize=(4, 4))
j = 1
for i in data:
    plt.subplot(4, 4, j)
    j += 1
    plt.imshow(np.reshape(i, (n, n)), cmap="gray", vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])

######################################################################
# Next we compute the integers represented by the valid configurations. We will use them later
# to evaluate the performance of the QCBM.
#

bitstrings = []
nums = []
for d in data:
    bitstrings += ["".join(str(int(i)) for i in d)]
    nums += [int(bitstrings[-1], 2)]
print(nums)

######################################################################
# Using the dataset, we can compute the target probability distribution :math:`\pi(x)` which is visualized below.
#

probs = np.zeros(2**size)
probs[nums] = 1 / len(data)

plt.figure(figsize=(12, 5))
plt.bar(np.arange(2**size), probs, width=2.0, label=r"$\pi(x)$")
plt.xticks(nums, bitstrings, rotation=80)

plt.xlabel("Samples")
plt.ylabel("Prob. Distribution")
plt.legend(loc="upper right")
plt.subplots_adjust(bottom=0.3)
plt.show()

######################################################################
# One can observe that it is a uniform distribution and only the probabilities of the valid configurations are
# non-zero while rest are zero. Next we define a parameterized quantum circuit to train. This quantum circuit
# will act as a generative model, thus, realize a Born machine.
#

import pennylane as qml
qml.capture.enable()

np.random.seed(42)


n_qubits = size
dev = qml.device("default.qubit", wires=n_qubits)

n_layers = 6
wshape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
weights = np.random.random(size=wshape)


@qml.qnode(dev)
def circuit(weights):
    qml.StronglyEntanglingLayers(weights=weights, ranges=[1] * n_layers, wires=range(n_qubits))
    return qml.probs()


jit_circuit = jax.jit(circuit)

######################################################################
# Using the ``MMD`` and ``QCBM`` classes defined earlier, we create their respective objects. Using Optax, we
# define the Adam optimizer.
#

import optax

bandwidth = jnp.array([0.25, 0.5, 1])
space = jnp.arange(2**n_qubits)

mmd = MMD(bandwidth, space)
qcbm = QCBM(jit_circuit, mmd, probs)

opt = optax.adam(learning_rate=0.1)
opt_state = opt.init(weights)

######################################################################
# We can also verify that the summation in the first line of :math:`\mathcal{L}` is equal to the
# expectation values in the second line.
#

loss_1, px = qcbm.mmd_loss(weights)  # squared MMD
loss_2 = mmd.k_expval(px, px) - 2 * mmd.k_expval(px, probs) + mmd.k_expval(probs, probs)
print(loss_1)
print(loss_2)

######################################################################
# Training
# ~~~~~~~~
#
# We define the ``update_step`` method which
#
# - computes the squared MMD loss and gradients.
#
# - apply the update step of our optimizer.
#
# - updates the parameter values.
#
# - calculates the KL divergence.
#
# The KL divergence [#Kull]_ is a measure of how far the predicted distribution :math:`p_\boldsymbol{\theta}(x)`
# is from the target distribution :math:`\pi(x).`
#


@jax.jit
def update_step(params, opt_state):
    (loss_val, qcbm_probs), grads = jax.value_and_grad(qcbm.mmd_loss, has_aux=True)(params)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    log_ratio = jnp.where(qcbm.py == 0, 0, jnp.log(qcbm_probs / qcbm.py))

    kl_div = -jnp.sum(qcbm.py *  log_ratio)
    return params, opt_state, loss_val, kl_div


history = []
divs = []
n_iterations = 100

for i in range(n_iterations):
    weights, opt_state, loss_val, kl_div = update_step(weights, opt_state)

    if i % 10 == 0:
        print(f"Step: {i} Loss: {loss_val:.4f} KL-div: {kl_div:.4f}")

    history.append(loss_val)
    divs.append(kl_div)

######################################################################
# Visualizing the training results, we get the following plot.
#

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(history)
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("MMD Loss")

ax[1].plot(divs, color="green")
ax[1].set_xlabel("Iteration")
ax[1].set_ylabel("KL Divergence")
plt.show()

######################################################################
# Comparing the target probability distribution with the QCBM predictions, we can see that the
# predictions results in a good approximation.
#

qcbm_probs = np.array(qcbm.circ(weights))

plt.figure(figsize=(12, 5))

plt.bar(
    np.arange(2**size),
    probs,
    width=2.0,
    label=r"$\pi(x)$",
    alpha=0.4,
    color="tab:blue",
)
plt.bar(
    np.arange(2**size),
    qcbm_probs,
    width=2.0,
    label=r"$p_\theta(x)$",
    alpha=0.9,
    color="tab:green",
)

plt.xlabel("Samples")
plt.ylabel("Prob. Distribution")

plt.xticks(nums, bitstrings, rotation=80)
plt.legend(loc="upper right")
plt.subplots_adjust(bottom=0.3)
plt.show()

######################################################################
# Testing
# ~~~~~~~
#
# To visualize the performance of the model, we generate samples and compute
# :math:`\chi \equiv` P(:math:`x` is a bar or stripe) which is a measure of
# generation quality.
#


def circuit(weights):
    qml.StronglyEntanglingLayers(weights=weights, ranges=[1] * n_layers, wires=range(n_qubits))
    return qml.sample()


for N in [2000, 20000]:
    dev = qml.device("default.qubit", wires=n_qubits)
    circ = qml.set_shots(qml.QNode(circuit, device=dev), shots = N)
    preds = circ(weights)
    mask = np.any(np.all(preds[:, None] == data, axis=2), axis=1)  # Check for row-wise equality
    chi = np.sum(mask) / N
    print(f"χ for N = {N}: {chi:.4f}")

print(f"χ for N = ∞: {np.sum(qcbm_probs[nums]):.4f}")

######################################################################
# Few of the samples are plotted below. The ones with a red border represents
# invalid images.
#

plt.figure(figsize=(8, 8))
j = 1
for i, m in zip(preds[:64], mask[:64]):
    ax = plt.subplot(8, 8, j)
    j += 1
    plt.imshow(np.reshape(i, (n, n)), cmap="gray", vmin=0, vmax=1)
    if ~m:
        plt.setp(ax.spines.values(), color="red", linewidth=1.5)
    plt.xticks([])
    plt.yticks([])

######################################################################
# The model is able to learn the target distribution due to a circuit with larger layers. Also,
# [#Liu]_ has argued that training a QCBM with deeper circuits does not suffer from the
# vanishing gradients problem.
#

######################################################################
# Learning a mixture of Gaussians
# -------------------------------
#
# Now we use a QCBM to model a mixture of Gaussians
#
# .. math::
#
#    \pi(x)\propto e^{-\frac{1}{2}\left(\frac{x-\mu_1}{\sigma}\right)^2}+e^{-\frac{1}{2}\left(\frac{x-\mu_2}{\sigma}\right)^2},
#
# with :math:`x` ranging from :math:`0 \dots 2^{n}-1,` where :math:`n` is the
# number of qubits.
#


def mixture_gaussian_pdf(x, mus, sigmas):
    mus, sigmas = np.array(mus), np.array(sigmas)
    vars = sigmas**2
    values = [
        (1 / np.sqrt(2 * np.pi * v)) * np.exp(-((x - m) ** 2) / (2 * v)) for m, v in zip(mus, vars)
    ]
    values = np.sum([val / sum(val) for val in values], axis=0)
    return values / np.sum(values)


n_qubits = 6
x_max = 2**n_qubits
x_input = np.arange(x_max)
mus = [(2 / 7) * x_max, (5 / 7) * x_max]
sigmas = [x_max / 8] * 2
data = mixture_gaussian_pdf(x_input, mus, sigmas)

plt.plot(data, label=r"$\pi(x)$")
plt.legend()
plt.show()

######################################################################
# In contrast to the Bars and Stripes dataset, the Gaussian mixture distribution exhibits
# a smooth and non-zero probability for every basis state. Similar to the previous experiment,
# we will create an ansatz and measure probabilities.
#

dev = qml.device("default.qubit", wires=n_qubits)

n_layers = 4
wshape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
weights = np.random.random(size=wshape)


@qml.set_shots(N)
@qml.qnode(dev)
def circuit(weights):
    qml.StronglyEntanglingLayers(
        weights=weights, ranges=[1] * n_layers, wires=range(n_qubits)
    )
    return qml.probs()


jit_circuit = jax.jit(circuit)

qml.draw_mpl(circuit, level="device")(weights)
plt.show()

######################################################################
# With the quantum circuit defined, we are ready to optimize the squared MMD loss function
# which follows a code similar to the Bars and Stripes case.
#

bandwidth = jnp.array([0.25, 60])
space = jnp.arange(2**n_qubits)

mmd = MMD(bandwidth, space)
qcbm = QCBM(jit_circuit, mmd, data)

opt = optax.adam(learning_rate=0.1)
opt_state = opt.init(weights)

history = []
divs = []
n_iterations = 100

for i in range(n_iterations):
    weights, opt_state, loss_val, kl_div = update_step(weights, opt_state)

    if i % 10 == 0:
        print(f"Step: {i} Loss: {loss_val:.4f} KL-div: {kl_div:.4f}")

    history.append(loss_val)
    divs.append(kl_div)

######################################################################
# Finally, we plot the histogram with the probabilities obtained from QCBM and compare
# it with the actual probability distribution.
#

qcbm_probs = qcbm.circ(weights)

plt.plot(range(x_max), data, linestyle="-.", label=r"$\pi(x)$")
plt.bar(range(x_max), qcbm_probs, color="green", alpha=0.5, label="samples")

plt.xlabel("Samples")
plt.ylabel("Prob. Distribution")

plt.legend()
plt.show()

######################################################################
# The histogram (green bars) aligns remarkably well with the exact probability distribution (blue
# dashed curve).
#

######################################################################
# Conclusion
# ----------
#
# In this tutorial, we introduced and implemented Quantum Circuit Born Machine (QCBM) using PennyLane.
# The algorithm is a gradient-based learning involving optimizing the squared MMD loss. We also
# evaluated QCBMs on the Bars and Stripes and two peaks datasets. One can also leverage the differentiable
# learning of the QCBM to solve combinatorial problems where the output is binary strings.
#

######################################################################
# References
# ----------
#
# .. [#Liu]
#
#    Liu, Jin-Guo, and Lei Wang. “Differentiable learning of quantum circuit born machines.” Physical
#    Review A 98.6 (2018): 062324.
#
# .. [#Ben]
#
#    Benedetti, Marcello, et al. “A generative modeling approach for benchmarking and training shallow
#    quantum circuits.” npj Quantum Information 5.1 (2019): 45.
#
# .. [#Du]
#
#    Du, Yuxuan, et al. "Expressive power of parametrized quantum circuits." Physical Review Research
#    2.3 (2020): 033125.
#
# .. [#Ack]
#
#    Ackley, David H., Geoffrey E. Hinton, and Terrence J. Sejnowski. "A learning algorithm for
#    Boltzmann machines." Cognitive science 9.1 (1985): 147-169.
#
# .. [#Gret]
#
#    Gretton, Arthur, et al. "A kernel method for the two-sample-problem." Advances in neural
#    information processing systems 19 (2006).
#
# .. [#Kull]
#
#    Kullback, Solomon, and Richard A. Leibler. "On information and sufficiency." The annals
#    of mathematical statistics 22.1 (1951): 79-86.
#
