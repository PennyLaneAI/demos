r"""Fast optimization of instantaneous quantum polynomial circuits
===============================================================

Instantaneous Quantum Polynomial (IQP) circuits are a class of circuits that are expected to be hard
to sample from using classical computers [#marshall1]_. In this demo, we take a look at the `IQPopt <https://github.com/XanaduAI/iqpopt>`__ package [#recio1]_,
which shows that despite this, such circuits can still be optimized efficiently!

As we will see, this hinges on a surprising fact about these circuits: while sampling is hard,
estimating expectation values of certain observables is easy.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_iqp_circuit_optimization_jax.png
    :align: center
    :width: 70%
    :target: javascript:void(0)
    :alt: IQP circuit optimization

.. note::
    The IQP circuit and expectation values estimator are now directly implemented and
    available in PennyLane v0.44 via the :class:`IQP <pennylane.IQP> template
    and :func:`~pennylane.qnn.iqp_expval` function, respectively.

Parameterized IQP circuits
--------------------------

IQPopt is designed to optimize a class of IQP circuits called *parameterized IQP circuits*. These are
comprised of gates :math:`\text{exp}(i\theta_j X_j)`, where the generator :math:`X_j` is a tensor
product of `Pauli X operators <https://docs.pennylane.ai/en/stable/code/api/pennylane.PauliX.html>`__ acting on some subset of qubits and :math:`\theta_j` is a trainable
parameter. We will represent the parameterized gates by a list


"""

######################################################################
#
# .. raw:: html
#
#    <center>
#
# ``gates = [gen1, gen2, ...]``
#
# .. raw:: html
#
#    </center>
#
# that specifies the generators of the gates (each generator with an independent trainable parameter).
# Each element of ``gates`` is associated with a different parameter, and is a list of lists of integers
# that specifies the generator for that parameter. For example,
#
# .. raw:: html
#
#    <center>
#
# ``gen1 = [[0,1]]``
#
# .. raw:: html
#
#    </center>
#
# specifies that the generator of the first gate is :math:`X\otimes X` acting on
# the first two qubits of the circuit, i.e. `wires` `0` and `1`. We can also define generators that are sums
# of tensor products of Pauli X operators by including more elements. For example,
#
# .. raw:: html
#
#    <center>
#
# ``gen1 = [[0,1], [0]]``
#
# .. raw:: html
#
#    </center>
#
# would correspond to a gate with a single parameter with generator :math:`X\otimes X` +
# :math:`X\otimes I`.
#
# Here we will work with the following set of gates
#
gates = [[[0]], [[1]], [[2]], [[0,1]], [[0,2]], [[1,2]]]
######################################################################
# i.e. one and two body generators acting on three qubits, each with an independent parameter.
#
# Expectation values
# ------------------
#
# IQPopt can be applied to problems that involve measuring expectation values of tensor products of Pauli Z observables of
# parameterized IQP circuits.
#
# We will represent these observables with binary lists, where a nonzero element denotes the
# presence of a Pauli Z operator on that qubit. For example, in a three-qubit circuit, the operator
#
# .. math:: O = Z \otimes I \otimes Z
#
# will be represented as
#
# .. raw:: html
#
#    <center>
#
# ``op = [1,0,1]``
#
# .. raw:: html
#
#    </center>
#
# Let's now put this into practice and build a PennyLane circuit out of a ``gates`` list.
#
# Creating an IQP circuit with PennyLane
# --------------------------------------
#
# To build a parameterized IQP circuit in PennyLane (with input state
# :math:`\vert 0 \rangle`), we can use the :class:`~pennylane.IQP` template, which
# takes the gate parameters, the number of qubits in the circuit and the generators of the gates.
# Additionally, we can evaluate expectation values of tensor products of Pauli Z operators
# specified by lists of the form ``op`` above with the following
#
import pennylane as qml
qml.capture.enable()
import numpy as np

# Suppress the warning caused by iqpopt
import warnings
from pennylane.exceptions import PennyLaneDeprecationWarning
warnings.filterwarnings("ignore", category=PennyLaneDeprecationWarning)

def penn_obs(op: np.ndarray) -> qml.operation.Operator:
    """Returns a PennyLane observable from a bitstring representation.

    Args:
        op (np.ndarray): Bitstring representation of the Z operator.

    Returns:
        qml.Observable: PennyLane observable.
    """
    for i, z in enumerate(op):
        if i==0:
            if z:
                obs = qml.Z(i)
            else:
                obs = qml.I(i)
        else:
            if z:
                obs @= qml.Z(i)
    return obs

######################################################################
# Now we have our circuit as
#
def penn_iqp_circuit(params: np.ndarray, gates: list, op: np.ndarray, n_qubits: int) -> qml.measurements.ExpectationMP:
    """Defines the circuit that calculates the expectation value of the operator with the IQP circuit with PennyLane tools.

    Args:
        params (np.ndarray): The parameters of the IQP gates.
        gates (list): The gates representation of the IQP circuit.
        op (np.ndarray): Bitstring representation of the Z operator.
        n_qubits (int): The total number of qubits in the circuit.

    Returns:
        qml.measurements.ExpectationMP: PennyLane circuit with an expectation value.
    """
    qml.IQP(weights=params, num_wires=n_qubits, pattern=gates)
    obs = penn_obs(op)
    return qml.expval(obs)

def penn_iqp_op_expval(params: np.ndarray, gates: list, op: np.ndarray, n_qubits: int) -> float:
    """Calculates the expectation value of the operator with the IQP circuit with PennyLane tools.

    Args:
        params (np.ndarray): The parameters of the IQP gates.
        gates (list): The gates representation of the IQP circuit.
        op (np.ndarray): Bitstring representation of the Z operator.
        n_qubits (int): The total number of qubits in the circuit.

    Returns:
        float: Expectation value.
    """
    dev = qml.device("lightning.qubit", wires=n_qubits)
    penn_iqp_circuit_exe = qml.QNode(penn_iqp_circuit, dev)
    return penn_iqp_circuit_exe(params, gates, op, n_qubits)

######################################################################
# With this, we can now calculate all expectation values of Pauli Z tensors of any parameterized IQP
# circuit we can think of. Let's see an example using our ``gates`` list from above.
#
n_qubits = 3
op = np.array([1,0,1]) # operator ZIZ
params = np.random.rand(len(gates)) # random parameters for all the gates (remember we have 5 gates with 6 generators in total)

penn_op_expval = penn_iqp_op_expval(params, gates, op, n_qubits)
print("Expectation value: ", penn_op_expval)

######################################################################
#
# Estimating expectation values with IQPopt
# -----------------------------------------
#
# We can perform the same operations as our PennyLane circuit above, although using approximations instead of exact values. The benefit is that we can work with very
# large circuits.
#
# Starting from a paper on simulating quantum computers with probabilistic methods [#nest]_ from Van den Nest (Theorem 3), one can arrive at the following expression
# for expectation values of Pauli Z operators
#
# .. math:: \langle Z_{\boldsymbol{a}} \rangle = \mathbb{E}_{\boldsymbol{z}\sim U}\Big[ \cos\Big(\sum_j \theta_{j}(-1)^{\boldsymbol{g}_{j}\cdot \boldsymbol{z}}(1-(-1)^{\boldsymbol{g}_j\cdot \boldsymbol{a}}\Big) \Big],
#
# where:
#
# - :math:`\boldsymbol{a}` is the bitstring representation of the operator whose expectation value we want to compute.
# - :math:`\boldsymbol{z}` represents bitstring samples drawn from the uniform distribution :math:`U`.
# - :math:`\theta_{j}` are the trainable parameters.
# - :math:`\boldsymbol{g}_{j}` are the different generators, also represented as bitstrings.
#
# Although this expression is exact, computing the expectation exactly requires an exponential number of samples :math:`\boldsymbol{z}`. Instead, we can
# replace the expectation with an empirical mean and compute an unbiased estimate of
# :math:`\langle Z_{\boldsymbol{a}} \rangle` efficiently. That is, if we sample a batch of :math:`s`
# bitstrings :math:`\{\boldsymbol{z}_i\}` from the uniform distribution and compute the sample mean
#
# .. math:: \hat{\langle Z_{\boldsymbol{a}}\rangle} = \frac{1}{s}\sum_{i}\cos\Big(\sum_j \theta_j(-1)^{\boldsymbol{g}_j\cdot \boldsymbol{z}_i}(1-(-1)^{\boldsymbol{g}_j\cdot \boldsymbol{a}})\Big),
#
# we obtain an unbiased estimate :math:`\hat{\langle Z_{\boldsymbol{a}}\rangle}` of
# :math:`\langle Z_{\boldsymbol{a}}\rangle`, meaning that
#
# .. math:: \mathbb{E}[\hat{\langle Z_{\boldsymbol{a}}\rangle}] = \langle Z_{\boldsymbol{a}}\rangle
#
# The error of this approximation is well known since, by the central limit theorem, the standard
# deviation of the sample mean of a bounded random variable decreases as
#
# .. math:: \mathcal{O}(1/\sqrt{s})
#
# where :math:`s` is the number of samples.
#
# Let's see now how to use the :func:`~pennylane.qnn.iqp_expval` function to calculate expectation values, based on the same
# arguments in the previous example. First, we create the circuit object with ``IqpSimulator``,
# which takes in the number of qubits ``n_qubits`` and the ``gates`` in our usual format:
#
# .. note:: 
#   Here we are using the estimator method added in PennyLane v0.44 but note that this functionality is also available in
#   IQPopt via the class method ``IqpSimulator.op_expval``)
#
import iqpopt as iqp

small_circuit = iqp.IqpSimulator(n_qubits, gates)

######################################################################
# The :func:`~pennylane.qnn.iqp_expval` function requires a PauliZ operator specified by its binary
# representation ``op``, a parameter array ``params``, the circuit ``gates``, the number of qubits ``n_qubits``,
# a new parameter ``n_samples`` (the number of samples :math:`s`) that controls
# the precision of the approximation (the more the better), and a JAX pseudo random number generator
# key to seed the randomness of the sampling. It returns the expectation value estimate as well as its
# standard error.
#
# Using the same ``params`` and ``op`` as before:
#
from pennylane.qnn import iqp_expval as op_expval
import jax

n_samples = 2000
key = jax.random.PRNGKey(66)

expval, std = op_expval(op, params, small_circuit.gates, small_circuit.n_qubits, n_samples, key)

print("Expectation value:  ", expval)
print("Standard error: ", std)

######################################################################
# Since the calculation is stochastic, the result is not exactly the same as
# the one obtained with PennyLane's ``qml.expval`` method. However, as we can see, they are within the standard error `std`. You can try
# increasing ``n_samples`` in order to obtain a more accurate approximation.
#
# Additionally, this function supports fast batch evaluation of expectation values. By specifying a batch of operators ``ops`` as an array, we can compute expectation values and errors in parallel using the same syntax.
#
ops = np.array([[1,0,0],[0,1,0],[0,0,1]]) # batch of single qubit Pauli Zs

expvals, stds = op_expval(ops, params, small_circuit.gates, small_circuit.n_qubits, n_samples, key)

print("Expectation values: ", expvals)
print("Standard errors: ", stds)

######################################################################
# With PennyLane's simulator method, surpassing 30 qubits would be extremely time-consuming. However, with :func:`~pennylane.qnn.iqp_expval`, we can scale far beyond that with ease.
#
n_qubits = 1000
n_gates = 1000

gates = []
for _ in range(n_gates):
    # First we create a generator with bodyness 2
    gen = list(np.random.choice(n_qubits, 2, replace=False))
    # Each gen will have its independent trainable parameter, so we can directly build the gate
    gates.append([gen])

large_circuit = iqp.IqpSimulator(n_qubits, gates)

params = np.random.rand(len(gates))
op = np.random.randint(0, 2, n_qubits)
n_samples = 1000
key = jax.random.PRNGKey(42)

expval, std = op_expval(op, params, large_circuit.gates, large_circuit.n_qubits, n_samples, key)

print("Expectation value: ", expval)
print("Standard error: ", std)

######################################################################
#
# Sampling and probabilities
# --------------------------
#
# If we measure the output qubits of an IQP circuit, we generate samples of binary vectors according to
# the distribution
#
# .. math:: q_{\boldsymbol{\theta}}(\boldsymbol{x}) \equiv q(\boldsymbol{x}\vert\boldsymbol{\theta})=\vert (\langle \boldsymbol{x} \vert U(\boldsymbol{\theta})\vert 0 \rangle )\vert^2,
#
# where :math:`U(\boldsymbol{\theta})` is the parametrized IQP circuit. For a low number of qubits, we
# can use PennyLane to obtain the output probabilities of the circuit as well as sample from it. Note
# that there is not an efficient algorithm to do this for large numbers of qubits, so PennyLane
# returns an error in this case.
#
# These functions are already implemented in the ``IqpSimulator`` object. The ``.probs()`` method
# works as it does in PennyLane, where the returned array of probabilities is in lexicographic order.
#
sample = small_circuit.sample(params, shots=1)
print("Sample: ", sample)
#
probabilities = small_circuit.probs(params)
print("Probabilities: ", probabilities)

try:
    sample = large_circuit.sample(params, shots=1)
    print(sample) # large circuit will return error
except Exception as e:
    print(e)

try:
    probabilities = large_circuit.probs(params)
    print(probabilities) # large circuit will return error
except Exception as e:
    print(e)

######################################################################
# As we can see, we can't sample or know the probabilities of the circuit for the large one. The only
# efficient approximation algorithm we have is for the calculation of expectation values. Let's see
# how time scales for each of the methods using a logarithmic plot.

import time
import matplotlib.pyplot as plt

range_qubits = range(15, 26)
n_gates = 10
n_samples = 1000

times_op, times_sample, times_probs = [], [], []
for n_qubits in range_qubits:

    gates = []
    for _ in range(n_gates):
        gen = list(np.random.choice(n_qubits, 2, replace=False))
        gates.append([gen])

    circuit = iqp.IqpSimulator(n_qubits, gates)
    params_init = np.random.uniform(0, 2*np.pi, len(gates))
    key = jax.random.PRNGKey(np.random.randint(0, 99999))
    op = np.random.randint(0, 2, n_qubits)

    # Timing op_expval
    start = time.perf_counter()
    op_expval(op, params_init, circuit.gates, circuit.n_qubits, n_samples, key)
    times_op.append(time.perf_counter() - start)

    # Timing sample
    start = time.perf_counter()
    circuit.sample(params_init, shots=1)
    times_sample.append(time.perf_counter() - start)

    # Timing probs
    start = time.perf_counter()
    circuit.probs(params_init)
    times_probs.append(time.perf_counter() - start)

plt.scatter(range_qubits[1:], times_op[1:], label="op_expval")
plt.scatter(range_qubits[1:], times_sample[1:], label="sample")
plt.scatter(range_qubits[1:], times_probs[1:], label="probs")

plt.xlabel("n_qubits")
plt.ylabel("Time [s]")
plt.yscale("log")
plt.title(f"Time vs n_qubits")
plt.legend()
plt.show()

######################################################################
# In the previous figure, you can see that the time to sample or compute probabilities scales
# exponentially, however expectation values are very efficient (the scaling can be shown to be
# linear).
#
# Optimizing an IQPopt circuit
# ----------------------------
#
# Circuits can be optimized via a separate ``Trainer`` class. To instantiate a trainer object, we first
# define a `loss function` (also called an objective function), an `optimizer`, and an initial `stepsize` for
# the gradient descent. Continuing our ``small_circuit`` example from before, below we define a simple
# loss function that is a sum of expectation values returned by ``op_expval()`` .
#
import jax.numpy as jnp

def loss_fn(params, circuit, ops, n_samples, key):
    expvals = op_expval(ops, params, circuit.gates, circuit.n_qubits, n_samples, key)[0]
    return jnp.sum(expvals)

optimizer = "Adam"
stepsize = 0.001

trainer = iqp.Trainer(optimizer, loss_fn, stepsize)

######################################################################
# Any differentiable loss function expressible in `JAX <https://pennylane.ai/qml/demos/tutorial_jax_transformations>`__ can be defined, but must have a first argument
# ``params`` that corresponds to the optimization parameters of the circuit. To minimize the loss
# function, we call the method ``train()`` , which requires the number of iterations ``n_iters`` and
# the initial arguments of the loss function ``loss_kwargs`` given as a dictionary object. This
# dictionary must contain a key ``params`` whose corresponding value specifies the initial parameters.
#
np.random.seed(0)

n_iters = 1000
params_init = np.random.normal(0, 1, len(small_circuit.gates))
n_samples = 100

loss_kwargs = {
    "params": params_init,
    "circuit": small_circuit,
    "ops": ops,
    "n_samples": n_samples,
    "key": key
}

trainer.train(n_iters, loss_kwargs, turbo=100) # the turbo option trains in iteration batches of the number that you input, using jit and lax.scan

trained_params = trainer.final_params
plt.plot(trainer.losses) # plot the loss curve
plt.show()

######################################################################
# This training process finds its global minimum at loss = -3.0, which is the minimum possible with
# the defined loss function.
#
# Generative machine learning tools
# ---------------------------------
#
# The IQPopt package contains a dedicated module ``gen_qml`` with functionality to train and evaluate
# generative models expressed as ``IqpSimulator`` circuits. Note that, since sampling from IQP circuits
# is hard, these circuits may lead to advantages for generative machine learning tasks relative to
# classical models!
#
# Training via the maximum mean discrepancy loss
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The Maximum Mean Discrepancy (MMD) [#gretton]_ is an integral probability metric that measures the
# similarity between two probability distributions and can serve as a loss function to train
# generative models. We will focus on the square of the MMD, which can be written as
#
# .. math::  \text{MMD}^2(\boldsymbol{\theta}) = \mathbb{E}_{\boldsymbol{x},\boldsymbol{y}\sim q_{\boldsymbol{\theta}} }[k(\boldsymbol{x},\boldsymbol{y})] - 2  \mathbb{E}_{\boldsymbol{x} \sim q_{\boldsymbol{\theta}},\boldsymbol{y}\sim p }[k(\boldsymbol{x},\boldsymbol{y})] + \mathbb{E}_{\boldsymbol{x},\boldsymbol{y}\sim p }[k(\boldsymbol{x},\boldsymbol{y})] \,,
#
# Using a Gaussian kernel,
#
# .. math::  k(\boldsymbol{x},\boldsymbol{y}) = \exp\Big(\frac{\vert\vert \boldsymbol{x}-\boldsymbol{y} \vert\vert^2}{2\sigma^2}\Big) \,,
#
# it has one parameter called ``sigma``, the bandwidth of this kernel. The
# squared MMD is typically calculated using samples from both probability distributions, and we have an implementation available for this in the ``gen_qml`` module.
#
import iqpopt.gen_qml as genq
from iqpopt.gen_qml.utils import median_heuristic

n_bits = 100
# toy datasets of low weight bitstrings
X1 = np.random.binomial(1, 0.05, size=(100, n_bits))
X2 = np.random.binomial(1, 0.07, size=(100, n_bits))

sigma = median_heuristic(X1) # bandwidth for MMD

mmd = genq.mmd_loss_samples(X1, X2, sigma)
print(mmd)

######################################################################
# This metric can also be estimated efficiently with expectation values of Pauli Z operators only [#recio1]_.
# This means that if we have an ``IqpSimulator`` object, we can also estimate the MMD loss.
#
# The implementation in the ``gen_qml`` module only needs an additional parameter, ``n_ops``, that
# controls the accuracy of this value. For each of these ``n_ops``, an expectation value will be
# calculated with ``n_samples``. The higher the number of operators and samples, the better the
# precision.
#
# Let's see it in an example with 20 qubits:
#
n_qubits = 20

# toy dataset of low weight bitstrings (from ground truth p)
ground_truth = np.random.binomial(1, 0.2, size=(100, n_qubits))

gates = iqp.utils.local_gates(n_qubits, 3)  # all gates with Pauli weight <= 3
circuit = iqp.IqpSimulator(n_qubits, gates)

params = np.random.normal(0, 0.1, len(gates))
sigma = median_heuristic(ground_truth)/3 # bandwidth for MMD

print("Sigma:", sigma)

mmd = genq.mmd_loss_iqp(params,
                       circuit,
                       ground_truth,
                       sigma,
                       n_ops=1000,
                       n_samples=1000,
                       key=jax.random.PRNGKey(42))
print("MMD: ", mmd)

######################################################################
# Now, similar to what we did a few sections back in *Optimizing a circuit*, this function can be used
# with a ``Trainer`` object to train a quantum generative model given as a parameterized IQP circuit.
#
# Using the last defined 20 qubits circuit:
#
X_train = np.random.binomial(1, 0.2, size=(1000, n_qubits))
params_init = np.random.normal(0, 0.1, len(gates))
loss = genq.mmd_loss_iqp # MMD loss

loss_kwargs = {
    "params": params_init,
    "iqp_circuit": circuit,
    "ground_truth": X_train, # samples from ground truth distribution
    "sigma": sigma,
    "n_ops": 1000,
    "n_samples": 1000,
    "key": jax.random.PRNGKey(42),
}

trainer = iqp.Trainer("Adam", loss, stepsize=0.01)
trainer.train(n_iters=200, loss_kwargs=loss_kwargs, turbo=10)

trained_params = trainer.final_params
plt.plot(trainer.losses)
plt.show()

######################################################################
# We can now try to see how well this generative IQP circuit resembles the ground truth. Since we are
# not working with a large number of qubits, we can sample from the circuit with the PennyLane
# machinery. We can then compare our trained and untrained samples with the ground truth through a
# histogram of the bitstring weights and evaluate the distributions.
#
samples_untrained = circuit.sample(params_init, shots=1000)
samples_trained = circuit.sample(trainer.final_params, shots=1000)

plt.hist(np.sum(samples_untrained, axis=1), bins=20, range=[0,20], alpha=0.5, label = 'untrained circuit')
plt.hist(np.sum(samples_trained, axis=1), bins=20, range=[0,20], alpha=0.5, label = 'trained circuit')
plt.hist(np.sum(X_train, axis=1), bins=20, range=[0,20], alpha=0.5, label = 'ground truth')
plt.xlabel('bitstring weight')
plt.ylabel('count')
plt.legend()
plt.show()

######################################################################
# As we can see the trained circuit closely resembles the ground truth distribution. Although we won't
# cover it in this demo, the package also contains tools to evaluate generative models and investigate
# model dropping via the Kernel Generalized Empirical Likelihood [#ravuri]_.
#
#
# Conclusion
# ---------------------------------
#
# As a final takeaway, IQPopt stands out as perhaps the only tool that enables researchers to analyze large-scale circuits with potential advantages using numerical methods. By doing so, it opens up opportunities
# to explore systems that are too intricate for traditional pen-and-paper calculations and, as a
# result, it has the potential to uncover insights that were previously inaccessible.
#
#
# References
# ----------
#
# .. [#marshall1]
#
#     Simon C. Marshall, Scott Aaronson, Vedran Dunjko
#     "Improved separation between quantum and classical computers for sampling and functional tasks"
#     `arXiv:410.20935 <https://arxiv.org/abs/2410.20935>`__, 2024.
#
# .. [#recio1]
#
#     Erik Recio, Joseph Bowles.
#     "IQPopt: Fast optimization of instantaneous quantum polynomial circuits in JAX".
#     `arXiv:2501.04776 <https://arxiv.org/abs/2501.04776>`__, 2025.
#
# .. [#nest]
#
#    M. Van den Nest.
#    "Simulating quantum computers with probabilistic methods"
#    `arXiv:0911.1624 <https://arxiv.org/abs/0911.1624>`__, 2010.
#
# .. [#gretton]
#
#    Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Schölkopf, Alexander Smola.
#    "A Kernel Two-Sample Test"
#    `http://jmlr.org/papers/v13/gretton12a.html <http://jmlr.org/papers/v13/gretton12a.html>`__, in Journal of Machine Learning Research 13.25, pp. 723-773, 2012.
#
# .. [#ravuri]
#
#    Suman Ravuri, Mélanie Rey, Shakir Mohamed, Marc Peter Deisenroth.
#    "Understanding Deep Generative Models with Generalized Empirical Likelihoods"
#    `arXiv:2306.09780 <https://arxiv.org/abs/2306.09780>`__, 2023.
#
