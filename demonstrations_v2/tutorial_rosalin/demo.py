r"""
Frugal shot optimization with Rosalin
=====================================

.. meta::
    :property="og:description": The Rosalin optimizer uses a measurement-frugal optimization strategy to minimize the
         number of times a quantum computer is accessed.
    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets/sphx_glr_tutorial_rosalin_002.png

.. related::

   tutorial_vqe A brief overview of VQE
   tutorial_quantum_natural_gradient Quantum natural gradient
   tutorial_doubly_stochastic Doubly stochastic gradient descent
   tutorial_rotoselect Quantum circuit structure learning

In this tutorial we investigate and implement the Rosalin (Random Operator Sampling for
Adaptive Learning with Individual Number of shots) from
Arrasmith et al. [#arrasmith2020]_. In this paper, a strategy
is introduced for reducing the number of shots required when optimizing variational quantum
algorithms, by both:

* Frugally adapting the number of shots used per parameter update, and
* Performing a weighted sampling of operators from the cost Hamiltonian.

.. note::

    The Rosalin optimizer is available in PennyLane via the
    :class:`~.pennylane.ShotAdaptiveOptimizer`.

Background
----------

While a large number of papers in variational quantum algorithms focus on the
choice of circuit ansatz, cost function, gradient computation, or initialization method,
the optimization strategy—an important component affecting both convergence time and
quantum resource dependence—is not as frequently considered. Instead, common
'out-of-the-box' classical optimization techniques, such as gradient-free
methods (COBLYA, Nelder-Mead), gradient-descent, and Hessian-free methods (L-BFGS) tend to be used.

However, for variational algorithms such as :doc:`VQE <demos/tutorial_vqe>`, which involve evaluating
a large number of non-commuting operators in the cost function, decreasing the number of
quantum evaluations required for convergence, while still minimizing statistical noise, can
be a delicate balance.

Recent work has highlighted that 'quantum-aware' optimization techniques
can lead to marked improvements when training variational quantum algorithms:

* :doc:`demos/tutorial_quantum_natural_gradient` descent by Stokes et al. [#stokes2019]_, which

  takes into account the quantum geometry during the gradient-descent update step.

* The work of Sweke et al. [#sweke2019]_, which shows
  that quantum gradient descent with a finite number of shots is equivalent to
  `stochastic gradient descent <https://en.wikipedia.org/wiki/Stochastic_gradient_descent>`_,
  and has guaranteed convergence. Furthermore, combining a finite number of shots with
  weighted sampling of the cost function terms leads to :doc:`demos/tutorial_doubly_stochastic`.

* The iCANS (individual Coupled Adaptive Number of Shots) optimization technique by
  Jonas Kuebler et al. [#kubler2020]_ adapts the number
  of shots measurements during training, by maximizing the expected gain per shot.

In this latest result by Arrasmith et al. [#arrasmith2020]_, the
idea of doubly stochastic gradient descent has been used to extend the iCANS optimizer,
resulting in faster convergence.

Over the course of this tutorial, we will explore their results; beginning first with a
demonstration of *weighted random sampling* of the cost Hamiltonian operators, before
combining this with the shot-frugal iCANS optimizer to perform doubly stochastic
Rosalin optimization.

Weighted random sampling
------------------------

Consider a Hamiltonian :math:`H` expanded as a weighted sum of operators :math:`h_i` that can
be directly measured:

.. math:: H = \sum_{i=1}^N c_i h_i.

Due to the linearity of expectation values, the expectation value of this Hamiltonian
can be expressed as the weighted sum of each individual term:

.. math:: \langle H\rangle = \sum_{i=1}^N c_i \langle h_i\rangle.

In the :doc:`doubly stochastic gradient descent demonstration <demos/tutorial_doubly_stochastic>`,
we estimated this expectation value by **uniformly sampling** a subset of the terms
at each optimization step, and evaluating each term by using the same finite number of shots
:math:`N.`

However, what happens if we use a weighted approach to determine how to distribute
our samples across the terms of the Hamiltonian? In **weighted random sampling** (WRS),
the number of shots used to determine the expectation value :math:`\langle h_i\rangle`
is a discrete random variable distributed according to a
`multinomial distribution <https://en.wikipedia.org/wiki/Multinomial_distribution>`__,

.. math:: S \sim \text{Multinomial}(p_i),

with event probabilities

.. math:: p_i = \frac{|c_i|}{\sum_i |c_i|}.

That is, the number of shots assigned to the measurement of the expectation value of the
:math:`i\text{th}` term of the Hamiltonian is drawn from a probability distribution
*proportional to the magnitude of its coefficient* :math:`c_i.`

To see this strategy in action, consider the Hamiltonian

.. math:: H = 2I\otimes X + 4 I\otimes Z  - X\otimes X + 5Y\otimes Y + 2 Z\otimes X.

We can solve for the ground state energy using the variational quantum eigensolver (VQE) algorithm.

First, let's import NumPy and PennyLane, and define our Hamiltonian.
"""
import pennylane as qml
qml.capture.enable()
from pennylane import numpy as np

# set the random seed
np.random.seed(4)

coeffs = [2, 4, -1, 5, 2]

obs = [
  qml.PauliX(1),
  qml.PauliZ(1),
  qml.PauliX(0) @ qml.PauliX(1),
  qml.PauliY(0) @ qml.PauliY(1),
  qml.PauliZ(0) @ qml.PauliZ(1)
]


##############################################################################
# We can now create our quantum device (let's use the ``default.qubit`` simulator).

num_layers = 2
num_wires = 2

# create a device that estimates expectation values using a finite number of shots
non_analytic_dev = qml.device("default.qubit", wires=num_wires, seed=432423)

# create a device that calculates exact expectation values
analytic_dev = qml.device("default.qubit", wires=num_wires)

##############################################################################
# Now, let's set the total number of shots, and determine the probability
# for sampling each Hamiltonian term.

total_shots = 8000
prob_shots = np.abs(coeffs) / np.sum(np.abs(coeffs))
print(prob_shots)

##############################################################################
# We can now use SciPy to create our multinomial distributed random variable
# :math:`S,` using the number of trials (total shot number) and probability values:

from scipy.stats import multinomial

si = multinomial(n=total_shots, p=prob_shots)

##############################################################################
# Sampling from this distribution will provide the number of shots used to
# sample each term in the Hamiltonian:

samples = si.rvs()[0]
print(samples)
print(sum(samples))

##############################################################################
# As expected, if we sum the sampled shots per term, we recover the total number of shots.
#
# Let's now create our cost function. Recall that the cost function must do the
# following:
#
# 1. It must sample from the multinomial distribution we created above,
#    to determine the number of shots :math:`s_i` to use to estimate the expectation
#    value of the ith Hamiltonian term.
#
# 2. It then must estimate the expectation value :math:`\langle h_i\rangle`
#    by creating the required QNode. For our ansatz, we'll use the 
#    :class:`~.pennylane.templates.layers.StronglyEntanglingLayers`.
#
# 3. And, last but not least, estimate the expectation value
#    :math:`\langle H\rangle = \sum_i c_i\langle h_i\rangle.`
#

from pennylane.templates.layers import StronglyEntanglingLayers


@qml.set_shots(100)
@qml.qnode(non_analytic_dev, diff_method="parameter-shift", interface="autograd")
def qnode(weights, observable):
    StronglyEntanglingLayers(weights, wires=non_analytic_dev.wires)
    return qml.expval(observable)

def cost(params):
    # sample from the multinomial distribution
    shots_per_term = si.rvs()[0]

    result = 0

    for o, c, s in zip(obs, coeffs, shots_per_term):
        # evaluate the QNode corresponding to
        # the Hamiltonian term, and add it on to our running sum
        result += c * qml.set_shots(qnode, shots=int(s))(params, o)

    return result


##############################################################################
# Evaluating our cost function with some initial parameters, we can test out
# that our cost function evaluates correctly.

param_shape = StronglyEntanglingLayers.shape(n_layers=num_layers, n_wires=num_wires)
init_params = np.random.uniform(low=0.0, high=2*np.pi, size=param_shape, requires_grad=True)
print(cost(init_params))


##############################################################################
# Performing the optimization, with the number of shots randomly
# determined at each optimization step:

opt = qml.AdamOptimizer(0.05)
params = init_params

cost_wrs = []
shots_wrs = []

for i in range(100):
    params, _cost = opt.step_and_cost(cost, params)
    cost_wrs.append(_cost)
    shots_wrs.append(total_shots*i)
    print("Step {}: cost = {} shots used = {}".format(i, cost_wrs[-1], shots_wrs[-1]))

##############################################################################
# Let's compare this against an optimization not using weighted random sampling.
# Here, we will split the 8000 total shots evenly across all Hamiltonian terms,
# also known as *uniform deterministic sampling*.

@qml.set_shots(100)
@qml.qnode(non_analytic_dev, diff_method="parameter-shift", interface="autograd")
def qnode(weights, obs):
    StronglyEntanglingLayers(weights, wires=non_analytic_dev.wires)
    return qml.expval(obs)

def cost(params):
    shots_per_term = int(total_shots / len(coeffs))

    result = 0

    for o, c in zip(obs, coeffs):

        # evaluate the QNode corresponding to
        # the Hamiltonian term, and add it on to our running sum
        result += c * qml.set_shots(qnode, shots=shots_per_term)(params, o)

    return result

opt = qml.AdamOptimizer(0.05)
params = init_params

cost_adam = []
shots_adam = []

for i in range(100):
    params, _cost = opt.step_and_cost(cost, params)
    cost_adam.append(_cost)
    shots_adam.append(total_shots*i)
    print("Step {}: cost = {} shots used = {}".format(i, cost_adam[-1], shots_adam[-1]))

##############################################################################
# Comparing these two techniques:

from matplotlib import pyplot as plt

plt.style.use("seaborn-v0_8")
plt.plot(shots_wrs, cost_wrs, "b", label="Adam WRS")
plt.plot(shots_adam, cost_adam, "g", label="Adam")

plt.ylabel("Cost function value")
plt.xlabel("Number of shots")
plt.legend()
plt.show()

##############################################################################
# We can see that weighted random sampling performs just as well as the uniform
# deterministic sampling. However, weighted random sampling begins to show a
# non-negligible improvement over deterministic sampling for large Hamiltonians
# with highly non-uniform coefficients. For example, see Fig (3) and (4) of
# Arrasmith et al. [#arrasmith2020]_, comparing weighted random sampling VQE optimization
# for both :math:`\text{H}_2` and :math:`\text{LiH}` molecules.
#
# .. note::
#
#     While not covered here, another approach that could be taken is
#     *weighted deterministic sampling*. Here, the number of shots is distributed
#     across terms as per
#
#     .. math:: s_i = \left\lfloor N \frac{|c_i|}{\sum_i |c_i|}\right\rfloor,
#
#     where :math:`N` is the total number of shots.
#

##############################################################################
# Rosalin: Frugal shot optimization
# ---------------------------------
#
# We can see above that both methods optimize fairly well; weighted random
# sampling converges just as well as evenly distributing the shots across
# all Hamiltonian terms. However, deterministic shot distribution approaches
# will always have a minimum shot value required per expectation value, as below
# this threshold they become biased estimators. This is not the case with random
# sampling; as we saw in the
# :doc:`doubly stochastic gradient descent demonstration <demos/tutorial_doubly_stochastic>`,
# the introduction of randomness allows for as little
# as a single shot per expectation term, while still remaining an unbiased estimator.
#
# Using this insight, Arrasmith et al. [#arrasmith2020]_ modified the iCANS frugal
# shot-optimization technique [#kubler2020]_ to include weighted random sampling, making it
# 'doubly stochastic'.
#
# iCANS optimizer
# ~~~~~~~~~~~~~~~
#
# Two variants of the iCANS optimizer were introduced in Kübler et al., iCANS1 and iCANS2.
# The iCANS1 optimizer, on which Rosalin is based, frugally distributes a shot budget
# across the partial derivatives of each parameter, which are computed using the
# :doc:`parameter-shift rule </glossary/quantum_gradient>`. It works roughly as follows:
#
# 1. The initial step of the optimizer is performed with some specified minimum
#    number of shots, :math:`s_{min},` for all partial derivatives.
#
# 2. The parameter-shift rule is then used to estimate the gradient :math:`g_i`
#    for each parameter :math:`\theta_i,` parameters, as well as the *variances*
#    :math:`v_i` of the estimated gradients.
#
# 3. Gradient descent is performed for each parameter :math:`\theta_i,` using
#    the pre-defined learning rate :math:`\alpha` and the gradient information :math:`g_i:`
#
#    .. math:: \theta_i = \theta_i - \alpha g_i.
#
# 4. The improvement in the cost function per shot, for a specific parameter value,
#    is then calculated via
#
#    .. math::
#
#        \gamma_i = \frac{1}{s_i} \left[ \left(\alpha - \frac{1}{2} L\alpha^2\right)
#                    g_i^2 - \frac{L\alpha^2}{2s_i}v_i \right],
#
#    where:
#
#    * :math:`L \leq \sum_i|c_i|` is the bound on the `Lipschitz constant
#      <https://en.wikipedia.org/wiki/Lipschitz_continuity>`__ of the variational quantum algorithm objective function,
#
#    * :math:`c_i` are the coefficients of the Hamiltonian, and
#
#    * :math:`\alpha` is the learning rate, and *must* be bound such that :math:`\alpha < 2/L`
#      for the above expression to hold.
#
# 5. Finally, the new values of :math:`s_i` (shots for partial derivative of parameter
#    :math:`\theta_i`) is given by:
#
#    .. math::
#
#        s_i = \frac{2L\alpha}{2-L\alpha}\left(\frac{v_i}{g_i^2}\right)\propto
#              \frac{v_i}{g_i^2}.
#
# In addition to the above, to counteract the presence of noise in the system, a
# running average of :math:`g_i` and :math:`s_i` (:math:`\chi_i` and :math:`\xi_i` respectively)
# are used when computing :math:`\gamma_i` and :math:`s_i.`
#
# .. note::
#
#     In classical machine learning, the Lipschitz constant of the cost function is generally
#     unknown. However, for a variational quantum algorithm with cost of the form
#     :math:`f(x) = \langle \psi(x) | \hat{H} |\psi(x)\rangle,`
#     an upper bound on the Lipschitz constant is given by :math:`L < \sum_i|c_i|,`
#     where :math:`c_i` are the coefficients of :math:`\hat{H}` when decomposed
#     into a linear combination of Pauli-operator tensor products.
#
# Rosalin implementation
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Let's now modify iCANS above to incorporate weighted random sampling of Hamiltonian
# terms — the Rosalin frugal shot optimizer.
#
# Rosalin takes several hyper-parameters:
#
# * ``min_shots``: the minimum number of shots used to estimate the expectations
#   of each term in the Hamiltonian. Note that this must be larger than 2 for the variance
#   of the gradients to be computed.
#
# * ``mu``: The running average constant :math:`\mu\in[0, 1].` Used to control how quickly the
#   number of shots recommended for each gradient component changes.
#
# * ``b``: Regularization bias. The bias should be kept small, but non-zero.
#
# * ``lr``: The learning rate. Recall from above that the learning rate *must* be such
#   that :math:`\alpha < 2/L = 2/\sum_i|c_i|.`
#
# Since the Rosalin optimizer has a state that must be preserved between optimization steps,
# let's use a class to create our optimizer.
#

class Rosalin:

    def __init__(self, obs, coeffs, min_shots, mu=0.99, b=1e-6, lr=0.07):
        self.obs = obs
        self.coeffs = coeffs

        self.lipschitz = np.sum(np.abs(coeffs))

        if lr > 2 / self.lipschitz:
            raise ValueError("The learning rate must be less than ", 2 / self.lipschitz)

        # hyperparameters
        self.min_shots = min_shots
        self.mu = mu  # running average constant
        self.b = b    # regularization bias
        self.lr = lr  # learning rate

        # keep track of the total number of shots used
        self.shots_used = 0
        # total number of iterations
        self.k = 0
        # Number of shots per parameter
        self.s = np.zeros_like(params, dtype=np.float64) + min_shots

        # Running average of the parameter gradients
        self.chi = None
        # Running average of the variance of the parameter gradients
        self.xi = None

    def estimate_hamiltonian(self, params, shots):
        """Returns an array containing length ``shots`` single-shot estimates
        of the Hamiltonian. The shots are distributed randomly over
        the terms in the Hamiltonian, as per a Multinomial distribution.

        Since we are performing single-shot estimates, the QNodes must be
        set to 'sample' mode.
        """
        # note that convergence depends on seed for random number generation
        rosalin_device = qml.device("default.qubit", wires=num_wires, seed=93754352)

        # determine the shot probability per term
        prob_shots = np.abs(coeffs) / np.sum(np.abs(coeffs))

        # construct the multinomial distribution, and sample
        # from it to determine how many shots to apply per term
        si = multinomial(n=shots, p=prob_shots)
        shots_per_term = si.rvs()[0]

        results = []

        @qml.set_shots(100)
        @qml.qnode(rosalin_device, diff_method="parameter-shift", interface="autograd")
        def qnode(weights, observable):
            StronglyEntanglingLayers(weights, wires=rosalin_device.wires)
            return qml.sample(observable)

        for o, c, p, s in zip(self.obs, self.coeffs, prob_shots, shots_per_term):

            # if the number of shots is 0, do nothing
            if s == 0:
                continue

            # evaluate the QNode corresponding to
            # the Hamiltonian term
            res = qml.set_shots(qnode, shots=int(s))(params, o)

            # Note that, unlike above, we divide each term by the
            # probability per shot. This is because we are sampling one at a time.
            results.append(c * res / p)

        return np.concatenate(results)

    def evaluate_grad_var(self, i, params, shots):
        """Evaluate the gradient, as well as the variance in the gradient,
        for the ith parameter in params, using the parameter-shift rule.
        """
        shift = np.zeros_like(params)
        shift[i] = np.pi / 2

        shift_forward = self.estimate_hamiltonian(params + shift, shots)
        shift_backward = self.estimate_hamiltonian(params - shift, shots)

        g = np.mean(shift_forward - shift_backward) / 2
        s = np.var((shift_forward - shift_backward) / 2, ddof=1)

        return g, s

    def step(self, params):
        """Perform a single step of the Rosalin optimizer."""
        # keep track of the number of shots run
        self.shots_used += int(2 * np.sum(self.s))

        # compute the gradient, as well as the variance in the gradient,
        # using the number of shots determined by the array s.
        grad = []
        S = []

        p_ind = list(np.ndindex(*params.shape))

        for l in p_ind:
            # loop through each parameter, performing
            # the parameter-shift rule
            g_, s_ = self.evaluate_grad_var(l, params, self.s[l])
            grad.append(g_)
            S.append(s_)

        grad = np.reshape(np.stack(grad), params.shape)
        S = np.reshape(np.stack(S), params.shape)

        # gradient descent update
        params = params - self.lr * grad

        if self.xi is None:
            self.chi = np.zeros_like(params, dtype=np.float64)
            self.xi = np.zeros_like(params, dtype=np.float64)

        # running average of the gradient variance
        self.xi = self.mu * self.xi + (1 - self.mu) * S
        xi = self.xi / (1 - self.mu ** (self.k + 1))

        # running average of the gradient
        self.chi = self.mu * self.chi + (1 - self.mu) * grad
        chi = self.chi / (1 - self.mu ** (self.k + 1))

        # determine the new optimum shots distribution for the next
        # iteration of the optimizer
        s = np.ceil(
            (2 * self.lipschitz * self.lr * xi)
            / ((2 - self.lipschitz * self.lr) * (chi ** 2 + self.b * (self.mu ** self.k)))
        )

        # apply an upper and lower bound on the new shot distributions,
        # to avoid the number of shots reducing below min(2, min_shots),
        # or growing too significantly.
        gamma = (
            (self.lr - self.lipschitz * self.lr ** 2 / 2) * chi ** 2
            - xi * self.lipschitz * self.lr ** 2 / (2 * s)
        ) / s

        argmax_gamma = np.unravel_index(np.argmax(gamma), gamma.shape)
        smax = s[argmax_gamma]
        self.s = np.clip(s, min(2, self.min_shots), max(2, smax))

        self.k += 1
        return params


##############################################################################
# Rosalin optimization
# ~~~~~~~~~~~~~~~~~~~~
#
# We are now ready to use our Rosalin optimizer to optimize the initial VQE problem. But first let's
# also create a separate cost function using an 'exact' quantum device, so that we can keep track of the
# *exact* cost function value at each iteration.

@qml.qnode(analytic_dev, interface="autograd")
def cost_analytic(weights):
    StronglyEntanglingLayers(weights, wires=analytic_dev.wires)
    return qml.expval(qml.Hamiltonian(coeffs, obs))

##############################################################################
# Creating the optimizer and beginning the optimization:


opt = Rosalin(obs, coeffs, min_shots=10)
params = init_params

cost_rosalin = [cost_analytic(params)]
shots_rosalin = [0]

for i in range(60):
    params = opt.step(params)
    cost_rosalin.append(cost_analytic(params))
    shots_rosalin.append(opt.shots_used)
    print(f"Step {i}: cost = {cost_rosalin[-1]}, shots_used = {shots_rosalin[-1]}")


##############################################################################
# Let's compare this to a standard Adam optimization. Using 100 shots per quantum
# evaluation, for each update step there are 2 quantum evaluations per parameter.

adam_shots_per_eval = 100
adam_shots_per_step = 2 * adam_shots_per_eval * len(params.flatten())
print(adam_shots_per_step)

##############################################################################
# Thus, Adam is using 2400 shots per update step.

params = init_params
opt = qml.AdamOptimizer(0.07)

adam_dev = qml.device('default.qubit', seed=595905)

@qml.set_shots(adam_shots_per_eval)
@qml.qnode(adam_dev, diff_method="parameter-shift", interface="autograd")
def cost(weights):
    StronglyEntanglingLayers(weights, wires=non_analytic_dev.wires)
    return qml.expval(qml.Hamiltonian(coeffs, obs))

cost_adam = [cost_analytic(params)]
shots_adam = [0]

for i in range(100):
    params = opt.step(cost, params)
    cost_adam.append(cost_analytic(params))
    shots_adam.append(adam_shots_per_step * (i + 1))
    print("Step {}: cost = {} shots_used = {}".format(i, cost_adam[-1], shots_adam[-1]))

##############################################################################
# Plotting both experiments:

plt.style.use("seaborn-v0_8")
plt.plot(shots_rosalin, cost_rosalin, "b", label="Rosalin")
plt.plot(shots_adam, cost_adam, "g", label="Adam")

plt.ylabel("Cost function value")
plt.xlabel("Number of shots")
plt.legend()
plt.xlim(0, 300000)
plt.show()

##############################################################################
# The Rosalin optimizer performs significantly better than the Adam optimizer,
# approaching the ground state energy of the Hamiltonian with strikingly
# fewer shots.
#
# While beyond the scope of this demonstration, the Rosalin optimizer can be
# modified in various other ways; for instance, by incorporating *weighted hybrid
# sampling* (which distributes some shots deterministically, with the remainder
# done randomly), or by adapting the variant iCANS2 optimizer. Download
# this demonstration from the sidebar 👉 and give it a go! ⚛️


##############################################################################
# References
# ----------
#
# .. [#arrasmith2020]
#
#     Andrew Arrasmith, Lukasz Cincio, Rolando D. Somma, and Patrick J. Coles. "Operator Sampling
#     for Shot-frugal Optimization in Variational Algorithms." `arXiv:2004.06252
#     <https://arxiv.org/abs/2004.06252>`__ (2020).
#
# .. [#stokes2019]
#
#     James Stokes, Josh Izaac, Nathan Killoran, and Giuseppe Carleo. "Quantum Natural Gradient."
#     `arXiv:1909.02108 <https://arxiv.org/abs/1909.02108>`__ (2019).
#
# .. [#sweke2019]
#
#     Ryan Sweke, Frederik Wilde, Johannes Jakob Meyer, Maria Schuld, Paul K. Fährmann, Barthélémy
#     Meynard-Piganeau, and Jens Eisert. "Stochastic gradient descent for hybrid quantum-classical
#     optimization." `arXiv:1910.01155 <https://arxiv.org/abs/1910.01155>`__ (2019).
#
# .. [#kubler2020]
#
#     Jonas M. Kübler, Andrew Arrasmith, Lukasz Cincio, and Patrick J. Coles. "An Adaptive Optimizer
#     for Measurement-Frugal Variational Algorithms." `Quantum 4, 263
#     <https://quantum-journal.org/papers/q-2020-05-11-263/>`__ (2020).
#
