r"""
Is quantum computing useful before fault tolerance?
===================================================

.. meta::
    :property="og:description": Evidence for the utility of quantum computing before fault tolerance
    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets/thumbnail_tutorial_mitigation_advantage.png

.. related::

    tutorial_error_mitigation Error mitigation with Mitiq and PennyLane
    tutorial_diffable-mitigation Differentiable quantum error mitigation
    tutorial_noisy_circuits Noisy circuits
    gbs Quantum advantage with Gaussian Boson Sampling

Can we use contemporary quantum computers for tasks that are both useful *and* hard to classically simulate?
A recent `Nature paper <https://www.nature.com/articles/s41586-023-06096-3>`__ from the team at IBM claims that we can! See how they managed to faithfully estimate expectation
values of reasonably large and reasonably deep quantum circuits using an exciting new :doc:`zero noise extrapolation <demos/tutorial_diffable-mitigation>`
technique for error mitigation in this demo.

Introduction
------------

We know that quantum computers can do things that classical computers cannot.
But quantum algorithms like Shor's algorithm necessitate fault tolerance via 
error correction, which is not yet feasible with currently available machines. One highly debated 
question in the field is whether or not noisy devices we have access to `right now` are already 
useful or can outperform a classical computer for certain tasks. For the latter point, 
demonstrations of quantum computational advantage have been put forward with the 
`2019 Sycamore <https://www.nature.com/articles/s41586-019-1666-5>`_
(Google), the `2020 Jiuzhang <https://www.science.org/doi/10.1126/science.abe8770>`_ (Chinese academy of science)
and the `2022 Borealis <https://www.nature.com/articles/s41586-022-04725-x>`_ (Xanadu) experiments. 
These demonstrations, however, are only of limited practical utility.

A new quest has been set out 
for 'practical' quantum computational advantage. That is, a 'useful' application for which a quantum computer 
outperforms the best known classical methods. In the 
new article [#ibm]_ by a team of scientists at IBM, a case is made that with their latest device 
`ibm_kyiv <https://quantum-computing.ibm.com/services/resources?system=ibm_kyiv>`_
comprising 127 qubits and record-breaking coherence times, they can faithfully simulate the time 
dynamics of a complex quantum many-body system. One of the key achievements of the paper is the 
successful application of error mitigation on a large system (that is making use of a learned noise model [#PEC]_), 
and demonstrating that it can yield 
faithful results even in very noisy scenarios with reasonably deep circuits. 

Problem setting
---------------
Before we go into the details of the error mitigation methods, let us briefly summarize the problem 
setting. The authors of [#ibm]_ are concerned with simulating the time dynamics of the 2D transverse field Ising model  

.. math:: H = -J \sum_{\langle qp \rangle}Z_q Z_p + h \sum_q X_q,

with nearest neighbor interactions (indicated by :math:`\langle qp \rangle`) matching the topology of their 127-qubit 
`ibm_kyiv <https://quantum-computing.ibm.com/services/resources?system=ibm_kyiv>`_ device.
The system is described by the positive coupling strength :math:`J` and transverse field `h.`
The time evolution is approximated by trotterization of the time evolution operator

.. math:: U(T) \approx \left(\prod_{\langle qp \rangle} e^{i \delta t J Z_q Z_p} \prod_{q} e^{-i \delta t h X_q} \right)^{\frac{T}{\delta t}}

for an evolution time :math:`T` and a Trotter step size :math:`\delta t.` That means the circuit of concern here is a 
series of consecutive :math:`\text{RZZ}(\theta_J)` and :math:`\text{RX}(\theta_h)` rotations. The corresponding 
angles are related to the physical parameters via :math:`\theta_J = -2J \delta t` and :math:`\theta_h = 2h \delta t```.` 
From here on, we are going to focus just on the values of :math:`\theta_h` and keep :math:`\theta_J=-\pi/2` fixed 
(in the paper, this is due to the simplification this introduces in the decomposition of the :math:`\text{RZZ}` gate in 
terms of the required CNOT gates).

Noisy simulation of the circuits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The complexity of the classical simulation varies with the parameter :math:`\theta_h.` For the extrema 
:math:`\theta_h=0` and :math:`\theta_h=\pi/2,` the system becomes trivially solvable. We interpolate between 
those extrema and show the final value of a single weight observable :math:`\langle Z_4\rangle` as is done in [#ibm]_.

To reproduce the key ingredients of [#ibm]_, we are going to simulate a scaled down version of the real system using PennyLane. Instead of 127 qubits, we will use only 9, placed on a :math:`3 \times 3` grid with
nearest neighbor interactions.
We start by setting up the circuits for the time evolution and a noise model consisting of
:class:`~pennylane.DepolarizingChannel` applied to each gate the circuit executes. Physically, this corresponds to applying either of the 
single qubit Pauli gates :math:`\{X, Y, Z\}` with probability :math:`p/3` after each gate in the circuit. In simulation, we can simply look
at the classical mixtures introduced by the Kraus operators of the noise channel. That is why we need to use the mixed state simulator.
For more information see e.g. our :doc:`demo on simulating noisy circuits <demos/tutorial_noisy_circuits>`.
"""
import pennylane as qml
qml.capture.enable()
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

n_wires = 9
# Describe noise
noise_gate = qml.DepolarizingChannel
p = 0.005

# Load devices
dev_ideal = qml.device("default.mixed", wires=n_wires)
dev_noisy = qml.noise.insert(dev_ideal, noise_gate, p, position="all")

# 3x3 grid with nearest neighbors
connections = [(0, 1), (1, 2),
               (3, 4), (4, 5),
               (6, 7), (7, 8),
               (0, 3), (3, 6),
               (1, 4), (4, 7),
               (2, 5), (5, 8)]

def time_evolution(theta_h, n_layers = 10, obs = qml.PauliZ(4)):
    for _ in range(n_layers):
        for i, j in connections:
            qml.IsingZZ(-jnp.pi/2, wires=(i, j))
        for i in range(n_wires):
            qml.RX(theta_h, wires=i)
    return qml.expval(obs)

qnode_ideal = qml.QNode(time_evolution, dev_ideal, interface="jax")
qnode_noisy = qml.QNode(time_evolution, dev_noisy, interface="jax")

##############################################################################
# We can now simulate the final expectation value with and without noise. Note that the ``IsingZZ`` gate is not natively
# supported by the ``default.mixed`` device and will be decomposed into a supported gate set. The noise channels will be
# inserted after all gates in the final decomposed circuit. We use ``jax.vmap`` to vectorize and speed up the execution
# for different values of :math:`\theta_h.`

thetas = jnp.linspace(0, jnp.pi/2, 50)

res_ideal = jax.vmap(qnode_ideal)(thetas)
res_noisy = jax.vmap(qnode_noisy)(thetas)

plt.plot(thetas, res_ideal, label="exact")
plt.plot(thetas, res_noisy, label="noisy")
plt.xticks([0, jnp.pi/8, jnp.pi/4, 3*jnp.pi/8, jnp.pi/2], ["0", "$\\pi$/8", "$\\pi/4$", "$3\\pi/4$", "$\\pi/2$"])
plt.xlabel("$\\theta_h$")
plt.ylabel("$\\langle Z_4 \\rangle$")
plt.legend()
plt.show()



##############################################################################
# We see that the fidelity of the result is decreased by the noise. Next, we show how this noise can be effectively mitigated.
# 
# 
# Error mitigation via zero noise extrapolation
# ---------------------------------------------
# 
# :doc:`Error mitigation <demos/tutorial_error_mitigation>` is the process of retrieving more accurate information via classical post-processing
# of noisy quantum executions. The authors in [#ibm]_ employ zero noise extrapolation (ZNE), which serves as
# a biased estimator of expectation values. The idea of ZNE is fairly straightforward: Imagine we want to
# obtain the exact quantum function :math:`f` that estimates an expectation value under noiseless evolution.
# However, we only have access to a noisy version :math:`f^{⚡}.` Now suppose we can controllably increase 
# the noise present in terms of some noise gain parameter :math:`G.` Here, :math:`G=1` corresponds to
# the default noise present in the device. In ZNE, we evaluate :math:`f^{⚡}` at increasing values of :math:`G,`
# from which we can extrapolate back to zero noise :math:`G=0` via a suitable curve fit. 
# 
# In order to perform ZNE, we need a control knob that increases the noise of our circuit execution.
# One such method is described in our
# :doc:`demo on differentiable error mitigation <demos/tutorial_diffable-mitigation>` using circuit folding.
# 
# Noise-aware ZNE
# ~~~~~~~~~~~~~~~
#
# In [#ibm]_, the authors use a more sophisticated control knob to artificially increase the noise. They first learn the parameters
# of an assumed noise model (in their case a Pauli Lindblad model) of their device [#PEC]_. Ideally, one would counteract those effects by
# probabilistically inverting the noise action (probabilistic error cancellation [#PEC]_). However, this comes with an increased sampling overhead, which is not feasible for the size
# of their problem. So instead, they use the knowledge of the learned noise model to artificially add extra noise and perform ZNE.
# 
# The noise model of our simulation is relatively simple and we have full control over it. This means that we can simply attenuate the noise of 
# our model by an appropriate gain factor. Here, :math:`G=(1, 1.2, 1.6)` in accordance with [#ibm]_. In order to do this in PennyLane, we simply
# set up two new noisy devices with the appropriately attenuated noise parameters.

dev_noisy1 = qml.noise.insert(dev_ideal, noise_gate, p*1.2, position="all")
dev_noisy2 = qml.noise.insert(dev_ideal, noise_gate, p*1.6, position="all")

qnode_noisy1 = qml.QNode(time_evolution, dev_noisy1, interface="jax")
qnode_noisy2 = qml.QNode(time_evolution, dev_noisy2, interface="jax")

res_noisy1 = jax.vmap(qnode_noisy1)(thetas)
res_noisy2 = jax.vmap(qnode_noisy2)(thetas)

##############################################################################
# We can take these results and simply extrapolate back to :math:`G=0` with a polynomial fit.
# We can visualize this by plotting the noisy, exact and extrapolated results.

Gs = jnp.array([1., 1.2, 1.6])
y = jnp.array([res_noisy[0], res_noisy1[0], res_noisy2[0]])
coeff = jnp.polyfit(Gs, y, 2)
x = jnp.linspace(0, 1.6, 100)

plt.plot(x, jnp.polyval(coeff, x), label="fit")
plt.plot(Gs, y, "x", label="noisy results")
plt.plot([0], res_ideal[0], "X", label="exact result")
plt.xlabel("noise gain G")
plt.ylabel("$\\langle Z_4 \\rangle$")
plt.legend()
plt.show()

##############################################################################
# We now repeat this procedure for all values of :math:`\theta_h` and see how the results are much improved.
# We can use :func:`~pennylane.noise.richardson_extrapolate` that performs a polynomial fit of a degree matching the input data size.

res_mitigated = [qml.noise.richardson_extrapolate(Gs, [res_noisy[i], res_noisy1[i], res_noisy2[i]]) for i in range(len(res_ideal))]

plt.plot(thetas, res_ideal, label="exact")
plt.plot(thetas, res_mitigated, label="mitigated")
plt.plot(thetas, res_noisy, label="noisy")
plt.xticks([0, jnp.pi/8, jnp.pi/4, 3*jnp.pi/8, jnp.pi/2], ["0", "$\\pi$/8", "$\\pi/4$", "$3\\pi/4$", "$\\pi/2$"])
plt.xlabel("$\\theta_h$")
plt.ylabel("$\\langle Z_4 \\rangle$")
plt.legend()
plt.show()

##############################################################################
# The big achievement in [#ibm]_ is that they managed to showcase the feasibility of this approach on a large scale experimentally for their device.
# This is really good news, as it has not been clear whether or not noise mitigation can be successfully employed on larger scales. The key ingredient
# is the noise-aware attenuation, which allows for more realistic and finer extrapolation at low resource overhead.
# 
# 
# Comparison with classical methods
# ---------------------------------
# The authors of [#ibm]_ compare their experimental results with classical methods. For this, they consider three 
# scenarios of different classical complexity.
#
# For :math:`\theta_h=-\pi/2` (case 1) the dynamics become trivial with just a global phase factor introduced, such that 
# starting from the initial state :math:`|0\rangle^{\otimes 127},` the expectation values :math:`\langle Z_q \rangle` are trivially 
# one at all times. This serves as an anchor point of orientation. Varying :math:`\theta_h` (case 2) then increases the 
# classical simulation complexity. For the circuits chosen, it is still possible to simulate the dynamical 
# expectation values of local observables by taking into account their light-cone in the evolution with reduced depth 
# (note that these are not full state vector evolutions but rather just directly looking at the dynamical expectation 
# values of interest). In the third and most complex case, the circuit is altered such that the light-cone trick from 
# before does not work anymore.
# 
# One of the points of the paper is to compare the experiments with sophisticated classical simulation methods. 
# The authors chose tensor networks, in particular matrix product states (MPS) and isometric tensor network states 
# (isoTNS) for simulation. MPS are native to one dimensional topologies, but are often still employed for two 
# dimensional systems as is the case here. The justification for that is their lower computational and algorithmic
# complexity as well as the opportunity to draw from
# decades of algorithmic development and optimization. Better suited to the given problem are isoTNS, which are 
# restrictions of projected entangled pair states (PEPS) with some simplifications reducing the high computational 
# and algorithmic complexity, at the cost of more approximation errors.
# 
# In both cases, the so-called bond-dimension :math:`\chi,` a hyperparameter chosen by the user, directly determines 
# the bipartite entanglement entropy these states can capture. It is known that due to the area law of entanglement, 
# many ground states of relevant physical system can be faithfully approximated with suitably chosen tensor network states 
# with finite bond dimension. However, that is generally not the case for time dynamics as the entanglement entropy 
# grows linearly and the area law no longer holds. Therefore, the employed tensor network methods are doomed for 
# most dynamical simulations, as is showcased in the paper.
# 
# `It can be argued <https://twitter.com/gppcarleo/status/1669251392156860418>`_ that there are better suited 
# classical algorithms for these kind of dynamical simulations, 
# `with neural quantum states being one of them <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.100503>`_.
# Further, tensor network methods in two and higher dimensions are extremely difficult to implement. The employed methods
# are not well suited for the problem and do not grasp the full breadth of possibilities of
# classical simulation methods. We are curious to see what experts in the field will come up
# with to showcase faithful classical simulations of these circuits.
#
# 
#
#
# References
# ----------
#
# .. [#ibm]
#
#     Youngseok Kim, Andrew Eddins, Sajant Anand, Ken Xuan Wei, Ewout van den Berg, Sami Rosenblatt, Hasan Nayfeh, Yantao Wu, Michael Zaletel, Kristan Temme & Abhinav Kandala 
#     "Evidence for the utility of quantum computing before fault tolerance"
#     `Nature 618, 500–505 <https://www.nature.com/articles/s41586-023-06096-3>`__, 2023.
#
# .. [#PEC]
#
#     Ewout van den Berg, Zlatko K. Minev, Abhinav Kandala, Kristan Temme
#     "Probabilistic error cancellation with sparse Pauli-Lindblad models on noisy quantum processors"
#     `arXiv:2201.09866 <https://arxiv.org/abs/2201.09866>`__, 2022.
#
