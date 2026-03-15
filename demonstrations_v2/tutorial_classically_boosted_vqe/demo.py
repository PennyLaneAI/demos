r"""
Classically-boosted variational quantum eigensolver
===================================================

.. meta::
    :property="og:description": Learn how to implement classically-boosted VQE in PennyLane.
    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets/CB_VQE.png

.. related::

    tutorial_quantum_chemistry Building molecular Hamiltonians
    tutorial_vqe Variational Quantum Eigensolver


One of the most important applications of quantum computers is expected
to be the computation of ground-state energies of complicated molecules
and materials. Even though there are already some solid proposals on how
to tackle these problems when fault-tolerant quantum computation comes
into play, we currently live in the `noisy intermediate-scale quantum
(NISQ) <https://en.wikipedia.org/wiki/Noisy_intermediate-scale_quantum_era>`__
era, meaning that we can only access noisy and limited devices.
That is why a large part of the current research on quantum algorithms is
focusing on what can be done with few resources. In particular, most
proposals rely on variational quantum algorithms (VQA), which are
optimized classically and adapt to the limitations of the quantum
devices. For the specific problem of computing ground-state energies,
the paradigmatic algorithm is the `Variational Quantum Eigensolver
(VQE) <https://en.wikipedia.org/wiki/Variational_quantum_eigensolver>`__ algorithm.


Although VQE is intended to run on NISQ devices, it is nonetheless
sensitive to noise. This is particularly problematic when applying VQE to complicated molecules which requires a large number of gates. 
As a consequence, several modifications to the
original VQE algorithm have been proposed. These variants are usually
intended to improve the algorithm’s performance on NISQ-era devices.

In this demo, we will go through one of these proposals step-by-step: the
Classically-Boosted Variational Quantum Eigensolver (CB-VQE) [#Radin2021]_.
Implementing CB-VQE reduces the number of measurements required to obtain the
ground-state energy with a certain precision. This is done by making use
of classical states, which in this context are product states that can be
written as a single `Slater determinant  <https://en.wikipedia.org/wiki/Slater_determinant>`__
and that already contain some information about the ground-state of the problem.
Their structure allows for efficient classical computation of expectation values. 
An example of such classical state would be the `Hartree-Fock state <https://en.wikipedia.org/wiki/Hartree%E2%80%93Fock_method>`__,
in which the electrons occupy the molecular orbitals with the lowest energy.

.. figure:: ../_static/demonstration_assets/classically_boosted_vqe/CB_VQE.png
    :align: center
    :width: 50%

We will restrict ourselves to the :math:`H_2` molecule for
the sake of simplicity. First, we will give a short introduction to how
to perform standard VQE for the molecule of interest. For more details,
we recommend the tutorial :doc:`demos/tutorial_vqe` to learn
how to implement VQE for molecules step by step. Then, we will implement
the CB-VQE algorithm for the specific case in which we rely only on one
classical state⁠—that being the Hartree-Fock state. Finally, we will
discuss the number of measurements needed to obtain a certain
error-threshold by comparing the two methods.

Let’s get started!

"""

######################################################################
# Prerequisites: Standard VQE
# ---------------------------
#
# If you are not already familiar with the VQE family of algorithms and
# wish to see how one can apply it to the :math:`H_2` molecule, feel free to
# work through the aforementioned demo before reading this section.
# Here, we will only briefly review the main idea behind standard VQE
# and highlight the important concepts in connection with CB-VQE.
#
# Given a Hamiltonian :math:`H,` the main goal of VQE is to find the ground state energy of a system governed by the Schrödinger
# equation
#
# .. math:: H \vert \phi \rangle = E  \vert \phi \rangle.
#
# This corresponds to the problem of diagonalizing the Hamiltonian and
# finding the smallest eigenvalue. Alternatively, one can formulate the
# problem using the `variational principle <https://en.wikipedia.org/wiki/Variational_principle>`__,
# in which we are interested in minimizing the energy
#
# .. math:: E = \langle \phi \vert H \vert \phi \rangle.
#
# In VQE, we prepare a statevector :math:`\vert \phi \rangle` by applying
# the parameterized ansatz :math:`A(\Theta),` represented by a unitary matrix,
# to an initial state :math:`\vert 0 \rangle^{\otimes n}` where :math:`n` is the number of qubits. Then, the parameters :math:`\Theta` are
# optimized to minimize a cost function, which in this case is the energy:
#
# .. math::  E(\Theta) = \langle 0 \vert^{\otimes n} A(\Theta)^{\dagger} H A(\Theta) \vert 0 \rangle^{\otimes n}.
#
# This is done using a classical optimization method, which is typically
# gradient descent.
#
# To implement our example of VQE, we first define the molecular
# Hamiltonian for the :math:`H_2` molecule in the minimal `STO-3G basis <https://en.wikipedia.org/wiki/STO-nG_basis_sets>`__
# using PennyLane
#

import pennylane as qml
qml.capture.enable()
from pennylane import qchem
import numpy as np
from jax import numpy as jnp

symbols = ["H", "H"]
coordinates = jnp.array([[0.0, 0.0, -0.6614], [0.0, 0.0, 0.6614]])
basis_set = "sto-3g"
electrons = 2

molecule = qchem.Molecule(symbols, coordinates, basis_name=basis_set)

H, qubits = qchem.molecular_hamiltonian(molecule)


######################################################################
# We then initialize the Hartree-Fock state
# :math:`\vert \phi_{HF}\rangle=\vert 1100 \rangle`
#

hf = qchem.hf_state(electrons, qubits)


######################################################################
# Next, we implement the ansatz :math:`A(\Theta).` In this case, we use the
# class :class:`~.pennylane.AllSinglesDoubles`, which enables us to apply all possible combinations of single and
# double excitations obeying the Pauli principle to the Hartree-Fock
# state. Single and double excitation gates, denoted :math:`G^{(1)}(\Theta)` and :math:`G^{(2)}(\Theta)` respectively, are
# conveniently implemented in PennyLane with :class:`~.pennylane.SingleExcitation`
# and :class:`~.pennylane.DoubleExcitation` classes. You can find more information
# about how these gates work in this `video <https://youtu.be/4Xnxa6tzPeA>`__ and in the demo :doc:`demos/tutorial_givens_rotations`.
#

singles, doubles = qchem.excitations(electrons=electrons, orbitals=qubits)
num_theta = len(singles) + len(doubles)


def circuit_VQE(theta, wires):
    qml.AllSinglesDoubles(
        weights=theta, wires=wires, hf_state=hf, singles=singles, doubles=doubles
    )


######################################################################
# Once this is defined, we can run the VQE algorithm. We first need to
# define a circuit for the cost function.
#

import optax
import jax

jax.config.update("jax_enable_x64", True) # use double-precision numbers

dev = qml.device("lightning.qubit", wires=qubits)


@qml.qjit
@qml.qnode(dev, interface="jax")
def cost(theta):
    circuit_VQE(theta, range(qubits))
    return qml.expval(H)


######################################################################
# We then fix the classical optimization parameters ``stepsize`` and
# ``max_iteration``:
#

stepsize = 0.4
max_iterations = 30
opt = optax.sgd(learning_rate=stepsize) # sgd stands for StochasticGradientDescent
init_params = jnp.zeros(num_theta)


######################################################################
# Finally, we run the algorithm.
#


@qml.qjit
def update_step(i, params, opt_state):
    """Perform a single gradient update step"""
    grads = qml.grad(cost)(params)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return (params, opt_state)


loss_history = []

opt_state = opt.init(init_params)
params = init_params

for i in range(30):
    params, opt_state = update_step(i, params, opt_state)
    energy = cost(params)

energy_VQE = cost(params)
theta_opt = params

print("VQE energy: %.4f" % (energy_VQE))
print("Optimal parameters:", theta_opt)


######################################################################
# Note that as an output we obtain the VQE approximation to the ground
# state energy and a set of optimized parameters :math:`\Theta` that
# define the ground state through the ansatz :math:`A(\Theta).` We will need
# to save these two quantities, as they are necessary to implement CB-VQE
# in the following steps.
#


######################################################################
# Classically-Boosted VQE
# -----------------------
#
# Now we are ready to present the classically-boosted version of VQE.
#
# The key of this new method relies on the notion of the
# `generalized eigenvalue problem <https://en.wikipedia.org/wiki/Generalized_eigenvalue_problem>`__.
# The main idea is to restrict the problem of finding the ground state to
# an eigenvalue problem in a subspace :math:`\mathcal{H}^{\prime}` of the
# complete Hilbert space :math:`\mathcal{H}.` If this subspace is spanned
# by a combination of both classical and quantum states, we can run parts
# of our algorithm on classical hardware and thus reduce the number of
# measurements needed to reach a certain precision threshold. The generalized
# eigenvalue problem is expressed as
#
# .. math:: \bar{H} \vec{v}=  \lambda \bar{S} \vec{v},
#
# where the matrix :math:`\bar{S}` contains the overlaps between the basis states and :math:`\bar{H}`
# is the Hamiltonian :math:`H` projected into the
# subspace of interest, i.e. with the entries
#
# .. math:: \bar{H}_{\alpha, \beta} = \langle \phi_\alpha \vert H \vert \phi_\beta \rangle,
#
# for all :math:`\vert \phi_\alpha \rangle` and :math:`\vert \phi_\beta \rangle` in :math:`\mathcal{H}^{\prime}.`
# For a complete orthonormal basis, the overlap matrix
# :math:`\bar{S}` would simply be the identity matrix. However, we need to
# take a more general approach which works for a subspace spanned by
# potentially non-orthogonal states. We can retrieve the representation of
# :math:`S` by calculating
#
# .. math:: \bar{S}_{\alpha, \beta} = \langle \phi_\alpha \vert \phi_\beta \rangle,
#
# for all :math:`\vert \phi_\alpha \rangle` and :math:`\vert \phi_\beta \rangle` in :math:`\mathcal{H}^{\prime}.`
# Finally, note that :math:`\vec{v}` and :math:`\lambda` are the eigenvectors and
# eigenvalues respectively. Our goal is to find the lowest
# eigenvalue :math:`\lambda_0.`
#


######################################################################
# Equipped with the useful mathematical description of generalized
# eigenvalue problems, we can now choose our subspace such that some of
# the states :math:`\vert \phi_{\alpha} \rangle \in \mathcal{H}^{\prime}` are
# classically tractable.
#
# We will consider the simplest case in which the subspace is spanned only
# by one classical state :math:`\vert \phi_{HF} \rangle` and one quantum
# state :math:`\vert \phi_{q} \rangle.` More precisely, we define the
# classical state to be a single
# `Slater determinant <https://en.wikipedia.org/wiki/Slater_determinant>`__,
# which directly hints towards using the *Hartree-Fock* state for several
# reasons. First of all, it is well-known that the Hartree-Fock state is a
# good candidate to approximate the ground state in the mean-field limit.
# Secondly, we already computed it when we built the molecular Hamiltonian
# for the standard VQE!
#


######################################################################
# To summarize, our goal is to build the Hamiltonian :math:`\bar{H}` and
# the overlap matrix :math:`\bar{S},` which act on the subspace
# :math:`\mathcal{H}^{\prime} \subseteq \mathcal{H}` spanned by
# :math:`\{\vert \phi_{HF} \rangle, \vert \phi_q \rangle\}.` These will be
# two-dimensional matrices, and in the following sections we will show how
# to compute all their entries step by step.
#
# As done previously, we start by importing *PennyLane*, *Qchem* and
# differentiable *NumPy* followed by defining the molecular Hamiltonian in
# the Hartree-Fock basis for :math:`H_2.`
#

symbols = ["H", "H"]
coordinates = np.array([[0.0, 0.0, -0.6614], [0.0, 0.0, 0.6614]])
basis_set = "sto-3g"

molecule = qchem.Molecule(symbols, coordinates, basis_name=basis_set)

H, qubits = qchem.molecular_hamiltonian(molecule)


######################################################################
# Computing Classical Quantities
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# We first set out to calculate the purely classical part of the
# Hamiltonian :math:`H.` Since we only have one classical state this will
# already correspond to a scalar energy value. The terms can be expressed
# as
#
# .. math:: H_{11} = \langle \phi_{HF} \vert H \vert \phi_{HF} \rangle \quad \text{and} \quad S_{11} = \langle \phi_{HF} \vert \phi_{HF} \rangle
#
# which is tractable using classical methods. This energy corresponds to
# the Hartree-Fock energy due to our convenient choice of the classical
# state. Note that the computation of the classical component of the
# overlap matrix
# :math:`S_{11} = \langle \phi_{HF} \vert \phi_{HF} \rangle = 1` is
# trivial.
#
# Using PennyLane, we can access the Hartree-Fock energy by looking at the
# fermionic Hamiltonian, which is the Hamiltonian on the basis of Slater
# determinants. The basis is organized in lexicographic order, meaning
# that if we want the entry corresponding to the Hartree-Fock determinant
# :math:`\vert 1100 \rangle,` we will have to take the entry
# :math:`H_{i,i},` where :math:`1100` is the binary representation of the
# index :math:`i.`
#

hf_state = qchem.hf_state(electrons, qubits)
fermionic_Hamiltonian = H.sparse_matrix().todense()

# we first convert the HF slater determinant to a string
binary_string = "".join([str(i) for i in hf_state])
# we then obtain the integer corresponding to its binary representation
idx0 = int(binary_string, 2)
# finally we access the entry that corresponds to the HF energy
H11 = fermionic_Hamiltonian[idx0, idx0]
S11 = 1


######################################################################
# Computing Quantum Quantities
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# We now move on to the purely quantum part of the Hamiltonian, i.e. the
# entry
#
# .. math:: H_{22} = \langle \phi_{q} \vert H \vert \phi_{q} \rangle,
#
# where :math:`\vert \phi_q \rangle` is the quantum state. This state is
# just the output of the standard VQE with a given ansatz, following the
# steps in the first section. Therefore, the entry :math:`H_{22}` just
# corresponds to the final energy of the VQE. In particular, note that the
# quantum state can be written as
# :math:`\vert \phi_{q} \rangle = A(\Theta^*) \vert \phi_{HF} \rangle`
# where :math:`A(\Theta^*)` is the ansatz of the VQE with the optimised
# parameters :math:`\Theta^*.` Once again, we have
# :math:`S_{22}=\langle \phi_{q} \vert \phi_{q} \rangle = 1` for the
# overlap matrix.
#

H22 = energy_VQE
S22 = 1


######################################################################
# Computing Mixed Quantities
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# The final part of the algorithm computes the cross-terms between the
# classical and quantum state
#
# .. math:: H_{12} = \langle \phi_{HF} \vert H \vert \phi_{q} \rangle = H_{21}^{\dagger}.
#
# This part of the algorithm is slightly more complicated than the
# previous steps, since we still want to make use of the classical component
# of the problem in order to minimize the number of required measurements.
#
# Keep in mind that most algorithms usually perform computations either on
# fully classically or quantum tractable Hilbert spaces. CB-VQE takes
# advantage of the classical part of the problem while still calculating a
# classically-intractable quantity by using the so-called
# `Hadamard test <https://en.wikipedia.org/wiki/Hadamard_test_(quantum_computation)>`__
# to construct :math:`H_{12}.` The Hadamard test is a prime example of an
# indirect measurement, which allows us to measure properties of a state
# without (completely) destroying it.
#
# .. figure:: ../_static/demonstration_assets/classically_boosted_vqe/boosted_hadamard_test.png
#     :align: center
#     :width: 50%
#
# As the Hadamard test returns the real part of a coefficient from a unitary representing
# an operation, we will focus on calculating the quantities
#
# .. math:: H_{12} = \sum_{i} Re(\langle \phi_q \vert i \rangle) \langle i \vert H \vert \phi_{HF} \rangle,
#
# .. math:: S_{12} = Re(\langle \phi_q \vert \phi_{HF} \rangle),
#
# where :math:`\lvert i \rangle` are the computational basis states of the system,
# i.e. the basis of single Slater determinants. Note that we have to decompose the Hamiltonian
# into a sum of unitaries. For the problem under consideration, the set of relevant computational basis states for which
# :math:`\langle i \vert H \vert \phi_{HF}\rangle \neq 0` contains all the
# single and double excitations (allowed by spin symmetries), namely, the states
#
# .. math:: \vert 1100 \rangle, \vert 1001 \rangle, \vert 0110 \rangle, \vert 0011 \rangle.
#
# Specifically, the set of computational basis states includes the
# *Hartree-Fock* state :math:`\lvert i_0 \rangle = \vert \phi_{HF} \rangle = \vert 1100 \rangle` and the
# projections :math:`\langle i \vert H \vert \phi_{HF} \rangle` can be
# extracted analytically from the fermionic Hamiltonian that we computed
# above. This is done by accessing the entries by the index given by the binary
# expression of each Slater determinant.
#
# The Hadamard test is required to compute the real part of
# :math:`\langle \phi_q \vert i \rangle.`
#
# To implement the Hadamard test, we need a register of :math:`n` qubits
# given by the size of the molecular Hamiltonian (:math:`n=4` in our case)
# initialized in the state :math:`\rvert 0 \rangle^{\otimes n}` and an ancillary
# qubit prepared in the :math:`\rvert 0 \rangle` state.
#
# In order to generate :math:`\langle \phi_q \vert i \rangle,` we take
# :math:`U_q` such that
# :math:`U_q \vert 0 \rangle^{\otimes n} = \vert \phi_q \rangle.`
# This is equivalent to using the standard VQE ansatz with the optimized
# parameters :math:`\Theta^*` that we obtained in the previous section
# :math:`U_q = A(\Theta^*).` Moreover,
# we also need :math:`U_i` such that
# :math:`U_i \vert 0^n \rangle = \vert \phi_i \rangle.` In this case, this
# is just a mapping of a classical basis state into the circuit consisting
# of :math:`X` gates and can be easily implemented using PennyLane’s
# function ``qml.BasisState(i, n)``.
#

wires = range(qubits + 1)
dev = qml.device("lightning.qubit", wires=wires)


@qml.qnode(dev, interface="jax")
def hadamard_test(Uq, Ucl, component="real"):
    if component == "imag":
        qml.RX(math.pi / 2, wires=wires[1:])

    qml.Hadamard(wires=[0])
    qml.ControlledQubitUnitary(
        Uq.conjugate().T @ Ucl, wires=wires
    )
    qml.Hadamard(wires=[0])

    return qml.probs(wires=[0])


######################################################################
# Now, we are ready to compute the Hamiltonian
# cross-terms.
#


def circuit_product_state(state):
    qml.BasisState(state, range(qubits))


wire_order = list(range(qubits))
Uq = qml.matrix(circuit_VQE, wire_order=wire_order)(theta_opt, wire_order)

H12 = 0
relevant_basis_states = np.array(
    [[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1], [0, 0, 1, 1]]
)
for j, basis_state in enumerate(relevant_basis_states):
    Ucl = qml.matrix(circuit_product_state, wire_order=wire_order)(basis_state)
    probs = hadamard_test(Uq, Ucl)
    # The projection Re(<phi_q|i>) corresponds to 2p-1
    y = 2 * probs[0] - 1
    # We retrieve the quantities <i|H|HF> from the fermionic Hamiltonian
    binary_string = "".join([str(coeff) for coeff in basis_state])
    idx = int(binary_string, 2)
    overlap_H = fermionic_Hamiltonian[idx0, idx]
    # We sum over all computational basis states
    H12 += y * overlap_H
    # y0 corresponds to Re(<phi_q|HF>)
    if j == 0:
        y0 = y

H21 = np.conjugate(H12)


######################################################################
# The cross terms of the :math:`S` matrix are defined making
# use of the projections with the *Hartree-Fock* state.
#

S12 = y0
S21 = y0.conjugate()


######################################################################
# Solving the generalized eigenvalue problem
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# We are ready to solve the generalized eigenvalue problem. For
# this, we will build the matrices :math:`H` and :math:`S` and use `scipy`
# to obtain the lowest eigenvalue.
#

from scipy import linalg

S = np.array([[S11, S12], [S21, S22]])
H = np.array([[H11, H12], [H21, H22]])

evals = linalg.eigvals(H, S)
energy_CBVQE = np.min(evals).real

print("CB-VQE energy %.4f" % (energy_CBVQE))


######################################################################
# Measurement analysis
# -----------------------
#
#
# CB-VQE is helpful when it comes to reducing the number of measurements
# that are required to reach a given precision in the ground state energy.
# In fact, for very small systems it can be shown that the classically-boosted method
# reduces the number of required measurements by a factor of :math:`1000` [#Radin2021]_.
#
# Let's see if this is the case for the example above.
# Now that we know how to run standard VQE and CB-VQE algorithms, we can re-run the code above
# for a finite number of measurements. This is done by specifying the number of
# shots in the definition of the devices, for example, ``num_shots = 20``. By doing this, Pennylane
# will output the expectation value of the energy computed from a sample of 20 measurements.
# Then, we simply run both VQE and CB-VQE enough times to obtain statistics on the results.
#
# .. figure:: ../_static/demonstration_assets/classically_boosted_vqe/energy_deviation.png
#     :align: center
#     :width: 80%
#
# In the plot above, the dashed line corresponds to the true ground state energy of the :math:`H_2` molecule.
# In the x-axis we represent the number of measurements that are used to compute the expected value of the
# Hamiltonian (`num_shots`). In the y-axis, we plot the mean value and the standard deviation of the energies
# obtained from a sample of 100 circuit evaluations.
# As expected, CB-VQE leads to a better approximation of the ground state energy - the mean energies are lower-
# and, most importantly, to a much smaller standard deviation, improving on the results given
# by standard VQE by several orders of magnitude when considering a small number of measurements.
# As expected, for a large number of measurements both algorithms start to converge to similar
# results and the standard deviation decreases.
#
#
# `Note: In order to obtain these results, we had to discard the samples in which the VQE shot noise
# underestimated the true ground state energy of the problem, since this was leading to large
# variances in the CB-VQE estimation of the energy.`
#
#


######################################################################
# Conclusions
# -----------------------
#
# In this demo, we have learnt how to implement the CB-VQE algorithm in PennyLane. Furthermore, it was observed that we require
# fewer measurements to be executed on a quantum computer to reach the same accuracy as standard VQE.
# Such algorithms could be executed on smaller quantum computers, potentially allowing us to implement useful
# quantum algorithms on real hardware sooner than expected.
#
#


######################################################################
# References
# -----------------------
#
# .. [#Radin2021]
#
#     M. D. Radin. (2021) "Classically-Boosted Variational Quantum Eigensolver",
#     `arXiv:2106.04755 [quant-ph] <https://arxiv.org/abs/2106.04755>`__ (2021)
#
