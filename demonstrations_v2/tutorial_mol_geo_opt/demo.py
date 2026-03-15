r"""
Optimization of molecular geometries
====================================

.. meta::
    :property="og:description": Find the equilibrium geometry of a molecule
    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets/fig_pes.png

.. related::
   tutorial_quantum_chemistry Building molecular Hamiltonians
   tutorial_vqe A brief overview of VQE
   tutorial_givens_rotations Givens rotations for quantum chemistry
   
Predicting the most stable arrangement of atoms in a molecule is one of the most important tasks
in quantum chemistry. Essentially, this is an optimization problem where the total energy of the
molecule is minimized with respect to the positions of the atomic nuclei. The molecular
geometry obtained from this calculation is in fact the starting point for many simulations of
molecular properties. If the geometry is inaccurate, then any calculations that rely on it may
also be inaccurate.

Since the nuclei are much heavier than the electrons, we can treat them as point particles
clamped to their positions. Under this assumption, the total energy of the molecule :math:`E(x)`
depends on the nuclear coordinates :math:`x,` which define the potential energy surface.
Solving the stationary problem :math:`\nabla_x E(x) = 0` corresponds to molecular geometry
optimization and the optimized nuclear coordinates determine the equilibrium geometry of the
molecule. The figure below illustrates these concepts for the
`trihydrogen cation <https://en.wikipedia.org/wiki/Trihydrogen_cation>`_. Its equilibrium
geometry in the electronic ground state corresponds to the minimum energy of the potential
energy surface. At this minimum, the three hydrogen atoms are located at the vertices of an
equilateral triangle whose side length is the optimized bond length :math:`d.`

|

.. figure:: /_static/demonstration_assets/mol_geo_opt/fig_pes.png
    :width: 50%
    :align: center

|

In this tutorial, you will learn how to recast the problem of finding the equilibrium
geometry of a molecule in terms of a general variational quantum algorithm. The
central idea is to consider explicitly that the target electronic Hamiltonian :math:`H(x)`
is a **parametrized** observable that depends on the nuclear coordinates :math:`x.` This
implies that the objective function, defined by the expectation value of the Hamiltonian
computed in the trial state prepared by a quantum computer, depends on both the quantum
circuit and the Hamiltonian parameters.

The quantum algorithm in a nutshell
-----------------------------------

The goal of the variational algorithm is to find the global minimum of 
the cost function :math:`g(\theta, x) = \langle \Psi(\theta) \vert H(x) \vert \Psi(\theta) \rangle`
with respect to the circuit parameters :math:`\theta` and the
nuclear coordinates :math:`x` entering the electronic Hamiltonian of the molecule. To that end,
we use a gradient-descent method and follow a **joint** optimization scheme where the gradients of 
the cost function with respect to circuit and Hamiltonian parameters are simultaneously computed 
at each step. This approach does not require nested optimization of the state 
parameters for each set of nuclear coordinates, as occurs in classical algorithms for
molecular geometry optimization, where the energy minimum is searched for along the potential energy 
surface of the electronic state [#jensenbook]_.

In this tutorial we demonstrate how to use PennyLane to implement
quantum optimization of molecular geometries. The algorithm consists of the following steps:

#. Build the parametrized electronic Hamiltonian :math:`H(x)` of the molecule.

#. Design the variational quantum circuit to prepare the electronic trial state of the
   molecule, :math:`\vert \Psi(\theta) \rangle.`

#. Define the cost function :math:`g(\theta, x) = \langle \Psi(\theta) \vert H(x) \vert
   \Psi(\theta) \rangle`.

#. Initialize the variational parameters :math:`\theta` and :math:`x.` Perform a joint
   optimization of the circuit and Hamiltonian parameters to minimize the cost function 
   :math:`g(\theta, x).` The gradient with respect to the circuit parameters can be obtained
   using a variety of established methods, which are natively supported in PennyLane. The
   gradients with respect to the nuclear coordinates can be computed using the formula

   .. math::

       \nabla_x g(\theta, x) = \langle \Psi(\theta) \vert \nabla_x H(x) \vert \Psi(\theta) \rangle.

Once the optimization is finalized, the circuit parameters determine the energy of the
electronic state, and the nuclear coordinates determine the equilibrium geometry of the
molecule in this state.

Let's get started! ⚛️

Building the parametrized electronic Hamiltonian
------------------------------------------------

In this example, we want to optimize the geometry of the trihydrogen cation
:math:`\mathrm{H}_3^+,` described in a minimal basis set, where two electrons are shared
between three hydrogen atoms (see figure above). The molecule is specified by providing a list
with the symbols of the atomic species and a one-dimensional array with the initial
set of nuclear coordinates in `atomic units
<https://en.wikipedia.org/wiki/Hartree_atomic_units>`_ .

"""

import jax
from jax import numpy as jnp

jax.config.update("jax_enable_x64", True)

symbols = ["H", "H", "H"]
x = jnp.array([[0.028, 0.054, 0.0], [0.986, 1.610, 0.0], [1.855, 0.002, 0.0]])

##############################################################################
# Next, we need to build the parametrized electronic Hamiltonian :math:`H(x).`
# We use the Jordan-Wigner transformation [#seeley2012]_ to represent the fermionic
# Hamiltonian as a linear combination of Pauli operators,
#
# .. math::
#
#     H(x) = \sum_j h_j(x) \prod_i^{N} \sigma_i^{(j)}.
#
# The expansion coefficients :math:`h_j(x)` carry the dependence on the coordinates :math:`x,`
# the operators :math:`\sigma_i` represent the Pauli group :math:`\{I, X, Y, Z\},` and
# :math:`N` is the number of qubits required to represent the electronic wave function.
#
# We define the function ``H(x)`` to build the parametrized Hamiltonian
# of the trihydrogen cation using the
# :func:`~.pennylane.qchem.molecular_hamiltonian` function.

import pennylane as qml
qml.capture.enable()


def H(x):
    molecule = qml.qchem.Molecule(symbols, x, charge=1)
    return qml.qchem.molecular_hamiltonian(molecule)[0]


##############################################################################
# The variational quantum circuit
# -------------------------------
#
# Here, we describe the second step of the quantum algorithm: define the quantum circuit
# to prepare the electronic ground-state :math:`\vert \Psi(\theta)\rangle` of the
# :math:`\mathrm{H}_3^+` molecule.
#
# Six qubits are required to encode the occupation number of the molecular spin-orbitals.
# To capture the effects of electronic correlations [#kohanoff2006]_, we need to prepare
# the :math:`N`-qubit system in a superposition of the Hartree-Fock state
# :math:`\vert 110000 \rangle` with other states that differ by a double- or single-excitation.
# For example, the state :math:`\vert 000011 \rangle` is obtained by exciting two particles
# from qubits 0, 1 to 4, 5. Similarly, the state :math:`\vert 011000 \rangle` corresponds to a
# single excitation from qubit 0 to 2. This can be done using the single-excitation and
# double-excitation gates :math:`G` and :math:`G^{(2)}` [#qchemcircuits]_ implemented
# in the form of Givens rotations in PennyLane. For more details see the tutorial
# :doc:`demos/tutorial_givens_rotations`.
#
# In addition, we use an adaptive algorithm [#geo_opt_paper]_ to select the excitation
# operations included in the variational quantum circuit. The algorithm proceeds as follows:
#
# #. Generate the indices of the qubits involved in all single- and
#    double-excitations.
#    For example, the indices of the singly-excited state :math:`\vert 011000 \rangle`
#    are given by the list ``[0, 2]``. Similarly, the indices of the doubly-excited
#    state :math:`\vert 000011 \rangle` are ``[0, 1, 4, 5]`.`
#
# #. Construct the circuit using all double-excitation gates. Compute the gradient
#    of the cost function :math:`g(\theta, x)` with respect to each double-excitation
#    gate and retain only those with non-zero gradient.
#
# #. Include the selected double-excitation gates and repeat the process for the
#    single-excitation gates.
#
# #. Build the final variational quantum circuit by including the selected gates.
#
# For the :math:`\mathrm{H}_3^+` molecule in a minimal basis set we have a total of eight
# excitations of the reference state. After applying the adaptive algorithm the final
# quantum circuit contains only two double-excitation operations that act on the qubits
# ``[0, 1, 2, 3]`` and ``[0, 1, 4, 5]``. The circuit is illustrated in the figure below.
#
# |
#
# .. figure:: /_static/demonstration_assets/mol_geo_opt/fig_circuit.png
#     :width: 60%
#     :align: center
#
# |
#
# To implement this quantum circuit, we use the
# :func:`~.pennylane.qchem.hf_state` function to generate the
# occupation-number vector representing the Hartree-Fock state

hf = qml.qchem.hf_state(electrons=2, orbitals=6)
print(hf)

##############################################################################
# The ``hf`` array is used by the :class:`~.pennylane.BasisState` operation to initialize
# the qubit register. Then, the :class:`~.pennylane.DoubleExcitation` operations are applied
# First, we define the quantum device used to compute the expectation value.
# In this example, we use the ``default.qubit`` simulator:
num_wires = 6
dev = qml.device("default.qubit", wires=num_wires)


@qml.qnode(dev, interface="jax")
def circuit(params, obs, wires):
    qml.BasisState(hf, wires=wires)
    qml.DoubleExcitation(params[0], wires=[0, 1, 2, 3])
    qml.DoubleExcitation(params[1], wires=[0, 1, 4, 5])

    return qml.expval(obs)


##############################################################################
# This circuit prepares the trial state
#
# .. math::
#
#     \vert\Psi(\theta_1, \theta_2)\rangle =
#     \mathrm{cos}(\theta_1)\mathrm{cos}(\theta_2)\vert110000\rangle -
#     \mathrm{cos}(\theta_1)\mathrm{sin}(\theta_2)\vert000011\rangle -
#     \mathrm{sin}(\theta_1)\vert001100\rangle,
#
# where :math:`\theta_1` and :math:`\theta_2` are the circuit parameters that need to be
# optimized to find the ground-state energy of the molecule.
#
# The cost function and the nuclear gradients
# -------------------------------------------
#
# The third step of the algorithm is to define the cost function
# :math:`g(\theta, x) = \langle \Psi(\theta) \vert H(x) \vert\Psi(\theta) \rangle.` It
# evaluates the expectation value of the parametrized Hamiltonian :math:`H(x)` in the
# trial state :math:`\vert\Psi(\theta)\rangle.`

##############################################################################
# Next, we define the ``cost`` function :math:`g(\theta, x)` which depends on
# both the circuit and the Hamiltonian parameters. Specifically we consider the
# expectation values of the Hamiltonian.


def cost(params, x):
    hamiltonian = H(x)
    return circuit(params, obs=hamiltonian, wires=range(num_wires))


##############################################################################
#
# We minimize the cost function :math:`g(\theta, x)` using a gradient-based
# method, and compute the gradients with respect to both the
# circuit parameters :math:`\theta` and the nuclear coordinates :math:`x.`
# The circuit gradients are computed analytically using the automatic differentiation
# techniques available in PennyLane. The nuclear gradients are evaluated
# by taking the expectation value of the gradient of the electronic Hamiltonian,
#
# .. math::
#
#     \nabla_x g(\theta, x) = \langle \Psi(\theta) \vert \nabla_x H(x) \vert \Psi(\theta) \rangle.
#
# We use the :func:`finite_diff` function to compute the gradient of
# the Hamiltonian using a central-difference approximation. Then, we evaluate the expectation
# value of the gradient components :math:`\frac{\partial H(x)}{\partial x_i}.` This is implemented by
# the function ``grad_x``:


def finite_diff(f, x, delta=0.01):
    """Compute the central-difference finite difference of a function"""
    gradient = []
    x = jnp.ravel(x)

    for i in range(len(x)):
        shift = jnp.zeros_like(x)
        shift = shift.at[i].set(0.5*delta)
        res = (f(x + shift) - f(x - shift)) * delta**-1
        gradient.append(res)

    return gradient


def grad_x(params, x):
    grad_h = finite_diff(H, x)
    grad = [circuit(params, obs=obs, wires=range(num_wires)) for obs in grad_h]
    return jnp.array(grad).reshape(x.shape)


##############################################################################
# Optimization of the molecular geometry
# --------------------------------------
#
# Finally, we proceed to minimize our cost function to find the ground state equilibrium
# geometry of the :math:`\mathrm{H}_3^+` molecule. As a reminder,
# the circuit parameters and the nuclear coordinates will be jointly optimized at
# each optimization step. This approach does not require nested VQE
# optimization of the circuit parameters for each set of nuclear coordinates.
#
# First, we initialize the circuit parameters :math:`\theta.` The angles
# :math:`\theta_1` and :math:`\theta_2` are set to zero so that the
# initial state :math:`\vert\Psi(\theta_1, \theta_2)\rangle`
# is the Hartree-Fock state.

theta = jnp.array([0.0, 0.0])

##############################################################################
# The initial set of nuclear coordinates :math:`x,` defined at
# the beginning of the tutorial, was computed classically within the Hartree-Fock
# approximation using the GAMESS program [#ref_gamess]_. This is a natural choice
# for the starting geometry that we are aiming to improve due to the electronic
# correlation effects included in the trial state :math:`\vert\Psi(\theta)\rangle.`
#
# We carry out the optimization over a maximum of 36 steps.
# The circuit parameters and the nuclear coordinates are optimized until the
# maximum component of the nuclear gradient :math:`\nabla_x g(\theta,x)` is
# less than or equal to :math:`10^{-5}` Hartree/Bohr. Typically, this is the
# convergence criterion used for optimizing molecular geometries in
# quantum chemistry simulations.

# store the values of the cost function
energies = []

# store the values of the bond length
bond_length = []

# Factor to convert from Bohrs to Angstroms
bohr_angs = 0.529177210903

for n in range(36):
    # gradient for params
    g_param = jax.grad(cost, argnums=[0])(theta, x)[0]
    theta = theta - 0.8 * g_param

    # gradient for coordinates
    value, _ = jax.value_and_grad(cost, argnums=1)(theta, x)
    grad = grad_x(theta, x)
    x = x - 0.8 * grad
    energies.append(value)
    bond_length.append(jnp.linalg.norm(x[0] - x[1]) * bohr_angs) 

    if n % 4 == 0:
        print(f"Step = {n},  E = {energies[-1]:.8f} Ha,  bond length = {bond_length[-1]:.5f} A")

    # Check maximum component of the nuclear gradient
    if jnp.max(grad_x(theta, x)) <= 1e-04:
        break

print("\n" f"Final value of the ground-state energy = {energies[-1]:.8f} Ha")
print("\n" "Ground-state equilibrium geometry")
print("%s %4s %8s %8s" % ("symbol", "x", "y", "z"))
for i, atom in enumerate(symbols):
    print(f"  {atom}    {x[0][i]:.4f}   {x[1][i]:.4f}   {x[2][i]:.4f}")

##############################################################################
# Next, we plot the values of the ground state energy of the molecule
# and the bond length as a function of the optimization step.

import matplotlib.pyplot as plt

fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(12)

# Add energy plot on column 1
E_fci = -1.27443765658
E_vqe = jnp.array(energies)
ax1 = fig.add_subplot(121)
ax1.plot(range(n + 1), E_vqe - E_fci, "go", ls="dashed")
ax1.plot(range(n + 1), jnp.full(n + 1, 0.001), color="red")
ax1.set_xlabel("Optimization step", fontsize=13)
ax1.set_ylabel("$E_{VQE} - E_{FCI}$ (Hartree)", fontsize=13)
ax1.text(5, 0.0013, r"Chemical accuracy", fontsize=13)
plt.yscale("log")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add bond length plot on column 2
d_fci = 0.986
ax2 = fig.add_subplot(122)
ax2.plot(range(n + 1), bond_length, "go", ls="dashed")
ax2.plot(range(n + 1), jnp.full(n + 1, d_fci), color="red")
ax2.set_ylim([0.965, 0.99])
ax2.set_xlabel("Optimization step", fontsize=13)
ax2.set_ylabel("bond length ($\AA$)", fontsize=13)
ax2.text(5, 0.9865, r"Equilibrium bond length", fontsize=13)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.subplots_adjust(wspace=0.3)
plt.show()

##############################################################################
# |
# Notice that despite the fact that the ground-state energy is already converged
# within chemical accuracy (:math:`0.0016` Ha) after the first iteration, more
# optimization steps are required to find the equilibrium bond length of the
# molecule.
#
# The figure below animates snapshots of the atomic structure of the
# trihydrogen cation as the quantum algorithm was searching for the equilibrium
# geometry. For visualization purposes, the initial nuclear coordinates were
# generated by perturbing the HF geometry. The quantum algorithm
# is able to find the correct equilibrium geometry of the :math:`\mathrm{H}_3^+`
# molecule where the three H atoms are located at the vertices of an equilateral triangle.
#
# |
#
# .. figure:: /_static/demonstration_assets/mol_geo_opt/fig_movie.gif
#     :width: 50%
#     :align: center
#
# |
#
# To summarize, we have shown how the scope of variational quantum algorithms can be
# extended to perform quantum simulations of molecules involving both the electronic and
# the nuclear degrees of freedom. The joint optimization scheme described here
# is a generalization of the usual VQE algorithm where only the electronic
# state is parametrized. Extending the applicability of the variational quantum algorithms to
# target parametrized Hamiltonians could be also relevant to simulate the optical properties of
# molecules where the fermionic observables depend also on the electric field of the
# incoming radiation [#pulay]_.
#
# References
# ----------
#
# .. [#jensenbook]
#
#     F. Jensen. "Introduction to computational chemistry".
#     (John Wiley & Sons, 2016).
#
# .. [#seeley2012]
#
#     Jacob T. Seeley, Martin J. Richard, Peter J. Love. "The Bravyi-Kitaev transformation for
#     quantum computation of electronic structure". `Journal of Chemical Physics 137, 224109
#     (2012).
#     <https://aip.scitation.org/doi/abs/10.1063/1.4768229>`__
#
# .. [#kohanoff2006]
#
#     Jorge Kohanoff. "Electronic structure calculations for solids and molecules: theory and
#     computational methods". (Cambridge University Press, 2006).
#
# .. [#qchemcircuits]
#
#     J.M. Arrazola, O. Di Matteo, N. Quesada, S. Jahangiri, A. Delgado, N. Killoran.
#     "Universal quantum circuits for quantum chemistry". arXiv:2106.13839, (2021)
#
# .. [#ref_gamess]
#
#     M.W. Schmidt, K.K. Baldridge, J.A. Boatz, S.T. Elbert, M.S. Gordon, J.H. Jensen,
#     S. Koseki, N. Matsunaga, K.A. Nguyen, S.Su, *et al.* "General atomic and molecular
#     electronic structure system". `Journal of Computational Chemistry 14, 1347 (1993)
#     <https://onlinelibrary.wiley.com/doi/10.1002/jcc.540141112>`__
#
# .. [#geo_opt_paper]
#
#     A. Delgado, J.M. Arrazola, S. Jahangiri, Z. Niu, J. Izaac, C. Roberts, N. Killoran.
#     "Variational quantum algorithm for molecular geometry optimization".
#     arXiv:2106.13840, (2021)
#
# .. [#pulay]
#
#     P. Pulay.
#     "Analytical derivative methods in quantum chemistry".
#     `Advances in Chemical Sciences (1987)
#     <https://onlinelibrary.wiley.com/doi/10.1002/9780470142943.ch4>`__
#
