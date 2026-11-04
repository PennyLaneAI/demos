r"""
An overview of VQE-LD
=====================

.. meta::
    :property="og:description": Improving energy and molecular properties by
                incorporating the convergence of the one-particle reduced density matrix in VQE.

.. related::

   tutorial_vq3 Find the ground state of a Hamiltonian using VQE.

The variational quantum eigensolver (VQE) is a relevant method for simulating
molecular systems on near-term quantum computers. While its primary application
is the estimation of ground-state energies, VQE also produces the one-particle
reduced density matrix (1-RDM), from which other relevant molecular properties
can be obtained. The accuracy of these properties depends on the reliability
and convergence of the 1-RDM, which is not guaranteed by energy-only
optimization.

Thus, a new algorithm is introduced: VQE-LD (VQE with Loss function
associated with a reduced Density matrix) that modifies the cost function by
adding to the energy a weighted term involving the RMSD of 1-RDM [#lima2026]_.

In this tutorial, you will learn how to implement the VQE-LD algorithm.
As an illustrative example, we use it to find the ground state of the
:math:`\mathrm{CH}_5^+` molecule. First, we define a set of auxiliary functions
to constructs the operator needed to compute the 1-RDM. Then, we set up the
molecular system using PySCF to obtain the Hartree-Fock reference and PennyLane
to calculate the Hamiltonian.

Next, we define the a quantum circuit using the GateFabric ansatz and the
expectation value function to compute the energy. We also define the function
to compute the 1-RDM from the expectation values of the one-body excitation
operators. Finally, we implement the VQE-LD optimization loop, combining the
gradients of the energy and the 1-RDM terms in the cost function.

Let's get started setting up the environment!

"""

import jax
import optax
import pennylane as qml
from pyscf import gto, scf, ci
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

# Constants
Ang2Bohr = 1.8897259886
CREATION = 0
ANNIHILATION = 1

##############################################################################
# 1-RDM Operator
# -----------------------------------
# The 1-RDM is defined as
#
# .. math::
#     D_{pq} = \langle \Psi | \hat{a}_p^\dagger \hat{a}_q | \Psi \rangle,
#
# where :math:`\hat{a}_p^\dagger` and :math:`\hat{a}_q` are the fermionic creation and annihilation operators.
# This matrix encodes the occupation of molecular orbitals through its diagonal elements
# and the quantum coherence between orbitals through its off-diagonal elements.
#
# Experimentally, the 1-RDM is reconstructed from the expectation values of fermionic operators
# mapped to qubits via the Jordan-Wigner transformation (function ``pauli_string_fermion_operator``).
#
# Since the operator :math:`\hat{a}_p^\dagger \hat{a}_q` is not Hermitian for :math:`p \neq q`,
# its expectation value may, in general, be complex.
# Furthermore, as only Hermitian operators correspond to physical observables,
# it cannot be directly measured on a quantum computer.
# To make these quantities measurable, the operator is decomposed into
# a sum of Hermitian operators: the symmetric :math:`\frac{1}{2}({a}_p^\dagger \hat{a}_q + {a}_q^\dagger \hat{a}_p)`
# and antisymmetric :math:`\frac{1}{2}({a}_p^\dagger \hat{a}_q - {a}_q^\dagger \hat{a}_p)` combinations,
# whose expectation values correspond to the real and imaginary parts of the 1-RDM, respectively.
# After the Jordan-Wigner transformation, these quantities can be expressed as combinations of Pauli strings
# (functions ``rdm1_real_operator`` and ``rdm1_imag_operator``):
#
# .. math::
#     \mathrm{Re}[D_{pq}] &= \tfrac{1}{2}\,\langle \Psi(\theta) |
#     \hat{a}_p^\dagger \hat{a}_q + \hat{a}_q^\dagger \hat{a}_p,
#     | \Psi(\theta) \rangle \\
#     &= \frac{1}{4} \langle \Psi(\theta) |
#     \left( X_p X_q + Y_p Y_q \right)
#     \prod_{j=p+1}^{q-1} Z_j
#     | \Psi(\theta) \rangle
#
# .. math::
#     \mathrm{Im}[D_{pq}] &= \frac{1}{2i}\,\langle \Psi(\theta) |
#     \hat{a}_p^\dagger \hat{a}_q - \hat{a}_q^\dagger \hat{a}_p
#     | \Psi(\theta) \rangle \\
#     &= \frac{1}{4} \langle \Psi(\theta) |
#     \left( X_p Y_q - Y_p X_q \right)
#     \prod_{j=p+1}^{q-1} Z_j
#     | \Psi(\theta) \rangle
#
# For the particular cases of the diagonal elements (:math:`p = q`),
# the imaginary term vanishes and the real part reduces to the
# expectation value of the number operator, that can be expressed as combinations of Pauli strings
# after the Jordan-Wigner transformation:
#
# .. math::
#     D_{pp} = \langle \Psi(\theta) | \hat{a}_p^\dagger \hat{a}_p | \Psi(\theta) \rangle
#     = \frac{1 - \langle \Psi(\theta) | Z_p | \Psi(\theta) \rangle}{2}
#
# The full 1-RDM can be constructed by evaluating only the independent elements,
# which correspond to the aforementioned diagonal elements, and
# in the upper triangular part of the matrix (:math:`p \leq q`) (``function rdm1_observables``):
#
# .. math::
#     D_{pq} = \mathrm{Re}[D_{pq}] + i\,\mathrm{Im}[D_{pq}],
#     \qquad p \leq q,
#
# and completing the matrix by Hermitian conjugation,
#
# .. math::
#     D_{qp} = D_{pq}^*, \qquad p < q.
#
# This procedure avoids redundant calculations and explicitly preserves
# the Hermitian character of the 1-RDM.


def pauli_string_fermion_operator(p, op_type):
    """
    Map a fermionic creation (0) or annihilation (1) operator at orbital p
    to its equivalent Pauli string via Jordan-Wigner transform.

    Parameters:
        p (int): orbital index
        op_type (int): 0 for creation, 1 for annihilation

    Returns:
        qml.operation.Tensor: Pauli string operator
    """
    f = (2 * op_type - 1) * 1j  # +i for creation, -i for annihilation
    Q_p = 0.5 * (qml.PauliX(p) + f * qml.PauliY(p))

    for i in range(p):
        Q_p = qml.PauliZ(i) @ Q_p

    return Q_p


def pauli_string_excitation_operator(p, q):
    """
    Constructs the Pauli string representation of the fermionic excitation operator a_p† a_q
    using the Jordan-Wigner transformation.

    This operator corresponds to exciting an electron from orbital q to orbital p
    (i.e., annihilation at q followed by creation at p).

    Args:
        p (int): Index of the target orbital (creation operator a_p†).
        q (int): Index of the source orbital (annihilation operator a_q).

    Returns:
        qml.operation.Tensor: The corresponding Pauli string operator representing a_p† a_q.

    Notes:
        - This is a non-Hermitian operator representing a single excitation.
        - To obtain a Hermitian excitation generator (for variational ansätze), you would
          need to combine this operator with its Hermitian conjugate (a_q† a_p).
        - The mapping relies on the Jordan-Wigner transformation, implemented via the
          `pauli_string_fermion_operator` function for single fermionic operators.
    """
    return pauli_string_fermion_operator(p, CREATION) @ pauli_string_fermion_operator(
        q, ANNIHILATION
    )


def rdm1_real_operator(p, q):
    """
    Constructs the Hermitian operator O_real = 1/2 (a_p† a_q + a_q† a_p),
    whose expectation value yields the real part of the 1-RDM element.

    Args:
        p (int): Index of the first orbital.
        q (int): Index of the second orbital.

    Returns: The Hermitian qubit operator corresponding to 1/2 (a_p† a_q + a_q† a_p).
    """
    op = pauli_string_excitation_operator(p, q)
    op_dag = pauli_string_excitation_operator(q, p)
    O_real = 0.5 * (op + op_dag)
    return O_real


def rdm1_imag_operator(p, q):
    """
    Constructs the Hermitian operator O_imag = (1/2i)(a_p† a_q - a_q† a_p),
    whose expectation value yields the imaginary part of the 1-RDM element.

    Args:
        p (int): Index of the first orbital.
        q (int): Index of the second orbital.

    Returns: The Hermitian qubit operator corresponding to (1/2i)(a_p† a_q - a_q† a_p).
    """
    op = pauli_string_excitation_operator(p, q)
    op_dag = pauli_string_excitation_operator(q, p)
    O_imag = 0.5j * (op - op_dag)  # same as (1/(2i))(op - op_dag)
    return O_imag


def rdm1_observables(qubits_active):
    """
    Construct all the observables needed to compute the 1-RDM.
    Diagonal elements correspond to occupation numbers,
    while off-diagonal elements are separated into real and imaginary components.

    Args: Number of active qubits in the system.

    Returns:
        observables (dict): Dictionary mapping keys to PennyLane observables.
        ordered_keys (list): Sorted list of keys for consistent ordering.
    """
    observables = {}
    for p in range(qubits_active):
        for q in range(p, qubits_active):
            observables[(p, q, "re")] = rdm1_real_operator(p, q)
            observables[(p, q, "im")] = rdm1_imag_operator(p, q)

    return observables, observables.keys()


##############################################################################
# The Molecular System
# -----------------------------------
# We begin by defining the molecular geometry of the system under study.
# In this example, we consider the :math:`\mathrm{CH}_5^+` molecule. It is
# specified in Cartesian coordinates (in Angstrom). To interface with quantum
# chemistry backends such as PySCF and PennyLane, the coordinates are
# converted from Angstrom to atomic units (Bohr).

geom = """
     C  0.0000000  0.1525520  0.0000000
     H  1.1165590  0.3217700  0.0000000
     H -0.5550270 -1.0611280  0.0000000
     H  0.3813020 -1.1309260  0.0000000
     H -0.4714170  0.4774870  0.9592110
     H -0.4714170  0.4774870 -0.9592110
     """

geom_data = geom.split()
symbols = tuple(geom_data[4 * i] for i in range(len(geom_data) // 4))
coordinates = tuple(geom_data[4 * i + 1 : 4 * i + 4] for i in range(len(geom_data) // 4))
coordinates = jnp.array(coordinates, dtype=float).reshape((-1, 3)) * Ang2Bohr

##############################################################################
# The Electronic Structure and Hamiltonian
# ----------------------------------------
# The next step is to formulate the electronic structure associated with the system.
# We begin by specifying the fundamental parameters: the total charge of the
# molecule, the spin multiplicity, the number of electrons, and the choice of
# basis set. ITo reduce the computational cost we use the minimal STO-3G basis
# and further restrict the problem by selecting an active space.
#
# Using PennyLane's quantum chemistry module, the molecular Hamiltonian is
# constructed in second quantization and then mapped to a qubit Hamiltonian
# via the Jordan–Wigner transformation. Finally, we prepare the Hartree–Fock
# reference state in the active space, which serves as the initial state for
# the variational quantum algorithm.

charge = 1
mult = 1
nelec = 10
basis = "STO-3G"
active_electrons = 2
active_orbitals = 2

# Define the Molecule in PennyLane
qml_mol = qml.qchem.Molecule(symbols, coordinates, charge=charge, mult=mult, basis_name=basis)

# Calculate the Hamiltonian
H_active, qubits_active = qml.qchem.molecular_hamiltonian(
    qml_mol, active_electrons=active_electrons, active_orbitals=active_orbitals
)

# Get the Hartree-Fock state
ref_active = qml.qchem.hf_state(active_electrons, qubits_active)

##############################################################################
# The Quantum Circuit and Observables
# -----------------------------------
# We now construct the variational quantum circuit based on a hardware-efficient
# ansatz (GateFabric), initialized with the Hartree–Fock reference state. In
# addition to the energy, our goal is to reconstruct the 1-RDM. This requires
# measuring a set of one-body operators, previously defined. For this reason, we
# define two quantum nodes (QNodes):
#
# - One for evaluating the expectation value of the Hamiltonian, yielding the energy.
# - Another for evaluating the expectation values of the 1-RDM observables,
#   which are later assembled into the full matrix.

# Define the device from PennyLane
dev = qml.device("default.qubit", wires=range(qubits_active))


@qml.qnode(dev, interface="jax")
def circuit(weights):
    """
    Executes a variational quantum circuit using an ansatz and computes
    the expectation value of the active space Hamiltonian.

    Args:
        weights (array): Variational parameters for the ansatz.

    Returns:
        float: Expectation value ⟨H_active⟩, i.e., the estimated energy of the quantum state.
    """
    qml.GateFabric(weights, wires=range(qubits_active), init_state=ref_active, include_pi=True)
    return qml.expval(H_active)


param_shape = qml.GateFabric.shape(n_layers=10, n_wires=qubits_active)

# Set observables to 1-RDM
observables, keys = rdm1_observables(qubits_active)
for k in keys:
    observables[k] = observables[k].simplify()


# Expectation value of the Hermitian one-body excitation operators
@qml.qnode(dev, interface="jax")
def rdm1_expectation(params):
    """
    Executes the variational quantum circuit and computes the expectation values of the
    one-body excitation operators needed to reconstruct the real part of the
    1-RDM for all orbital pairs (p, q).

    Args:
        params (array): Array of trainable parameters for the GateFabric ansatz.

    Returns:
        list[float]: A flat list of real-valued expectation values ⟨O_real⟩ for every
                     pair (p, q), corresponding to the real part of the 1-RDM.

    Key Steps:
        1. Prepares the quantum state using the GateFabric ansatz.
        2. For every pair of orbitals (p, q), construct the O_real and O_imag operators.
        3. Measures the expectation value.
        4. Returns all expectation values as a list.
    """
    qml.GateFabric(params, wires=range(qubits_active), init_state=ref_active, include_pi=True)
    return [qml.expval(observables[k]) for k in keys]


def compute_rdm1_active(params):
    """
    Computes the expectation values of the 1-RDM operators and organizes the results
    into a square matrix of shape (qubits_active × qubits_active).

    Args:
        params (array): Variational parameters for the quantum circuit.

    Returns:
        jnp.ndarray: The reconstructed real part of the one-electron 1-RDM
                     in the active space, as a (qubits_active, qubits_active) matrix.
    """
    values = rdm1_expectation(params)
    n = qubits_active
    rdm = jnp.zeros((n, n), dtype=complex)

    for val, key in zip(values, keys):
        p, q, part = key
        if part == "re":
            rdm = rdm.at[p, q].add(val)
            if p != q:
                rdm = rdm.at[q, p].add(val)
        else:  # Imaginary component
            rdm = rdm.at[p, q].add(1j * val)
            if p != q:
                rdm = rdm.at[q, p].add(-1j * val)

    return rdm


##############################################################################
# The Cost Function
# -----------------------------------
# The cost function in the VQE-LD approach is defined as a combination of
# two terms:
#
# (i) the energy expectation value, and
# (ii) a penalty term that enforces consistency and convergence of the 1-RDM across optimization steps.
#
# The 1-RDM term is constructed as the root-mean-square deviation (RMSD)
# between the 1-RDM of consecutive iterations.
# This promotes smooth convergence of the density matrix and mitigates
# situations where energy convergence alone leads to inaccurate
# electronic properties.


def term_energy(params):
    return jnp.real(circuit(params))


def term_rdm1(params, rdm1_prev):
    rdm1 = compute_rdm1_active(params)
    diff = rdm1 - rdm1_prev
    return jnp.sqrt(jnp.mean(jnp.abs(diff) ** 2))


grad_energy_fn = jax.jit(jax.grad(term_energy))
grad_rdm1_fn = jax.jit(lambda p, r_prev: jax.grad(term_rdm1, argnums=0)(p, r_prev))

##############################################################################
# The Parameter Optimization Loop
# -----------------------------------
# We define the parameters and hyperparameters that control the training process.
# This includes the total number of optimization steps, convergence thresholds
# for both the energy and the 1-RDM, and the initial values of the variational
# parameters. The parameters are typically initialized to zero, corresponding
# to the Hartree–Fock reference state when using structured ansätze.
#
# We also choose a classical optimizer, in this case, stochastic gradient descent
# (SGD), which updates the circuit parameters based on the combined gradients
# of the energy and the 1-RDM term.
#
# To monitor convergence, we keep track of the energy and the 1-RDM throughout the optimization.
# We also store the 1-RDM from the previous iteration. This allows us to evaluate
# both the energy difference and the variation in the 1-RDM between successive
# steps. Additionally, a scaling factor is introduced to control the relative
# influence of the 1-RDM term.

# Iterations
total_iterations = 50000

# Threshold
Etol = 1e-6  # for energy
RDM1_tol = 1e-6  # for rmd1

# Initial value of variational parameters
params = jnp.zeros(param_shape)

# Classical optimizer
opt = optax.sgd(0.4)

# Keep track of the energies
energies = [circuit(params)]

# Set parameters
opt_state = opt.init(params)
current_opt = opt
rdm1_active_prev = compute_rdm1_active(jnp.zeros(param_shape)).copy()
rdm1_active = compute_rdm1_active(params)
rate = 0.6

##############################################################################
# The VQE-LD optimization loop
# -----------------------------------
# We now implement the core optimization loop of the VQE-LD algorithm.
# The procedure can be summarized as follows:
#
# 1. Compute the gradient of the energy with respect to the circuit parameters.
# 2. Compute the gradient of the 1-RDM penalty term, which depends on the
#    deviation from the previous iteration.
# 3. Estimate the norm of each gradient and define a dynamic weight that
#    balances their relative contributions.
# 4. Combine the gradients into a single update direction.
# 5. Apply the optimizer step to update the variational parameters.
#
# The adaptive weighting plays a central role in the VQE-LD method. By scaling
# the 1-RDM gradient according to the ratio of gradient norms, the algorithm
# ensures that both terms contribute meaningfully throughout the optimization,
# avoiding regimes where either energy minimization or density consistency
# dominates.
#
# After each update, we evaluate the new energy and reconstruct the 1-RDM.
# Convergence is assessed using two criteria:
#
# - The change in energy between successive iterations.
# - The root-mean-square deviation (RMSD) between consecutive 1-RDMs.
#
# The optimization terminates only when both quantities fall below predefined
# thresholds, ensuring not only energetic convergence but also stability of
# the underlying electronic structure.

f = open("log.txt", "a", encoding="utf-8")

for n in range(1, total_iterations + 1):
    # Gradient of the energy
    grad_E = grad_energy_fn(params)

    # Gradient of the RDM1 -> Only apply the RDM1 penalty from the second iteration
    # If it's the first iteration, grad_R is a zero vector
    if n > 1:
        grad_R = grad_rdm1_fn(params, rdm1_active_prev)
    else:
        grad_R = jnp.zeros_like(params)

    # Calculate the norm of each gradient
    norm_grad_E = jnp.linalg.norm(grad_E)
    norm_grad_R = jnp.linalg.norm(grad_R)

    # Define automatic weight for RDM1.
    # The goal is to balance the magnitude of grad_R with grad_E
    if norm_grad_R == 0:
        w_rdm1_auto = (norm_grad_E) * rate
    else:
        w_rdm1_auto = (norm_grad_E / (norm_grad_R)) * rate

    # Combine gradients
    grads = grad_E + w_rdm1_auto * grad_R

    # Update circuit parameters
    updates, opt_state = current_opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    # Compute updated energy and RDM1
    energy = circuit(params)
    rdm1_trial = compute_rdm1_active(params)  # new RDM1

    # Update references
    energies.append(energy)
    rdm1_active_prev = rdm1_active.copy()
    rdm1_active = rdm1_trial.copy()

    # Difference in RDM1
    diff = rdm1_trial - rdm1_active_prev
    rdm1_diff = jnp.sqrt(jnp.mean(diff**2))

    grad_norm = jnp.linalg.norm(grads)
    print(f"Iter {n} | Energy: {energy:.9f} Ha | ΔRDM1: {rdm1_diff:.6e} | ‖∇C‖: {grad_norm:.3e}")
    print(
        f"Iter {n} | Energy: {energy:.9f} Ha | ΔRDM1: {rdm1_diff:.6e} | ‖∇C‖: {grad_norm:.3e}",
        file=f,
    )

    # Convergence criterion
    if len(energies) > 1:
        delta_E = jnp.abs(energies[-1] - energies[-2])
        # If both energy and RDM1 converge, stop the loop
        if delta_E < Etol and rdm1_diff < RDM1_tol:
            print("Complete convergence in energy and RDM1.")
            break
f.close()

print(f"Iter {n} | Energy: {energy:.9f} Ha | ΔRDM1: {rdm1_diff:.6e} | ‖∇C‖: {grad_norm:.3e}")

##############################################################################
# In this case, the VQE-LD algorithm converges to an energy of −39.917589475 Ha,
# while the CASCI calculation in the (4,4)-active space yields −39.91925976 Ha.
# Therefore, our approach achieves chemical accuracy with respect to this high-quality
# reference method. In contrast, a standard VQE optimization results in an error of
# 2.61 x :math:`10^{-1}` Ha relative to the CASCI value, which is well above the chemical
# accuracy threshold of 1 kcal/mol (1.6 x :math:`10^{-3}` Ha).
#
# Conclusion
# -----------------------------------
# In this tutorial, we presented the VQE-LD algorithm, an extension of the
# Variational Quantum Eigensolver that incorporates information from the
# one-particle reduced density matrix (1-RDM) directly into the optimization
# process.
#
# By augmenting the standard energy-based cost function with a term that
# enforces consistency of the 1-RDM, the method promotes not only convergence
# in energy but also stability in the underlying electronic structure.
# This is particularly important for accurately computing molecular properties
# that are more sensitive to fluctuations of the electronic density.
#
# A key feature of VQE-LD is the use of an adaptive weighting strategy based
# on gradient norms, which dynamically balances the contributions of the
# energy and 1-RDM terms throughout the optimization. This avoids the need
# for manual hyperparameter tuning and leads to a more robust optimization
# landscape.
#
# The framework presented here is general and can be extended in several
# directions, including larger active spaces, more expressive ansätze, and
# higher-order reduced density matrices. It also opens the door to improving
# estimation of observables beyond total energy, which is a central goal in
# quantum computational chemistry.
#
# Overall, VQE-LD provides a promising route toward enhancing the reliability
# of near-term quantum simulations for molecular systems.

##############################################################################
# .. _vqe_ld_references:
#
#
# References
# ----------
#
# .. [#lima2026]
#
#    Amanda Marques de Lima, Erico Souza Teixeira,
#    Eivson Darlivam Rodrigues de Aguiar Silva, Ricardo Luiz Longo,
#    *Improving Energy and Molecular Properties by Convergence of the
#    One-Particle Reduced Density Matrix in Variational Quantum
#    Eigensolvers (VQE)*. Journal of Computational Chemistry 47 (2026).
#    https://doi.org/10.1002/jcc.70289
#
