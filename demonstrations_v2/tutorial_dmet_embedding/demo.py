r"""Density Matrix Embedding Theory (DMET)
=========================================
Computer simulations of materials are very challenging. Mean field methods are inexpensive, but they are unreliable and inconsistent in describing strongly correlated systems. More accurate simulations can be obtained with wavefunction approaches such as configuration interaction or coupled cluster, but these are extremely expensive, becoming prohibitive even for relatively small systems.

Embedding theories provide a path to simulate materials with a balance of accuracy and efficiency.
The core idea is to divide the system
into two parts: an impurity, which is a strongly correlated subsystem that requires a high accuracy
description, and an environment, which can be treated with a more approximate but computationally
efficient level of theory.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_how_to_build_dmet_hamiltonian.png
    :align: center
    :width: 70%
    :target: javascript:void(0)
"""

#############################################
# In this demo, we will describe density matrix embedding theory (DMET) [#SWouters]_, a
# wavefunction-based approach that embeds the ground state density matrix. We provide a brief
# introduction of the method and demonstrate how to run DMET calculations to construct a Hamiltonian
# that can be used in quantum algorithms using PennyLane.
#
# Theory
# ------
# The ground state wavefunction for an embedded system composed of an impurity and an environment can be
# represented as [#SWouters]_:
#
# .. math::
#
#      | \Psi \rangle = \sum_i^{N_I} \sum_j^{N_E} \psi_{ij} | I_i \rangle | E_j \rangle,
#
# where :math:`I_i` and :math:`E_j` are respectively the basis states of the **impurity** :math:`I` and **environment**
# :math:`E`, :math:`\psi_{ij}` is the matrix of coefficients, and :math:`N` is the
# number of sites, e.g., orbitals. The key idea in DMET is to perform a singular value decomposition
# of the coefficient matrix :math:`\psi_{ij} = \sum_{\alpha} U_{i \alpha} \lambda_{\alpha} V_{\alpha j}`
# and rearrange the wavefunctions as:
#
# .. math::
#
#      | \Psi \rangle = \sum_{\alpha}^{N} \lambda_{\alpha} | A_{\alpha} \rangle | B_{\alpha} \rangle,
#
# where :math:`A_{\alpha} = \sum_i U_{i \alpha} | I_i \rangle` are states obtained from rotations of
# :math:`I_i` to a new basis, and :math:`B_{\alpha} = \sum_j V_{j \alpha} | E_j \rangle` are **bath states**
# representing the portion of the environment that interacts with the impurity [#Mineh]_. Note
# that the number of bath states is equal to the number of impurity states, regardless of the size
# of the environment. This representation is simply the `Schmidt decomposition <https://arxiv.org/html/2411.05703v1>`_
# of the system wave function.
#
# We are now able to project the full Hamiltonian to the space of impurity and bath states, known as
# the embedding space:
#
# .. math::
#
#      \hat{H}_{\text{emb}} = P^{\dagger} \hat{H} P,
#
# where :math:`P = \sum_{\alpha \beta} | A_{\alpha} B_{\beta} \rangle \langle A_{\alpha} B_{\beta}|`
# is a projection operator. A key point about this representation is that the wavefunction,
# :math:`|\Psi \rangle`, is the ground state of both the full system Hamiltonian :math:`\hat{H}` and
# the smaller embedded Hamiltonian :math:`\hat{H}_{\text{emb}}` [#Mineh]_.
#
# Note that the Schmidt decomposition requires a priori knowledge of the ground-state wavefunction. To alleviate
# this, DMET operates through a systematic iterative approach, starting with a mean field description
# of the wavefunction and refining it through feedback from solutions of the impurity Hamiltonian, as we describe below.
#
# Implementation
# --------------
# The DMET procedure starts by getting an approximate description of the system's wavefunction.
# Subsequently, this approximate wavefunction is partitioned with the Schmidt decomposition to get
# the impurity and bath orbitals. These orbitals are then employed to define an approximate projector
# :math:`P`, which is used to construct the embedded Hamiltonian. Then, accurate methods such as
# quantum algorithms are employed
# to find the energy spectrum of the embedded Hamiltonian. The results are used to
# re-construct the projector and the process is repeated until the wavefunction converges.
#
# Let's now implement these steps for a linear chain of 8 Hydrogen atoms. We will use the
# programs PySCF [#pyscf]_ and libDMET [#libdmet]_ [#libdmet2]_ which can be installed with
#
# .. code-block:: python
#
#    pip install pyscf
#    pip install git+https://github.com/gkclab/libdmet_preview.git@main
#
# Constructing the system
# ^^^^^^^^^^^^^^^^^^^^^^^
# We begin by defining a finite system containing a hydrogen chain with 8 atoms using PySCF. This is
# done by creating a ``Cell`` object containing the Hydrogen atoms. We construct the system to have
# a central :math:`H_4` chain with a uniform :math:`0.75` Å :math:`H-H` bond length, flanked by two
# :math:`H_2` molecules at its endpoints.
#
# .. figure:: ../_static/demonstration_assets/dmet/H8.png
#    :align: center
#    :width: 70%
#    :target: javascript:void(0)
#
# Then, we construct a ``Lattice`` object using libDMET and associate it with the defined cell.
#
# .. code-block:: python
#
#    import numpy as np
#    from pyscf.pbc import gto, df, scf, tools
#    from libdmet.system import lattice
#
#    cell = gto.Cell()
#    cell.unit = 'Angstrom'
#    cell.a = ''' 15.0     0.0     0.0
#                  0.0    15.0     0.0
#                  0.0     0.0    15.0 '''  # lattice vectors of the unit cell
#
#    cell.atom = ''' H 0.0  0.0   0.0
#                    H 0.0  0.0   0.75
#                    H 0.0  0.0   3.0
#                    H 0.0  0.0   3.75
#                    H 0.0  0.0   4.5
#                    H 0.0  0.0   5.25
#                    H 0.0  0.0   7.5
#                    H 0.0  0.0   8.25'''  # coordinates of atoms in unit cell
#
#    cell.basis = '321g'
#    cell.build()
#
#    kmesh = [1, 1, 1]                     # number of k-points in xyz direction
#
#    lattice = lattice.Lattice(cell, kmesh)
#    filling = cell.nelectron / (lattice.nscsites * 2.0)
#
#
# Performing mean field calculations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We can now perform a mean field calculation on the whole system through Hartree-Fock with density-fitted
# integrals, following the procedures outlined in the `PySCF documentation <https://pyscf.org/user/pbc/df.html>`_.
#
# .. code-block:: python
#
#    gdf = df.GDF(cell, lattice.kpts)
#    gdf._cderi_to_save = 'gdf_ints.h5' # output file for the integrals
#    gdf.build()                        # compute the integrals
#    kmf = scf.KRHF(cell, lattice.kpts, exxdiv=None).density_fit()
#    kmf.with_df = gdf                  # use density-fitted integrals
#    kmf.with_df._cderi = 'gdf_ints.h5' # input file for the integrals
#    kmf.kernel()                       # run Hartree-Fock
#
# This provides us with an approximate description of the whole system.
#
# Partitioning the orbital space
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Now that we have a full description of our system, we can start obtaining the impurity and
# bath orbitals. We will use a localized orbital basis which provides a convenient way to understand
# the contribution of each atom to the properties of the full system. We can use any localized basis
# such as molecular orbitals (MO) or intrinsic atomic orbitals (IAO) [#SWouters]_. Here, we rotate
# the one-electron and two-electron integrals into the IAO basis.
#
# .. code-block:: python
#
#    from libdmet.basis_transform import make_basis
#
#    ao_to_iao_transform, _, _ = make_basis.get_C_ao_lo_iao(
#    lattice, kmf, full_return=True)
#    ao_to_lo_transform = lattice.symmetrize_lo(ao_to_iao_transform)
#    lattice.set_Ham(kmf, gdf, ao_to_lo_transform, eri_symmetry=4) # rotate to IAO basis
#
# Next, we identify the orbital labels for each atom in the unit cell and define the impurity and
# bath by looking at the labels. We achieve this by utilizing a minimal basis
# to categorize orbitals: those present in the minimal basis are identified as valence,
# while the remaining orbitals are deemed virtual. In this example, we choose to keep the :math:`1s` orbitals in the
# central :math:`H_4` chain in the impurity, while the bath contains the :math:`2s` orbitals and
# the orbitals belonging to the terminal Hydrogen molecules form the environment.
#
# .. code-block:: python
#
#    from libdmet.lo.iao import get_labels, get_idx
#
#    orb_labels, val_labels, virt_labels = get_labels(cell)
#    imp_idx = get_idx(val_labels, atom_num=[2,3,4,5])
#    bath_idx = get_idx(virt_labels, atom_num=[2,3,4,5], offset=len(val_labels))
#    core_idx = []
#    lattice.set_val_virt_core(imp_idx, bath_idx, core_idx)
#
# Now that we have a description of our impurity, bath and environment orbitals, we can implement
# our DMET simulation.
#
# Self-Consistent DMET
# ^^^^^^^^^^^^^^^^^^^^
# The DMET calculations are performed by repeating four steps iteratively:
#
# 1) Construct an impurity Hamiltonian.
# 2) Solve the impurity Hamiltonian.
# 3) Compute the full system energy.
# 4) Update the interactions between the impurity and its environment.
#
# To simplify the calculations, we create dedicated functions for each step and implement them in a
# self-consistent loop. If we only perform one iteration, the process is referred to as single-shot DMET.
#
# We now construct the impurity Hamiltonian.
#
# .. code-block:: python
#
#    def construct_impurity_hamiltonian(lattice, v_cor, filling, mu, last_dmu, int_bath=True):
#        r"""A function to construct the impurity Hamiltonian
#
#        Args:
#            lattice: Lattice object containing information about the system
#            v_core: correlation potential
#            filling: average number of electrons occupying the orbitals
#            mu: chemical potential
#            last_dmu: change in chemical potential from last iterations
#            int_bath: Flag to determine whether we are using an interactive bath
#
#        Returns:
#            rho: mean field density matrix
#            mu: chemical potential
#            scf_result: object containing the results of mean field calculation
#            impHam: impurity Hamiltonian
#            basis: rotation matrix for embedding basis
#        """
#
#        rho, mu, scf_result = dmet.HartreeFock(lattice, v_corr, filling, mu,
#                                        ires=True)
#        impHam, _, basis = dmet.ConstructImpHam(lattice, rho, v_corr, int_bath=int_bath)
#        impHam = dmet.apply_dmu(lattice, impHam, basis, last_dmu)
#
#        return rho, mu, scf_result, impHam, basis
#
# Next, we solve this impurity Hamiltonian with a high-level method. The following function defines
# the electronic structure solver for the impurity, provides an initial point for the calculation,
# and passes the ``Lattice`` information to the solver. The solver then calculates the energy and
# density matrix for the impurity.
#
# .. code-block:: python
#
#    def solve_impurity_hamiltonian(lattice, cell, basis, impHam, filling_imp, last_dmu, scf_result):
#        r"""A function to solve impurity Hamiltonian
#
#        Args:
#            lattice: Lattice object containing information about the system
#            cell: Cell object containing information about the unit cell
#            basis: rotation matrix for embedding basis
#            impHam: impurity Hamiltonian
#            last_dmu: change in chemical potential from previous iterations
#            scf_result: object containing the results of mean field calculation
#
#        Returns:
#            rho_emb: density matrix for embedded system
#            energy_emb: energy for embedded system
#            impHam: impurity Hamiltonian
#            last_dmu: change in chemical potential from last iterations
#            solver_info: a list containing information about the solver
#        """
#
#        solver = dmet.impurity_solver.FCI(restricted=True, tol=1e-8) # impurity solver
#        basis_k = lattice.R2k_basis(basis) # basis in k-space
#
#        solver_args = {"nelec": min((lattice.ncore+lattice.nval)*2, lattice.nkpts*cell.nelectron), \
#                       "dm0": dmet.foldRho_k(scf_result["rho_k"], basis_k)}
#
#        rho_emb, energy_emb, impHam, dmu = dmet.SolveImpHam_with_fitting(lattice, filling_imp,
#                                            impHam, basis, solver, solver_args=solver_args)
#
#        last_dmu += dmu
#        solver_info = [solver, solver_args]
#        return rho_emb, energy_emb, impHam, last_dmu, solver_info
#
# Now we include the effect of the environment in the expectation value. So we define
# a function which returns the density matrix and energy for the whole system.
#
# .. code-block:: python
#
#    def solve_full_system(lattice, rho_emb, energy_emb, basis, impHam, last_dmu, solver_info):
#        r"""A function to solve impurity Hamiltonian
#
#        Args:
#            lattice: Lattice object containing information about the system
#            rho_emb: density matrix for embedded system
#            energy_emb: energy for embedded system
#            basis: rotation matrix for embedding basis
#            impHam: impurity Hamiltonian
#            last_dmu: change in chemical potential from last iterations
#            solver_info: a list containing information about the solver
#
#        Returns:
#            energy_imp_fci: Ground state energy of the impurity
#            energy: Ground state energy of the full system
#        """
#
#        from libdmet.routine.slater import get_E_dmet_HF
#
#        rho_imp, energy_imp, nelec_imp = \
#                dmet.transformResults(rho_emb, energy_emb, basis, impHam, \
#                lattice=lattice, last_dmu=last_dmu, int_bath=True, \
#                             solver=solver_info[0], solver_args=solver_info[1])
#        energy_imp_fci = energy_imp * lattice.nscsites # energy of impurity at FCI level
#        energy_imp_scf = get_E_dmet_HF(basis, lattice, impHam, last_dmu, solver_info[0])
#        energy = kmf.e_tot - kmf.energy_nuc() - energy_imp_scf + energy_imp_fci
#
#        return energy_imp_fci, energy
#
# Note here that the effect of the environment included in this step is at the mean field level. So
# if we stop the iteration here, the results will be that of the single-shot DMET.
#
# In the self-consistent DMET, the interaction between the environment and the impurity is improved
# iteratively. To do that, a correlation potential is introduced to account for the interactions
# between the impurity and its environment, which can be represented in terms of creation,
# :math:`a^{\dagger}`, and annihilation, :math:`a`, operators as
#
# .. math::
#
#      C = \sum_{kl} u_{kl} a_k^{\dagger} a_l.
#
# We start with an initial guess of zero for the coefficient matrix :math:`u` and optimize it by
# minimizing the difference between density matrices obtained from the mean field Hamiltonian and
# the impurity Hamiltonian.
#
# We define the following functions to initialize the correlation potential and optimize it.
#
# .. code-block:: python
#
#    def initialize_vcorr(lattice):
#        r"""A function to initialize the correlation potential
#
#        Args:
#            lattice: Lattice object containing information about the system
#
#        Returns:
#            Initial correlation potential
#        """
#
#        v_corr = dmet.VcorLocal(restricted=True, bogoliubov=False,
#                nscsites=lattice.nscsites, idx_range=lattice.imp_idx)
#        v_corr.assign(np.zeros((2, lattice.nscsites, lattice.nscsites)))
#        return v_corr
#
#    def fit_correlation_potential(rho_emb, lattice, basis, v_corr):
#        r"""A function to solve impurity Hamiltonian
#
#        Args:
#            rho_emb: density matrix for embedded system
#            lattice: Lattice object containing information about the system
#            basis: rotation matrix for embedding basis
#            v_corr: correlation potential
#
#        Returns:
#            v_corr: correlation potential
#            dVcorr_per_ele: change in correlation potential per electron
#        """
#
#        vcorr_new, _ = dmet.FitVcor(rho_emb, lattice, basis, \
#                    v_corr, beta=np.inf, filling=filling, MaxIter1=300, MaxIter2=0)
#
#        dVcorr_per_ele = np.max(np.abs(vcorr_new.param - v_corr.param))
#        v_corr.update(vcorr_new.param)
#        return v_corr, dVcorr_per_ele
#
# Now, we have defined all the ingredients of DMET. We can set up the self-consistency loop to get
# the final results.
#
# We set up the loop by defining the maximum number of iterations and a convergence criteria. We use
# both energy and correlation potential as our convergence parameters, so we define the initial
# values and convergence tolerance for both. Also, since dividing the system into multiple impurities
# might lead to the wrong number of electrons, we define and check the chemical potential :math:`\mu`
# as well.
#
# .. code-block:: python
#
#    import libdmet.dmet.Hubbard as dmet
#
#    max_iter = 20                     # maximum number of iterations
#    e_old = 0.0                       # initial value of energy
#    v_corr = initialize_vcorr(lattice)# initial value of correlation potential
#    dVcorr_per_ele = None             # initial value of correlation potential per electron
#    vcorr_tol = 1.0e-5                # tolerance for correlation potential convergence
#    energy_tol = 1.0e-5               # tolerance for energy convergence
#    mu = 0                            # initial chemical potential
#    last_dmu = 0.0                    # change in chemical potential
#
# Now we set up the iterations in a loop. Note that defining an impurity which is a part
# of the unit cell, necessitates readjusting of the filling value for solution of impurity
# Hamiltonian. This filling value represents the average electron occupation, which scales
# proportional to the fraction of the unit cell included, while taking into account the different
# electronic species. For example, we are using half the number of Hydrogens inside the impurity,
# therefore, the filling becomes half as well.
#
# .. code-block:: python
#
#    for i in range(max_iter):
#        rho, mu, scf_result, impHam, basis = construct_impurity_hamiltonian(lattice,
#                           v_corr, filling, mu, last_dmu) # construct impurity Hamiltonian
#        filling_imp = filling * 0.5
#        # solve impurity Hamiltonian
#        rho_emb, energy_emb, impHam, last_dmu, solver_info = solve_impurity_hamiltonian(lattice,
#                                      cell, basis, impHam, filling_imp, last_dmu, scf_result)
#        # include the environment interactions
#        energy_imp, energy = solve_full_system(lattice, rho_emb, energy_emb, basis, impHam,
#                           last_dmu, solver_info)
#        # fit correlation potential
#        v_corr, dVcorr_per_ele = fit_correlation_potential(rho_emb,
#                                         lattice, basis, v_corr)
#
#        dE = energy_imp - e_old
#        e_old = energy_imp
#        if dVcorr_per_ele < vcorr_tol and abs(dE) < energy_tol:
#            print("DMET Converged")
#            print("DMET Energy per cell: ", energy)
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     DMET Converged
#     DMET Energy per cell:  -8.203518641937336
#
# Quantum DMET
# ^^^^^^^^^^^^
# Our implementation of DMET used full configuration interaction (FCI) to accurately treat the
# impurity subsystem. The cost of using a high-level solver such as FCI increases exponentially with
# the system size which limits the number of orbitals we can have in the impurity. One way to solve
# this problem is to use a quantum algorithm as our accurate solver. We now `map our impurity
# Hamiltonian to a qubit Hamiltonian <https://pennylane.ai/qml/demos/tutorial_mapping/>`__
# that can be used in a quantum algorithm using PennyLane.
#
# The Hamiltonian object we generated above contains one-body and two-body integrals along with the
# nuclear repulsion energy which can be accessed and used to construct the qubit Hamiltonian. We
# first extract the information we need.
#
# .. code-block:: python
#
#    from pyscf import ao2mo
#    norb = impHam.norb
#    h1 = impHam.H1["cd"]
#    h2 = impHam.H2["ccdd"][0]
#    h2 = ao2mo.restore(1, h2, norb) # get the correct shape based on permutation symmetry
#
# Now we generate the qubit Hamiltonian in PennyLane.
#
# .. code-block:: python
#
#    from pennylane.qchem import one_particle, two_particle, observable
#
#    one_elec = one_particle(h1[0])
#    two_elec = two_particle(np.swapaxes(h2, 1, 3)) # swap to physicist's notation
#    qubit_op = observable([one_elec,two_elec], mapping="jordan_wigner")
#    print("Qubit Hamiltonian: ", qubit_op)
#
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#    Qubit Hamiltonian:  0.6230307293797223 * I(0) + -0.4700529413255728 * Z(0) + -0.4700529413255728 * Z(1) + -0.21375048863111926 * (Y(0) @ Z(1) @ Y(2)) + ...
#
# This Hamiltonian can be used in a quantum algorithm such as `quantum phase estimation <https://pennylane.ai/qml/demos/tutorial_qpe/>`__.
# We can get the ground state energy for the system by solving for the full system as done above in the
# self-consistency loop using the ``solve_full_system`` function. The qubit Hamiltonian is
# particularly relevant for a hybrid version of DMET, where classical mean field calculations are
# coupled with a post-HF classical solver for the iterative
# self-consistency between the impurity and the environment and treat the resulting converged
# impurity Hamiltonian with a quantum algorithm.
#
# Conclusion
# ----------
# Density matrix embedding theory is a robust method designed to tackle simulation of complex
# systems by partitioning them into manageable subsystems. It is particularly well suited for
# studying the ground state properties of strongly correlated materials and molecules. DMET offers a
# compelling alternative to dynamic quantum embedding schemes such as dynamic mean field theory. By
# employing the density matrix for embedding instead of the Green's function and
# utilizing the Schmidt decomposition to get a limited number of bath orbitals, DMET achieves a
# significant reduction in computational resources. Furthermore, a major strength lies in its
# compatibility with quantum algorithms, enabling the accurate study of smaller, strongly correlated
# subsystems on quantum computers while the environment is studied classically. Its successful
# application to a wide range of strongly correlated molecular and periodic systems underscores its
# power and versatility in electronic structure theory, paving the way for hybrid quantum-classical
# simulations of challenging materials [#DMETQC]_. Potential future extensions could involve applying
# these techniques to the wider field of spectroscopy, which probes the excited-state response of materials.
# A relevant example is the investigation of core-level excited states in `X-ray Absorption Spectroscopy (XAS)
# <https://pennylane.ai/qml/demos/tutorial_xas/>`__,
# a domain which can be explored using quantum algorithms.
#
#
# References
# ----------
#
# .. [#SWouters]
#    S. Wouters, C. A. Jiménez-Hoyos, *et al.*,
#    "A practical guide to density matrix embedding theory in quantum chemistry",
#    `arXiv:1603.08443 <https://arxiv.org/abs/1603.08443>`__.
#
# .. [#Mineh]
#    Lana Mineh, Ashley Montanaro,
#    "Solving the Hubbard model using density matrix embedding theory and the variational quantum eigensolver",
#    `arXiv:2108.08611 <https://arxiv.org/abs/2108.08611>`__.
#
# .. [#pyscf]
#     Q. Sun, X. Zhang, *et al.*,
#     "Recent developments in the PySCF program package",
#     `arXiv:2002.12531 <https://arxiv.org/abs/2002.12531>`__.
#
# .. [#libdmet]
#     Z. Cui, T. Zhu, *et al.*,
#     "Efficient Implementation of Ab Initio Quantum Embedding in Periodic Systems: Density Matrix Embedding Theory",
#     `arXiv:1909.08596 <https://arxiv.org/abs/1909.08596>`__.
#
# .. [#libdmet2]
#    T. Zhu, Z. Cui, *et al.*,
#    "Efficient Formulation of Ab Initio Quantum Embedding in Periodic Systems: Dynamical Mean-Field Theory",
#    `arXiv:1909.08592 <https://arxiv.org/abs/1909.08592>`__.
#
# .. [#DMETQC]
#    C. Cao, J. Sun, *et al.*,
#    "Ab initio simulation of complex solids on a quantum computer with orbital-based multifragment density matrix embedding",
#    `arXiv:2209.03202 <https://arxiv.org/abs/2209.03202>`__.
#
