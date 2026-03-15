r"""How to build spin Hamiltonians
==================================
Systems of interacting spins provide simple but powerful models for studying problems in physics,
chemistry, and quantum computing. PennyLane offers a comprehensive set of tools that enables users
to intuitively construct a broad range of spin Hamiltonians. Here we show you how to use these tools
to easily construct spin Hamiltonians for the `Fermi–Hubbard model <https://en.wikipedia.org/wiki/Hubbard_model>`__,
the `Heisenberg model <https://en.wikipedia.org/wiki/Quantum_Heisenberg_model>`__,
the `transverse-field Ising model <https://en.wikipedia.org/wiki/Transverse-field_Ising_model>`__,
`Kitaev's honeycomb model <https://arxiv.org/abs/cond-mat/0506438>`__,
the `Haldane model <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.61.2015>`__,
the `Emery model <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.58.2794>`__,
and more. And you can also already explore some of these models in detail using `PennyLane Spin Systems Datasets <https://pennylane.ai/datasets/collection/qspin>`__!

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_how_to_build_spin_hamiltonians.png
    :align: center
    :width: 70%
    :target: javascript:void(0)
"""

######################################################################
# Hamiltonian templates
# ---------------------
# PennyLane provides a set of built-in
# `functions <https://docs.pennylane.ai/en/latest/code/qml_spin.html#hamiltonian-functions>`__
# in the `qml.spin <https://docs.pennylane.ai/en/latest/code/qml_spin.html>`__ module for
# constructing spin Hamiltonians with minimal input needed from the user: we only need to specify
# the lattice that describes spin sites and the parameters that describe the interactions in our
# system. Let’s look at some examples for the models that are currently supported in PennyLane.
#
# Fermi–Hubbard model
# ^^^^^^^^^^^^^^^^^^^
# The `Fermi–Hubbard model Hamiltonian <https://docs.pennylane.ai/en/latest/code/api/pennylane.spin.fermi_hubbard.html>`__
# has a kinetic energy component, which is parameterized by a hopping parameter :math:`t`, and a
# potential energy component which is parameterized by the on-site interaction strength, :math:`U`:
#
# .. math::
#
#     H = -t\sum_{\left< i,j \right>, \sigma} c_{i\sigma}^{\dagger}c_{j\sigma} + U\sum_{i}n_{i \uparrow} n_{i\downarrow}.
#
# The terms :math:`c^{\dagger}, c` are the creation and annihilation operators,
# :math:`\left< i,j \right>` represents the indices of neighbouring spins, :math:`\sigma` is the
# spin degree of freedom, and :math:`n_{i \uparrow}, n_{i \downarrow}` are the number operators for
# the spin-up and spin-down fermions at site :math:`i`, denoted by :math:`0` and :math:`1`
# respectively. This model is often used as a simplified model to investigate superconductivity.
#
# The Fermi–Hubbard Hamiltonian can be
# constructed in PennyLane by passing the hopping and interaction parameters to the
# :func:`~.pennylane.spin.fermi_hubbard` function. We also need to specify the shape of the lattice
# that describes the positions of the spin sites. We will show an example here, and the full list of
# supported lattice shapes is
# provided in the :func:`~.pennylane.spin.generate_lattice` documentation.
# 
# We can also define the
# number of lattice cells we would like to include in our Hamiltonian as a list of integers for
# :math:`x, y, z` directions, depending on the lattice shape. Here we generate the Fermi–Hubbard
# Hamiltonian on a ``square`` lattice of shape :math:`2 \times 2`. The ``square`` lattice is
# constructed from unit cells that contain only one site such that we will have
# :math:`2 \times 2 = 4` sites in total. We will provide more details on constructing lattices in
# the following sections.

import pennylane as qml
qml.capture.enable()

n_cells = [2, 2]
hopping = 0.2
onsite = 0.3

hamiltonian = qml.spin.fermi_hubbard('square', n_cells, hopping, onsite)
print('Hamiltonian:\n')
hamiltonian

######################################################################
# Let's also visualize the square lattice we created. To do that, we need to
# create a simple plotting function, as well as the helper function
# :func:`~.pennylane.spin.generate_lattice`, which you will learn more about in the next sections.

import matplotlib.pyplot as plt

def plot(lattice, figsize=None, showlabel=True):

    # initialize the plot
    if not figsize:
        figsize = lattice.n_cells[::-1]

    plt.figure(figsize=figsize)

    # get lattice nodes and edges and plot them
    nodes = lattice.lattice_points

    for edge in lattice.edges:
        start_index, end_index, color = edge
        start_pos, end_pos = nodes[start_index], nodes[end_index]

        x_axis = [start_pos[0], end_pos[0]]
        y_axis = [start_pos[1], end_pos[1]]
        plt.plot(x_axis, y_axis, color='gold')

        plt.scatter(nodes[:,0], nodes[:,1], color='dodgerblue', s=100)

        if showlabel:
            for index, pos in enumerate(nodes):
                plt.text(pos[0]-0.2, pos[1]+0.1, str(index), color='gray')

    plt.axis("off")
    plt.show()

lattice = qml.spin.generate_lattice('square', n_cells)
plot(lattice)

######################################################################
# We currently support the following in-built lattice shapes: ``chain``, ``square``,
# ``rectangle``, ``triangle``, ``honeycomb``,  ``kagome``, ``lieb``, ``cubic``, ``bcc``, ``fcc``
# and ``diamond``. More details are provided
# `here <https://docs.pennylane.ai/en/latest/code/api/pennylane.spin.generate_lattice.html>`__.
#
# Heisenberg model
# ^^^^^^^^^^^^^^^^
# The `Heisenberg model Hamiltonian <https://docs.pennylane.ai/en/latest/code/api/pennylane.spin.heisenberg.html>`__
# is defined as
#
# .. math::
#
#     H = J\sum_{ < i, j >}(\sigma_i ^ x\sigma_j ^ x + \sigma_i ^ y\sigma_j ^ y + \sigma_i ^ z\sigma_j ^ z),
#
# where :math:`J` is the coupling constant and :math:`\sigma` is a Pauli operator. The Hamiltonian
# can be constructed on a ``triangle`` lattice as follows.

coupling = [0.5, 0.5, 0.5]
hamiltonian = qml.spin.heisenberg('triangle', n_cells, coupling)

lattice = qml.spin.generate_lattice('triangle', n_cells)
plot(lattice)

######################################################################
# Transverse-field Ising model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The `transverse-field Ising model (TFIM) Hamiltonian
# <https://docs.pennylane.ai/en/latest/code/api/pennylane.spin.transverse_ising.html>`__
# is defined as
#
# .. math::
#
#     H = -J \sum_{<i,j>} \sigma_i^{z} \sigma_j^{z} - h\sum_{i} \sigma_{i}^{x},
#
# where :math:`J` is the coupling constant, :math:`h` is the strength of the transverse magnetic
# field and :math:`\sigma` is a Pauli operator. The Hamiltonian can be constructed on the
# ``honeycomb`` lattice as follows.

coupling, h = 0.5, 1.0
hamiltonian = qml.spin.transverse_ising('honeycomb', n_cells, coupling, h)

lattice = qml.spin.generate_lattice('honeycomb', n_cells)
plot(lattice)

######################################################################
# Kitaev's honeycomb model
# ^^^^^^^^^^^^^^^^^^^^^^^^
# The `Kitaev honeycomb model Hamiltonian <https://docs.pennylane.ai/en/latest/code/api/pennylane.spin.kitaev.html>`__
# is defined on the honeycomb lattice, as
#
# .. math::
#
#     H = K_X \sum_{\langle i,j \rangle \in X}\sigma_i^x\sigma_j^x +
#     \:\: K_Y \sum_{\langle i,j \rangle \in Y}\sigma_i^y\sigma_j^y +
#     \:\: K_Z \sum_{\langle i,j \rangle \in Z}\sigma_i^z\sigma_j^z,
#
# where :math:`\sigma` is a Pauli operator and the parameters :math:`K_X`, :math:`K_Y`, :math:`K_Z`
# are the coupling constants in each direction. The Hamiltonian can be constructed as follows.

coupling = [0.5, 0.6, 0.7]
hamiltonian = qml.spin.kitaev(n_cells, coupling)

######################################################################
# Haldane model
# ^^^^^^^^^^^^^
# The `Haldane model Hamiltonian <https://docs.pennylane.ai/en/latest/code/api/pennylane.spin.haldane.html>`__
# is defined as
#
# .. math::
#
#     H = - t^{1} \sum_{\langle i,j \rangle, \sigma}
#         c_{i\sigma}^\dagger c_{j\sigma}
#         - t^{2} \sum_{\langle\langle i,j \rangle\rangle, \sigma}
#         \left( e^{i\phi} c_{i\sigma}^\dagger c_{j\sigma} + e^{-i\phi} c_{j\sigma}^\dagger c_{i\sigma} \right),
#
# where :math:`t^{1}` is the hopping amplitude between neighbouring sites
# :math:`\langle i,j \rangle`, :math:`t^{2}` is the hopping amplitude between next-nearest neighbour
# sites :math:`\langle \langle i,j \rangle \rangle`, :math:`\phi` is the phase factor that breaks
# time-reversal symmetry in the system, and :math:`\sigma` is the spin degree of freedom. This
# function assumes two fermions with opposite spins on each lattice site. The Hamiltonian can be
# constructed on the ``kagome`` lattice using the following code.

hopping = 0.5
hopping_next = 1.0
phi = 0.1
hamiltonian = qml.spin.haldane('kagome', n_cells, hopping, hopping_next, phi)

lattice = qml.spin.generate_lattice('kagome', n_cells)
plot(lattice)

######################################################################
# Emery model
# ^^^^^^^^^^^
# The `Emery model Hamiltonian <https://docs.pennylane.ai/en/latest/code/api/pennylane.spin.emery.html>`__
# is defined as
#
# .. math::
#
#     H = - t \sum_{\langle i,j \rangle, \sigma} c_{i\sigma}^{\dagger}c_{j\sigma}
#         + U \sum_{i} n_{i \uparrow} n_{i\downarrow} + V \sum_{<i,j>} (n_{i \uparrow} +
#         n_{i \downarrow})(n_{j \uparrow} + n_{j \downarrow})\ ,
#
# where :math:`t` is the hopping term representing the kinetic energy of electrons,
# :math:`U` is the on-site Coulomb interaction representing the repulsion between electrons,
# :math:`V` is the intersite coupling,
# :math:`\sigma` is the spin degree of freedom, and :math:`n_{k \uparrow}`, :math:`n_{k \downarrow}`
# are number operators for spin-up and spin-down fermions at site :math:`k`. This function assumes
# two fermions with opposite spins on each lattice site. The Hamiltonian can be
# constructed on the ``lieb`` lattice as follows.

hopping = 0.5
coulomb = 1.0
intersite_coupling = 0.2
hamiltonian = qml.spin.emery('lieb', n_cells, hopping, coulomb, intersite_coupling)

lattice = qml.spin.generate_lattice('lieb', n_cells)
plot(lattice)

######################################################################
# Building Hamiltonians manually
# ------------------------------
# The Hamiltonian template functions are great and simple tools for someone who just wants to build
# a Hamiltonian quickly. PennyLane also offers tools for building customized Hamiltonians. Let’s learn
# how to use these tools by constructing the Hamiltonian for the
# `transverse-field Ising model <https://docs.pennylane.ai/en/latest/code/api/pennylane.spin.transverse_ising.html>`__
# on a two-dimensional lattice.
#
# The Hamiltonian is represented as:
#
# .. math::
#
#     H =  -J \sum_{\left< i,j \right>} \sigma_i^{z} \sigma_j^{z} - h\sum_{i} \sigma_{i}^{x},
#
# where :math:`J` is the coupling defined for the Hamiltonian, :math:`h` is the strength of
# transverse magnetic field, and :math:`\left< i,j \right>` represents the indices of neighbouring
# spins.
#
# Our approach for doing this is to construct a lattice that represents the spin sites and their
# connectivity. This is done by using the :class:`~.pennylane.spin.Lattice` class, which can be
# constructed either by calling the helper function :func:`~.pennylane.spin.generate_lattice` or by
# manually constructing the object. Let's see examples of both methods. First we use
# :func:`~.pennylane.spin.generate_lattice` to construct a square lattice containing
# :math:`3 \times 3 = 9` cells. Because each cell of the ``square`` lattice contains only one
# site, we get :math:`9` sites in total, which are all connected to their nearest neighbor.

lattice = qml.spin.generate_lattice('square', [3, 3])

######################################################################
# To visualize this lattice, we use the plotting function we created before.

plot(lattice)

######################################################################
# Now, we construct a lattice manually by explicitly defining the positions of the sites in a
# unit cell, the primitive translation vectors defining the lattice and the number of cells in each
# direction [#ashcroft]_. Recall that a unit cell is the smallest repeating unit of a lattice.
#
# Let's create a square-octagon [#jovanovic]_ lattice manually. Our lattice
# can be constructed in a two-dimensional Cartesian coordinate system with two primitive
# translation vectors defined as unit vectors along the :math:`x` and :math:`y` directions, and four
# lattice point located inside the unit cell. We also assume that the lattice has three unit cells
# along each direction.

from pennylane.spin import Lattice

positions = [[0.2, 0.5], [0.5, 0.2],
             [0.5, 0.8], [0.8, 0.5]] # coordinates of sites
vectors = [[1, 0], [0, 1]] # primitive translation vectors
n_cells = [3, 3] # number of unit cells in each direction

lattice = Lattice(n_cells, vectors, positions, neighbour_order=2)

plot(lattice, figsize = (5, 5), showlabel=False)

######################################################################
# Constructing the lattice manually is more flexible, while :func:`~.pennylane.spin.generate_lattice`
# only works for some
# `predefined lattice shapes <https://docs.pennylane.ai/en/latest/code/api/pennylane.spin.generate_lattice.html#lattice-details>`__.
#
# Now that we have the lattice, we can use its attributes, e.g., edges and vertices, to construct
# our transverse-field Ising model Hamiltonian. For instance, we can access the number of sites
# with ``lattice.n_sites`` and the indices that define each edge with ``lattice.edges_indices``. For
# the full list of attributes, please see the documentation of the :class:`~.pennylane.spin.Lattice`
# class. We also need to define the coupling, :math:`J`, and onsite parameters of the
# Hamiltonian, :math:`h`.

from pennylane import X, Y, Z

coupling, onsite = 0.25, 0.75

hamiltonian = 0.0
# add the one-site terms
for vertex in range(lattice.n_sites):
    hamiltonian += -onsite * X(vertex)

# add the coupling terms
for edge in lattice.edges_indices:
    i, j = edge[0], edge[1]
    hamiltonian += - coupling * (Z(i) @ Z(j))

hamiltonian

######################################################################
# In this example, we just used the built-in attributes of our custom lattice without further
# customising them. The lattice can be constructed in a more flexible way that allows us to build
# fully general spin Hamiltonians. Let's look at an example.
#
# Building anisotropic Hamiltonians
# ---------------------------------
# Now we work on a more complicated Hamiltonian. We construct the anisotropic square-trigonal
# [#jovanovic]_ model, where the coupling parameters
# depend on the orientation of the bonds. We can construct the Hamiltonian by building the
# lattice manually and adding custom edges between the nodes. For instance, to define a custom
# ``XX`` edge with the coupling constant :math:`0.5` between nodes 0 and 1, we use the following.

custom_edge = [(0, 1), ('XX', 0.5)]

######################################################################
# Let's now build our Hamiltonian. We first define the unit cell by specifying the positions of the
# nodes and the translation vectors and then define the number of unit cells in each
# direction [#jovanovic]_.

positions = [[0.1830, 0.3169],
             [0.3169, 0.8169],
             [0.6830, 0.1830],
             [0.8169, 0.6830]]

vectors = [[1, 0], [0, 1]]

n_cells = [3, 3]

######################################################################
# Let's plot the lattice to see what it looks like.

plot(Lattice(n_cells, vectors, positions), figsize=(5, 5))

######################################################################
# Now we add custom edges to the lattice. In our example, we define four types of custom
# edges: the first type is the one that connects node 0 to 1, the second type is defined to connect
# node 0 to 2, and the third and fourth types connect node 1 to 3 and 2 to 3, respectively. Note that
# this is an arbitrary selection. You can define any type of custom edge you would like.

custom_edges = [[(0, 1), ('XX', 0.5)],
                [(0, 2), ('YY', 0.6)],
                [(1, 3), ('ZZ', 0.7)],
                [(2, 3), ('ZZ', 0.7)]]

lattice = Lattice(n_cells, vectors, positions, custom_edges=custom_edges)

######################################################################
# Let's print the lattice edges and check that our custom edge types are set correctly.

print(lattice.edges)

######################################################################
# You can compare these edges with the lattice plotted above and verify the correct translation of
# the edges over the entire lattice sites.
#
# Now we pass the lattice object to the :func:`~.pennylane.spin.spin_hamiltonian` function, which is
# a helper function that constructs a Hamiltonian from a lattice object.

hamiltonian = qml.spin.spin_hamiltonian(lattice=lattice)

######################################################################
# Alternatively, you can build the Hamiltonian manually by looping over the custom edges
# to build the Hamiltonian.

opmap = {'X': X, 'Y': Y, 'Z': Z}

hamiltonian = 0.0
for edge in lattice.edges:
    i, j = edge[0], edge[1]
    k, l = edge[2][0][0], edge[2][0][1]
    hamiltonian += opmap[k](i) @ opmap[l](j) * edge[2][1]

hamiltonian

######################################################################
# You can see that it is easy and intuitive to construct this anisotropic Hamiltonian with the tools
# available in the `qml.spin <https://docs.pennylane.ai/en/latest/code/qml_spin.html>`__ module. You can
# use these tools to construct custom Hamiltonians for other interesting systems.
#
# Conclusion
# ----------
# The `spin module <https://docs.pennylane.ai/en/latest/code/qml_spin.html>`__ in PennyLane provides
# a set of powerful tools for constructing spin Hamiltonians.
# Here we learned how to use these tools to construct predefined Hamiltonian templates such as the
# Fermi–Hubbard Hamiltonian. This can be done with our built-in functions that currently support
# several commonly used spin models and a variety of lattice shapes. More importantly, PennyLane
# provides easy-to-use function to manually build spin Hamiltonians on customized lattice structures
# with anisotropic interactions between the sites. This can be done intuitively using the
# :class:`~.pennylane.spin.Lattice` object and provided helper functions. The versatility of the new
# spin functionality allows you to construct any new spin Hamiltonian quickly and intuitively.
#
# References
# ----------
#
# .. [#ashcroft]
#
#     N. W. Ashcroft, D. N. Mermin,
#     "Solid State Physics", Chapter 4, New York: Saunders College Publishing, 1976.
#
# .. [#jovanovic]
#
#     D. Jovanovic, R. Gajic, K. Hingerl,
#     "Refraction and band isotropy in 2D square-like Archimedean photonic crystal lattices",
#     Opt. Express 16, 4048, 2008.
#
