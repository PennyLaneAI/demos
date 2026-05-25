r"""

Introducing Lattice Surgery
===========================

Lattice surgery is a way to perform fault-tolerant operations on two-dimensional quantum computer architectures
with local physical connectivity. Its introduction improved spatial overheads in topological quantum
error correction codes, such as surface codes, by replacing continuous code operations 
with discontinuous ones: lattice splitting and merging.
In this demo, we are going to see how these operations
enable parity measurements of arbitrary Pauli operators, which unlock universal fault-tolerant quantum computing.

.. figure:: _static/hero_illustrations/pennylane-demo-lattice-surgery-hero.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

For self-consistency, we are going to briefly recap the basic concepts of the surface code and explain why `Pauli product measurements <https://pennylane.ai/compilation/pauli-based-computation>`__ are sufficient 
for universal computing. We are then going to explain how to perform such 
`homological measurements <https://arxiv.org/abs/2410.02753>`__ using lattice splitting and merging.

Surface code quantum computing
------------------------------

Topological quantum error correction codes, like the surface code [#surfacecode]_, protect quantum information stored in logical qubits by
ensuring the underlying physical qubits are kept in the code space, defined by the :math:`+1` eigenspace of a set of :doc:`stabilizer measurements <demos/tutorial_stabilizer_codes>`.
A logical qubit is then represented by a patch of physical qubits, called data qubits, on a square lattice like so:

.. figure:: ../_static/demonstration_assets/lattice_surgery/surface_code_qubit1.png
    :align: center
    :width: 25%
    :target: javascript:void(0)
  
Each intersection of a line corresponds to a data qubit. Here we have a code distance of :math:`d=5`, so :math:`5 \times 5` data qubits in total. 
On yellow surfaces, we continually measure X stabilizers, which simply are the :math:`X_a X_b X_c X_d` operators
on the four data qubits of the corners. Similarly, Z stabilizers are measured on white surfaces. 
These stabilizer measurements are not performed directly on the data qubits, but via another kind of qubit that sits at the center of each square surface
as well as the center of each arch (shown below). 
These extra qubits are sometimes called measurement qubits or syndrome qubits and the measurement is performed by entangling four data qubits with one syndrome qubit and measuring it (see Fig. 1 in [#surfacecode]_).

Note that this representation of a qubit is in the so-called rotated surface code due to its :math:`45^\circ` rotation with respect to the original planar surface code.
Here we indicate the underlying data and syndrome qubits with black and red dots, respectively. 
The vertical lines do not correspond to physical connections, but are a mere guide to the eye that differentiate X vertex and Z face syndromes. In principle, all
nearest neighbor qubits are physically connected and can perform universal gates in a noisy fashion.

.. figure:: ../_static/demonstration_assets/lattice_surgery/surface_code_qubit2.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

One such :math:`d \times d` patch represents a logical qubit. An important feature is that we have sides with X arches and Z arches,
as this enables the error-corrected encoding of the qubit. Note that the edge with X arches is called a Z edge, because measuring along that edge
corresponds to a logical Z operator (more on logical X and Z operators that define the qubit further below).

We are interested in performing `logical` operations like a CNOT gate between two such patches that represent a logical qubit, each.
The most straight-forward way to compute a logical CNOT is by performing physical CNOT gates between the data qubits
of each logical qubit patch, indicated here by the red line exemplarily for one pair:

.. figure:: ../_static/demonstration_assets/lattice_surgery/transversal_cnot.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

This is called a transversal operation because the logical operation just corresponds to performing the physical operation on each qubit.
For a two-qubit gate, this is problematic on hardware with only local (physical) connectivity (like those built of `superconducting qubits <https://pennylane.ai/qml/demos/tutorial_sc_qubits>`__).
In that case, we want to perform CNOT gates non-transversally to accomodate the connectivity constraints of such hardware platforms.
In the early days, this was achieved via braiding [#braiding]_, 
a concept commonly encountered in algebraic topology.
In this setting, qubits are encoded by defects in the code, and operations via continuous deformations of the code.
However, defect based qubits suffer from requiring significantly more physical qubits per logical qubit.

This is where lattice surgery [#latticesurgery]_ comes into play: It has been shown to enable error-corrected logical operations
with significantly lower space overhead and comparable time requirements [#Fowler]_ [#Litinski]_ [#Chamberland]_. The fundamental operations in lattice
surgery are discontinuous deformations of the lattice, in particular lattice merging and lattice splitting, common in geometric topology.

Universal quantum computing with Pauli product measurements
-----------------------------------------------------------

To achieve universal quantum computing, we need to be able to perform all Clifford gates, particularly CNOT gates.
Furthermore, we need to be able to reliably inject states to enable `magic state injection <https://pennylane.ai/qml/glossary/what-are-magic-states>`__.
This line of reasoning is a bottom-up way to show that lattice surgery enables universal quantum computing, which was done in its original introduction [#latticesurgery]_.

Let us alternatively take a top-down approach here and show that we can perform arbitrary `Pauli product measurements <https://pennylane.ai/compilation/pauli-based-computation>`__ (PPMs), 
because we know this enables universal quantum computing, as illustrated in, e.g., the :doc:`Game of Surface Codes <demos/tutorial_game_of_surface_codes>` [#Litinski]_.
The gist of it is that `Pauli product rotations <https://pennylane.ai/compilation/pauli-product-rotations>`__ (PPRs) like :math:`e^{-i \tfrac{\theta}{2} P}`, represented as

.. figure:: ../_static/demonstration_assets/lattice_surgery/PPR.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

for arbitrary Pauli words :math:`P`
with Clifford angles (multiples of :math:`\pi/2`) and non-Clifford angles (odd multiples of :math:`\pi/4`) reproduce the `(Clifford + T) <https://pennylane.ai/compilation/clifford-t-gate-set>`__ gate set, and thus allow for universal quantum computing.
These PPRs can be directly executed using PPMs in the following way, so all we will need to show below is that the lattice surgery based quantum computer can perform arbitrary PPMs. Typically, this involves an auxiliary qubit in a specific state.
Clifford angles that are odd multiples of :math:`\frac{\pi}{2}` can be performed with an auxiliary qubit in :math:`|0\rangle` like so:

.. figure:: ../_static/demonstration_assets/lattice_surgery/clifford_PPM.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

We use the color coding from [#Litinski]_, where orange boxes correspond to Clifford gates, 
gray boxes to Pauli operators (note that :math:`\exp\left(-i \tfrac{(2k+1)\pi}{2}P\right) \propto P`), green boxes to non-Clifford gates, and
blue boxes to measurements.
Non-Clifford PPRs can be realized using a magic resource state in the following way:

.. figure:: ../_static/demonstration_assets/lattice_surgery/non_clifford_PPM.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

Here, we injected a magic state :math:`T|+\rangle = \tfrac{1}{\sqrt{2}} \left(|0\rangle + e^{-i \tfrac{1}{4}} |1\rangle\right)` that was produced separately via `magic state distillation <https://pennylane.ai/qml/demos/tutorial_magic_state_distillation>`__.
Note that Pauli operations that have angles that are multiples of 
:math:`\pi` do not need to be executed on hardware, but can be tracked in software.
In addition, we stress that there are different circuit identities to realize PPRs via PPMs. Here, we showed just the basic examples from [#Litinski]_.
For our purposes here it suffices to note that realizing PPRs via PPMs is possible, so the only thing left to show is how to perform arbitrary PPMs using lattice surgery.

Single-qubit measurements
-------------------------

Before we get to the meat of performing arbitrary PPMs with lattice surgery, let us recall how to perform single-qubit measurements (and thus single-qubit gates).
In the orientation we previously used, logical operators :math:`Z_L` and :math:`X_L` of the logical qubit are defined by a horizontal line of :math:`Z` and vertical line of :math:`X` measurements
on the data qubits:

.. figure:: ../_static/demonstration_assets/lattice_surgery/logical_X_Z.png
    :align: center
    :width: 30%
    :target: javascript:void(0)

Note that these are not unique, but rather examples of the topological equivalence class of connected lines between the two Z and X edges, respectively.
Any such operator represents the logical operators :math:`Z_L` and :math:`X_L`. In particular, the edges also correspond to the logical operators, which
leads to the somewhat confusing scenario where the edge with Z arches is called an X edge, and vice versa.

Because we only care about the homology of the measurement
(i.e., that it in the same equivalence class of the corresponding X and Z edges), this is also called a *homological measurement*.
It does not really matter which operator of the equivalence class we measure. We have access to any of them without extra effort because they are related to each other via stabilizer measurements, 
which we perform anyway during the continually performed error correction cycles.
So for the following, we will just consider those that are convenient for illustrational purposes.

An important feature of the logical operators is that they are not stabilizers, but commute with all stabilizers to ensure that the logical operator does not move the 
state out of the code space (:math:`O_L S |\psi\rangle_L = S O_L |\psi\rangle_L`).
The commutation can be seen from the fact
that the logical operator only ever overlaps with an even multiple of X or Z operators, and thus commutes with any other stabilizer.

We also note that the logical :math:`Y_L` measurement is a topic on its own. In principle, one could measure :math:`Y_L` by simultaneously measuring the logical
:math:`X_L` and :math:`Z_L` with a physical :math:`Y` measurement on the intersecting data qubit.
However, this is not a properly defined logical operator anymore as it does not commute with the stabilizers, and thus moves the qubit out of the code space.
Measuring (and applying, it's the same really) logical :math:`Y_L` operators is still possible, just a little more complicated as we will show further below.

Arbitrary Pauli product measurements via lattice merging and splitting
----------------------------------------------------------------------

We now want to show how to perform the two fundamental operations of measuring 
:math:`Z_L \otimes Z_L` and :math:`X_L \otimes X_L` between two surface code qubits via lattice surgery.
The operation is fairly simple in principle: merge the two patches, and split them again. 
We just need to make sure that the qubits are connected along the corresponding edges.

In particular, for measuring :math:`X_L \otimes X_L`, we merge and split two qubits on their X edges.
For simplicity, let us assume that the two qubits are already facing each other on their X edges with a column of
unused data qubits separating them. In principle we can have a larger area separating them, we just need to be able to make the boundaries match.
The lattice merging is started by preparing the intermediate data qubits in :math:`|0\rangle`.

.. figure:: ../_static/demonstration_assets/lattice_surgery/XX1.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

The actual merging is then done simply by including the intermediate X and Z stabilizers in the error correction cycles.

.. figure:: ../_static/demonstration_assets/lattice_surgery/XX2.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

Conveniently, the product of all stabilizers between the two logical :math:`X_L` operators, indicated by the red dots below, corresponds to the eigenvalue of :math:`X_L \otimes X_L`.

.. figure:: ../_static/demonstration_assets/lattice_surgery/XX3.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

Now that we retrieved our measurement result :math:`X_L \otimes X_L`, we want to restore the two qubits, which is achieved by lattice splitting. This, on the other hand,
is done by measuring the intermediate data qubits in the Z basis.

.. figure:: ../_static/demonstration_assets/lattice_surgery/XX4.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

In case the green Z measurement in the middle yields a negative sign, 
we need to assign it to one of the two logical :math:`Z_L` operators, 
on top of the product of the sign of the original two :math:`Z_L` operators we originally started from.

Measuring :math:`Z_L \otimes Z_L` works in the same fashion, but with reversed roles: We connect the Z edges with each other, initialize in :math:`|+\rangle` to merge, and measure in :math:`X` to split again:

.. figure:: ../_static/demonstration_assets/lattice_surgery/ZZ.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

The rules for arbitrary products like, e.g., :math:`X_L \otimes Z_L` are relatively simple: Merge and split the qubit patches on the corresponding edges and keep track of any signs you pick up on the way.
The same goes for measuring more than two operators (see, e.g., Figure 41 in [#Litinski]_ for an example of measuring :math:`Y_L \otimes X_L \otimes Z_L \otimes X_L`).
Note that we can also rotate patches to get the correct orientation, as we will see below.

Y measurements
--------------

At this point we can measure arbitrary :math:`X_L` and :math:`Z_L` Pauli product measurements. The last missing ingredient for universality is measuring :math:`Y_L` operators.

Introducing discontinuous operations on the surface code does not exclude the usage of continuous transformations of the code.
In particular, we can always extend a qubit to a larger surface or move the edges of it. These are important operations if we want to include logical :math:`Y_L` measurements.

The operations are very similar to the merging and splitting operations. To extend a qubit patch along its Z edge, we initialize the neighboring data qubits in :math:`|0\rangle` 

.. figure:: ../_static/demonstration_assets/lattice_surgery/extend1.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

and perform :math:`d` cycles of error correction in this configuration.

.. figure:: ../_static/demonstration_assets/lattice_surgery/extend2.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

To remove part of the extended patch again, we perform :math:`Z` measurements on the data qubits, effectively moving the qubit one patch size to the right:

.. figure:: ../_static/demonstration_assets/lattice_surgery/extend3.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

Recall from earlier that green and red lines merely indicate representatives of the logical :math:`Z_L` and :math:`X_L` operators, so moving the red line to the right is free within the error correction cycles.

Extending the qubit along the X edge works similarly, but moving vertically, initializing in :math:`|+\rangle` and measuring :math:`X` on the data qubits.

We can also move (or rotate?) the type of the edges within :math:`d` error correction code cycles.
This makes most sense on an already extended qubit because this way we do not change the code distance.
As we can see, the bulk of the qubit is untouched and all that is changed is the set of stabilizers we measure on the boundaries (for :math:`d` code cycles to take effect).

.. figure:: ../_static/demonstration_assets/lattice_surgery/corner_moving.png
    :align: center
    :width: 70%
    :target: javascript:void(0)
  
Here we show an extended single-qubit patch and three example re-orientations of the type of edges.
The smaller images are guides to the eye to indicate the settings, with X edges as solid lines and Z edges as dashed lines (i.e. the same as the logical measurements moved to the edges).
Note that the important property is for neighboring X and Z edges to overlap on two data qubits for the corresponding stabilizers to commute. 
This is why the right patches introduce the triangle-shaped stabilizers of three :math:`Z` operators.

Finally, to measure $Y$ we perform the following procedure:

.. figure:: ../_static/demonstration_assets/lattice_surgery/y_measurement.png
    :align: center
    :width: 90%
    :target: javascript:void(0)

Here, one extends and orients a qubit such that it has both :math:`X` and :math:`Z` edges on the same side. 
At the same time, an auxiliary qubit is initialized in :math:`|0\rangle` parallel to our extended qubit.
This is intentionally done such that there is a mismatch between the boundary stabilizers which introduces a so-called twist defect, highlighted in purple.
When we perform lattice merging between the two qubits, we obtain new stabilizers, some of which are mixed Z and X!
This is because they still need to commute with all other stabilizers. That this is indeed the case can be shown by checking that 
:math:`[ZZ, XY] = [XX, YZ,] = [XZ, ZX] = 0` on top of the previously discussed :math:`[XX, ZZ] = 0`:

.. figure:: ../_static/demonstration_assets/lattice_surgery/twist_boundaries.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

Some of the new stabilizers are trivial in the sense that they are simply the product of the already-measured stabilizers from each of the qubits. 
The non-trivial new ones are highlighted in red and correspond to the measurement of :math:`Y \otimes Z` of the joint state :math:`|\psi\rangle \otimes |0\rangle`
between the qubit and the auxiliary qubit in :math:`|0\rangle`, effectively measuring :math:`\langle Y \rangle = \langle \psi | Y |\psi \rangle = \langle \psi 0 | Y \otimes Z |\psi 0 \rangle`.
A more intuitive way to view this is that we simultaneously measured the X and Z edge of our extended qubit, yielding the :math:`Y \propto X Z` measurement (modulo some global phase). This is exactly the perspective in the game of surface codes [#Litinski]_.

Measuring :math:`Y_L` inside a Pauli product measurement, e.g. :math:`Y_L \otimes Z_L` works in the same fashion
as above with lattice merging and splitting while making sure a twist defect is introduced. This corresponds to having
the first qubit facing the other with both X and Z edges.

More details on twist-based lattice surgery can be found in [#Litinski2]_.

"""


##############################################################################
# 
#
# Conclusion
# ----------
#
# Lattice surgery is enabling error-corrected quantum computing on planar codes
# with local connectivity as it allows for non-transversal gates. We introduced its basics
# exemplified on the surface code and by realizing arbitrary Pauli product measurements.
# These are performed by merging and splitting qubits in such a way that the logical operators "topologically" align.
# In this process of merging and splitting, we only ever change which stabilizers we measure, but we never perform a strict projective measurement of the logical operator.
# In a sense, we are reading out parity information from the stabilizer measurements in a non-perturbative way.
# The parity on the other hand provides us only with the topological information of what the logical string is connecting,
# which is why this process is called homological measurement.
#
# Note that the introduction of the higher weight twist-defect stabilizer with 5 Pauli operators breaks the nearest-neighbor connectivity the surface code requires.
# This leads to deeper physical CNOT circuits for syndrome extraction, and thus leads to a higher error probability.
# This is why twist-free alternatives are being proposed at higher resource costs (time or space) [#Chamberland]_.
# 
#
# References
# ----------
#
# .. [#surfacecode]
#
#     Austin G. Fowler, Matteo Mariantoni, John M. Martinis, Andrew N. Cleland,
#     "Surface codes: Towards practical large-scale quantum computation",
#     `arXiv:1208.0928 <https://arxiv.org/abs/1208.0928>`__, 2012
#
#
# .. [#braiding]
#
#     Robert Raussendorf, Jim Harrington, Kovid Goyal,
#     "Topological fault-tolerance in cluster state quantum computation",
#     `arXiv:quant-ph/0703143 <https://arxiv.org/abs/quant-ph/0703143>`__, 2007
#
#
# .. [#latticesurgery]
#
#     Dominic Horsman, Austin G. Fowler, Simon Devitt, Rodney Van Meter,
#     "Surface code quantum computing by lattice surgery",
#     `arXiv:1111.4022 <https://arxiv.org/abs/1111.4022>`__, 2011
#
#
# .. [#Fowler]
#
#     Austin G. Fowler, Craig Gidney
#     "Low overhead quantum computation using lattice surgery"
#     `arXiv:1808.06709 <https://arxiv.org/abs/1808.06709>`__, 2018.
#
#
# .. [#Litinski]
#
#     Daniel Litinski
#     "A Game of Surface Codes: Large-Scale Quantum Computing with Lattice Surgery"
#     `arXiv:1808.02892 <https://arxiv.org/abs/1808.02892v3>`__, 2018.
#
#
# .. [#Chamberland]
#
#     Christopher Chamberland, Earl T. Campbell
#     "Universal quantum computing with twist-free and temporally encoded lattice surgery",
#     `arXiv:2109.02746 <https://arxiv.org/abs/2109.02746>`__, 2021
#
#
# .. [#Litinski2]
#
#     Daniel Litinski, Felix von Oppen
#     "Lattice Surgery with a Twist: Simplifying Clifford Gates of Surface Codes",
#     `arXiv:1709.02318 <https://arxiv.org/abs/1709.02318>`__, 2017
#
#
