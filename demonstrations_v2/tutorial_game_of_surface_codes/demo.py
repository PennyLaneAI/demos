r"""A Game of Surface Codes: Large-Scale Quantum Computing with Lattice Surgery
===============================================================================

In surface-code based fault tolerant quantum computing architectures, 
T gates are typically implemented via injected `magic states <https://pennylane.ai/qml/glossary/what-are-magic-states>`__.
The layout and design of the architecture plays a crucial role in how fast a magic state can be reliably produced and consumed for computation.
The game of surface codes [#Litinski]_ allows us to reason about such space-time tradeoffs in architecture designs, without having to get into
the nitty-gritty details of surface code physics. In this demo, we will see how different designs can lead to faster computations at the cost of involving more qubits and vice versa.

.. figure:: ../_static/hero_illustrations/Hero_Game_of_Surface_Codes.png
    :align: center
    :width: 70%
    :target: javascript:void(0)


Introduction
------------

The game of surface codes is a high-level framework for designing surface code quantum computing architectures.
The game helps us understand space-time trade-offs, where designs with a higher qubit overhead allow for faster computations and vice versa.
For example, a space-efficient design might allow a computation with :math:`10^8` T gates to run in :math:`4` hours using :math:`55k` physical qubits, 
whereas an intermediate design may run the same computation in :math:`22` minutes using :math:`120k` physical qubits, 
or a time-optimized design in :math:`1` second using :math:`1500` interconnected quantum computers with :math:`220k` physical qubits, each.

One can draw a rough comparison to microchip design in classical computing, 
where the equivalent game would be about how to arrange the transistors of a chip to perform fast and efficient computations.

The game can be understood entirely from the rules described in the next section. 
However, it still helps to understand the correspondences in physical fault tolerant quantum computing (FTQC) architectures.
First of all it is important to note that we consider surface codes that implement `(Clifford + T) <https://pennylane.ai/compilation/clifford-t-gate-set>`__ circuits.
In particular, these circuits can be compiled to circuits that just perform `Pauli product measurements <https://pennylane.ai/compilation/pauli-based-computation>`__.
This is because all Clifford operations can be moved to the end of the circuit and merged with measurements. 
The remaining non-Clifford gates are realized by `magic state injection <https://pennylane.ai/qml/glossary/what-are-magic-states>`__ and more Clifford operations, which can be merged with measurements again.
Hence, we mainly care about performing measurements on qubits in arbitrary bases and efficiently distilling and injecting magic states.

We also note that the patches that represent qubits correspond to surface code qubits.
There is a detailed explanation in Appendix A of the paper that describes the surface code realizations of all operations that we are going to see.
These are useful to know in order to grasp the full depth of the game, but are not essential to understanding its rules and concluding design principles that we cover in this demo.
For further reading on these subjects, we recommend the `blog posts on the surface code and quantum error correction <https://arthurpesah.me/blog/>`__ by Arthur Pesah, our :doc:`demo on the toric code <demos/tutorial_toric_code>`, as well as the three-part series on the `toric code <https://decodoku.blogspot.com/2016/03/6-toric-code.html>`__ by James Wooton.

Rules of the game
-----------------

The game is played on a board of tiles, where patches correspond to logical qubits.
Underlying these tiles are physical qubits that are statically arranged (:math:`2d^2` physical qubits per tile for code distance :math:`d`).
But we should view logical qubit patches as dynamic entities that can appear, move around, deform and disappear again.
The goal of this demo will be to understand the design principles and space-time trade-offs for surface code architectures.

Data qubits are realized by patches that occupy at least one tile, but potentially multiple.
They always have four distinct boundaries corresponding to X (dotted) and Z (solid) edges.
This is shown in the figure below.

.. figure:: ../_static/demonstration_assets/game_of_surface_codes/qubit_definition_cropped.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

    Qubits are defined as patches of tiles on the board. 
    A single qubit can occupy one tile (a) or multiple tiles (b), where dotted lines correspond to X and solid lines to Z operators.
    Image source: Daniel Litinski [1].

Every operation in the game has an associated time cost that we measure in units of code cycles 🕒.
There are some discrepancies to actual surface code cycles, but the correspondance is close enough to weigh out space-time trade-offs in architecture designs.
We are not going to give an exhaustive overview of all possible operations, but focus on a few important ones and fill the remaining gaps necessary for the architecture designs in the respective sections below.


Arbitrary Pauli product measurements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At the cost of 0🕒 we can measure patches in the X and Z basis. If two patches share a border, one can measure the product of their shared edges as highlighted by the blue region in the figure below at the cost of 1🕒.

.. figure:: ../_static/demonstration_assets/game_of_surface_codes/ZZ_measurement.png
    :align: center
    :width: 20%
    :target: javascript:void(0)

    Simultaneously measuring the patches of two adjacent patches corresponds to the product of their neighboring edges. Here, we measure :math:`ZZ`.
    Image source: Daniel Litinski [1].

In particular, if the shared edge contains both Z and X edges, we can measure in the Y basis. In the following example, the upper qubit A has both operator edges :math:`Z_A` and :math:`X_A` exposed.
Measuring it together with the auxillary qubit B, initialized in the :math:`|0\rangle` state below, we measure :math:`(Z_A X_A) \otimes Z_B \propto Y_A \otimes Z_B` alltogether.

.. figure:: ../_static/demonstration_assets/game_of_surface_codes/Y_measurement.png
    :align: center
    :width: 20%
    :target: javascript:void(0)

    `Y` operators can be measured by having both X and Z edges be exposed with an adjacent auxiliary qubit. The measurement corresponds to the product of all involved operators, involving :math:`Z_A X_A \propto Y_A`.
    Image source: Daniel Litinski [1].

If we want to measure a single qubit patch in practice, we start off deforming it at the cost of 1🕒, initialize an auxiliary qubit at no cost, and perform the joint measurement as shown above (1🕒).
The entire protocol costs 2🕒 and is shown below:

.. figure:: ../_static/demonstration_assets/game_of_surface_codes/Y_measurement_protocol.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

    The protocol for measuring a single qubit in the Y basis involves deforming the patch (Step 2, 1🕒), initializing an auxillary qubit in :math:`|0\rangle` (0🕒), simultaneously measuring both patches (1🕒) and deforming the qubit back again (0🕒).
    Image source: Daniel Litinski [1].

Auxiliary qubits play an important role as they allow measuring products of Pauli operators on different qubits, 
which is the most crucial operation in this framework, since everything is mapped to `Pauli product measurements <https://pennylane.ai/compilation/pauli-based-computation>`__.

.. figure:: ../_static/demonstration_assets/game_of_surface_codes/PPM.png
    :align: center
    :width: 30%
    :target: javascript:void(0)

    Measuring :math:`Y_1 X_3 Z_4 X_5` via a joint auxiliary qubit in 1🕒. In principle multi-qubit measurements with many qubits come at the same cost as with fewer qubit.
    However, the requirement of having an auxiliary region connecting all qubits may demand extra deformations.
    Image source: Daniel Litinski [1].

Non-Clifford Pauli rotations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Non-Clifford Pauli rotations :math:`e^{-i \frac{\pi}{8} P}` for some Pauli word :math:`P` are realized via `magic state distillation and injection <https://pennylane.ai/qml/glossary/what-are-magic-states>`__.
Magic state distillation blocks are a crucial part of the architecture design that we are going to cover later. 
For the moment we assume that we have means to prepare magic states :math:`|m\rangle = |0\rangle + e^{-i \frac{\pi}{4}} |1\rangle` on special qubit tiles (distillation blocks).
Magic state injection in this case then refers to the following protocol:

.. figure:: ../_static/demonstration_assets/game_of_surface_codes/magic_state_injection.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

    Performing a non-Clifford :math:`\pi/8` rotation corresponds to performing the joint measurement of the Pauli word and :math:`Z` on the magic state qubit.
    The measurement of :math:`P \otimes Z_m` costs 1🕒, the subsequent :math:`X` measurement is free.
    The additional classically controlled Clifford rotations can be merged again with the measurements at the end of the circuit.
    Image source: Daniel Litinski [1].

Take for example the Pauli word :math:`P = Z_1 Y_2 X_4` on the architecture layout below. 
This design allows one to directly perform :math:`e^{-i \frac{\pi}{8} P}` as we have access to all of :math:`X, Y, Z` on each qubit, as well as the :math:`Z` edge for the magic state qubit.

.. figure:: ../_static/demonstration_assets/game_of_surface_codes/non_clifford_rotation.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

    Performing :math:`e^{-i \frac{\pi}{8} Z_1 Y_2 X_4}` by measuring :math:`Z_1 Y_2 X_4 Z_m`. The additional measurement :math:`X` on the magic state qubit is not shown and has no additional cost. The remaining Clifford Pauli rotations are merged with the terminal measurements at the end of the circuit via compilation.
    Image source: Daniel Litinski [1].

We are going to see in the next section that one of the biggest problems is performing Y rotations and measurements (same thing, really, in this framework).

Data blocks design
------------------

Computation happens on logical data qubits that are arranged on a so-called data block.
We now have all the necessary tools to understand different designs and their space-time tradeoffs.
In particular, the speed of the quantum computer is determined by how fast a magic state can be distilled and consumed by a data block.
In this section we focus on how the design affects how fast a magic state can be consumed by a block and do not focus on the distillation itself (this will be handled in the next section).


Compact data blocks
^^^^^^^^^^^^^^^^^^^

The compact data block has the following form. The middle aisle is going to be used as an auxiliary qubit region.

.. figure:: ../_static/demonstration_assets/game_of_surface_codes/compact_block.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

    The compact data block design is efficient in space. However, only one edge is exposed to the auxiliary qubit region in the middle.
    Image source: Daniel Litinski [1].

This design only uses :math:`\frac{3}{2}n + 3` tiles for :math:`n` qubits.
The biggest drawback is rather obvious: we can only access :math:`Z` measurements in the auxiliary qubit region. In order to perform joint :math:`X` measurements,
we can perform a patch rotation at a cost of 3🕒:

.. figure:: ../_static/demonstration_assets/game_of_surface_codes/patch_rotation.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

    A patch rotation can be used to expose the :math:`X` edge to the auxiliary qubit region.
    Image source: Daniel Litinski [1].

The worst thing that can happen is to have two opposite qubits require an X measurement, 
e.g. qubits (3 and 4) or (5 and 6). If either or both occurs, it takes a total of 6🕒 to rotate the patches.

An additional problem of this design is the fact that there are no tiles for qubits to expand to in order to perform Y measurements.
This can be remedied by making use of the identity 

.. math:: e^{i \frac{\pi}{8} Y} = e^{-i \frac{\pi}{4} Z} e^{i \frac{\pi}{8} X} e^{i \frac{\pi}{4} Z}.

The Clifford rotation on the right :math:`e^{i \frac{\pi}{4} Z}`, which is applied first, needs to be explicitly performed in this case. The second Clifford rotation (:math:`e^{-i \frac{\pi}{4} Z}`) can be merged with the terminal measurements of the circuit.
Such a rotation :math:`e^{i \frac{\pi}{4} P}` can be performed with a joint measurement of :math:`P \otimes Y`, similar to the magic state distillation circuit:

.. figure:: ../_static/demonstration_assets/game_of_surface_codes/clifford_rotation.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

    A Clifford rotation :math:`e^{i \frac{\pi}{4} P}` is performed by measuring :math:`P \otimes Y`.
    Image source: Daniel Litinski [1].

In particular, we still need to be able to perform a :math:`Y` measurement `somewhere`.
In this case we just outsourced it to another resource qubit, which we can use for all others and for which we left space in the bottom left corner of the compact data block.
For example, we can perform the rotation :math:`e^{i \frac{\pi}{4} Z_3 Z_5 Z_6}` at a cost of 1🕒 in the following way:

.. figure:: ../_static/demonstration_assets/game_of_surface_codes/clifford_rotation_356.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

    A Clifford rotation :math:`e^{i \frac{\pi}{4} Z_3 Z_5 Z_6}` is performed by measuring :math:`Z_3 Z_5 Z_6 \otimes Y_\text{resource}` with the additional resource qubit in the bottom left corner of the compact block.
    Image source: Daniel Litinski [1].

The worst case here is having an even number of :math:`Y` operators in the Pauli word, as it requires two distinct :math:`\frac{\pi}{4}` rotations, each costing 2🕒.

Overall, in the worst case scenario an operation can cost 9🕒. This consists of the base cost of 1🕒 for performing the Pauli measaurement, 2🕒 for having an even number of :math:`Y` operators, and 6🕒 when opposite qubit patches require :math:`X` measurements.
The following protocol shows such a scenario by performing :math:`e^{i \frac{\pi}{8} Y_1 Y_3 Z_4 Y_5 Y_6}`, which is realized by :math:`e^{i \frac{\pi}{8} X_1 X_3 Z_4 X_5 X_6} e^{i \frac{\pi}{4} Z_3 Z_5 Z_6} e^{i \frac{\pi}{4} Z_1}` (ignoring again the additional two :math:`\frac{\pi}{4}` rotations that are merged with measurements).

.. figure:: ../_static/demonstration_assets/game_of_surface_codes/compact_block_worst_case.png
    :align: center
    :width: 90%
    :target: javascript:void(0)

    Worst case scenario in the compact block when performing :math:`e^{i \frac{\pi}{8} Y_1 Y_3 Z_4 Y_5 Y_6}`.
    Step 2 measures :math:`Z_1` together with :math:`Y` on the resource qubit in order to perform the :math:`e^{i \frac{\pi}{4} Z_1}` rotation at 1🕒.
    Step 3 performs the additional :math:`X` measurement on the resource qubit at 0🕒.
    Same for steps 4 and 5 for performing :math:`e^{i \frac{\pi}{4} Z_3 Z_5 Z_6}` at 1🕒 overall.
    Steps 6 and 7 perform the patch rotations at 3🕒, each. And the final measurement of :math:`X_1 X_3 Z_4 X_5 X_6 Z_m` at another 1🕒 in step 8 completes the computation.
    Image source: Daniel Litinski [1].


Intermediate data blocks
^^^^^^^^^^^^^^^^^^^^^^^^

The intermediate data block design gets rid of the problem of potentially having blocking :math:`X`
measurements on opposite qubit patches by simply removing the second row and laying out all qubits in a linear fashion.

.. figure:: ../_static/demonstration_assets/game_of_surface_codes/intermediate_block.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

    Intermediate data block design.
    Image source: Daniel Litinski [1].

As such, this architecture occupies :math:`2n + 4` tiles. One can get additional savings by having the auxiliary qubit region be flexibly the lower or upper row.
This way, one can save on the extra cost of rotating patches back to their original position.

.. figure:: ../_static/demonstration_assets/game_of_surface_codes/intermediate_worst_case.png
    :align: center
    :width: 90%
    :target: javascript:void(0)

    Performing a :math:`ZXZZX` measurement by performing patch rotations for the appropriate :math:`X` measurements and moving all qubits down into the auxiliary region to save time.
    Image source: Daniel Litinski [1].

Overall we get a maximum of 2🕒 for the rotations. Adding the base cost of 1🕒 for the measurement
and the maximum 2🕒 for the additional Clifford :math:`\pi/4` Z rotations as in the compact block design,
we obtain a maximum cost of 5🕒.


Fast data blocks
^^^^^^^^^^^^^^^^

In order to be able to access Y operations directly, we need both Z and X edges exposed to the auxiliary qubit region, demanding 2 tiles for 1 qubit.
We omitted this in the rule description before as it is only relevant for the fast data block, but we can also realize 2 qubits on a single patch using 2 tiles:

.. figure:: ../_static/demonstration_assets/game_of_surface_codes/2q_patch.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

    Two qubits can be realized by a patch on two tiles. The patch now has 6 distinct edges, corresponding to the operators as indicated in the figure.
    Image source: Daniel Litinski [1].

With this extra trick up our sleeve, we can construct the fast data block consisting of two-qubit patches with an all-encompassing auxiliary qubit region.

.. figure:: ../_static/demonstration_assets/game_of_surface_codes/fast_block.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

    Fast data block design.
    Image source: Daniel Litinski [1].

Here, all 15 distinct Pauli operators are readily available. This is because we have 
:math:`X_1`, :math:`X_1 \otimes X_2`, :math:`Z_2`, :math:`Z_1 \otimes Z_2` and all products thereof available.
For example, we can realize :math:`X_2` via :math:`X_1 (X_1 \otimes X_2)` and we have 
:math:`Y_1 \propto (X_1) (Z_1) = (X_1) (Z_1 \otimes Z_2) (Z_2)`. With the same logic we can obtain :math:`Y_2` and :math:`Z_1`.
Further, we have operators like :math:`X_1 Y_1 \propto (X_1 \otimes X_2) Z_2`, :math:`Z_1 \otimes X_2 = X_1 (X_1 \otimes X_2) Z_2 (Z_1 \otimes Z_2)` and :math:`Y_1 X_2 \propto (X_1 \otimes X_2) (Z_2) (Z_1 \otimes Z_2)`.

The maximum time cost for performing a non-Clifford Pauli rotation therefore is just 1🕒 on the fast data block.

Distillation blocks design
--------------------------

So far we have only been concerned with data blocks that perform Pauli product measurements and assumed magic states to be available
for consumption.
These magic states need to be distilled in separate blocks, which can in principle be of the same design as data blocks. But since
the blocks are used for a fixed protocol, this knowledge can be used for simplifications.

There are different approaches to perform magic state distillation. We consider the case where we can prepare a magic state with infidelity :math:`p`.
The distillation protocol is then such that this infidelity is decreased to an acceptable level. All other operations of the protocol are Clifford, so we can measure if an error has occured.
This then determines the success probability of the protocol, which in the case below is roughly :math:`(1-p)^n` for an :math:`n`-qubit protocol.
We are going to go through the simplest protocol in a 15-to-1 distillation block.

15-to-1 distillation
^^^^^^^^^^^^^^^^^^^^

This protocol uses 15 imperfect magic states with infidelity :math:`p` and outputs a single magic state with infidelity of :math:`35p^3`. 
The distillation circuit is shown below, with the details described in section 3.1 of the paper:

.. figure:: ../_static/demonstration_assets/game_of_surface_codes/15-to-1.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

    15-to-1 distillation protocol. Each :math:`\frac{\pi}{8}` rotation involves a magic state injection with an error-prone magic state.
    In total, we have :math:`4+11` magic states, each with infidelity :math:`p` and output a magic state :math:`|m\rangle` on the
    fifth qubit with infidelity :math:`35p^3`.
    Image source: Daniel Litinski [1].

Because all operations in the protocol are Z measurements, we can use the compact data block design to perform the distillation. 
Another trick the author of [#Litinski]_ proposes is to use the auto-corrected magic state injection protocol below that avoids the additional Clifford :math:`\frac{\pi}{4}` Pauli rotation (and to note that the :math:`\frac{\pi}{2}` Pauli rotation is just a sign flip that can be tracked classically).

.. figure:: ../_static/demonstration_assets/game_of_surface_codes/auto-corrected-non-clifford.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

    The auto-corrected magic state injection protocol avoids the additional Clifford :math:`\frac{\pi}{4}` Pauli rotation from above at the cost of having an additional qubit that is measured.
    However, note that the first two measurements commute and can be performed simultaneously. 
    Image source: Daniel Litinski [1].

Using this injection protocol to perform the non-Clifford :math:`\frac{\pi}{8}` rotations using the error prone magic states, the 15-to-1 protocol on a compact data block is performed in the following way:

.. figure:: ../_static/demonstration_assets/game_of_surface_codes/15-to-1-protocol.png
    :align: center
    :width: 99%
    :target: javascript:void(0)

    The 15-to-1 protocol executed on a compact data block using the auto-corrected magic state injection subroutine in each of the repeating steps. 
    Note that both :math:`P \otimes Z_m` and :math:`Z_m \otimes Y_{|0\rangle}` measurements are performed simultaneously.
    If all :math:`X` measurements on qubits 1-4 in step 23 yield a :math:`+1` result, a magic state is successfully prepared on qubit 5. The probability for failure is roughly :math:`(1-p)^n`.
    Image source: Daniel Litinski [1].

The 15-to-1 distillation protocol produces a magic state in 11🕒 on 11 tiles.

Quantum computer designs
------------------------

The 15-to-1 distillation protocol is the simplest of a variety of protocols each with different characteristics.
The best choice of distillation protocol heavily depends on the error probabilities of the quantum computer in use,
as well as the overall tolerance for errors we allow to still occur.
For example, assume we tolerate a T infidelity of :math:`10^{-10}` and have :math:`p=10^{-4}`, then
the 15-to-1 protocol would suffice as it yields an infidelity of :math:`35p^3 = 3.5 \times 10^{-11} < 10^{-10}`.

Another consideration is to combine data and distillation blocks that match in their maximum time requirements.
Since the 15-to-1 distillation above takes 11🕒 to procude a magic state, there is no point in using the fast or intermediate data blocks, and we can just resort to the compact one.

A minimal setup can be seen below. It consists of 100 logical qubits on 153 tiles in a compact block, as well as a 15-to-1 distillation block using another 11 tiles.

.. figure:: ../_static/demonstration_assets/game_of_surface_codes/minimal-setup.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

    Minimal setup with 100 logical qubits on 153 tiles and 11 extra tiles for a compact distillation block.
    Image source: Daniel Litinski [1].

For a code distance of :math:`d=13` we would require :math:`164 \cdot 2 \cdot d^2 \approx 55k` physical qubits.
An example computation with :math:`10^8` T gates at a code cycle of :math:`1\mu s` would finish in :math:`d \cdot 11🕒 \cdot 10^8 \approx 4h`.

In this setup, a magic state is produced every 11🕒 and takes at most 9🕒 for consumption.
The bottleneck is in the magic state distillation, and overall this setup takes 11🕒 per non-Clifford gate.
The most straight-forward way to speed this up is by adding magic state distillation blocks. Adding just one other distillation block halves the T-gate production time to 5.5🕒.
Now it makes sense to use the intermediate data block design, which takes at most 5🕒 for T-gate consumption:

.. figure:: ../_static/demonstration_assets/game_of_surface_codes/intermediate_setup.png
    :align: center
    :width: 70%
    :target: javascript:void(0)

    Intermediate setup consisting of the intermediate data block and two 15-to-1 distillation blocks on each end.
    Image source: Daniel Litinski [1].

In this case we require 222 tiles, so :math:`222 \cdot 2 \cdot d^2 \approx 75k` physical qubits, and the same computation mentioned before would finish in half the time after about :math:`2h`.


Conclusion
----------

We've been introduced to a high-level description of quantum computing that allows us to reason about space-time trade-offs in FTQC architecture designs.
We have seen some basic prototypes that allow computations involving :math:`10^8` T gates in orders of hours using :math:`55k` or :math:`75k` physical qubits.
With this knowledge, we should be able to follow the more involved tricks discussed in sections 4 and 5 of the Game of Surface Codes paper, that we have not covered in this demo yet.

"""

##############################################################################
#


##############################################################################
#
# References
# ----------
#
# .. [#Litinski]
#
#     Daniel Litinski
#     "A Game of Surface Codes: Large-Scale Quantum Computing with Lattice Surgery"
#     `arXiv:1808.02892 <https://arxiv.org/abs/1808.02892v3>`__, 2018.
#
# Attributions
# ------------
#
# Images from `Game of Surface Codes <https://quantum-journal.org/papers/q-2019-03-05-128/>`__ by Daniel Litinski [#Litinski]_, `CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`__
#
