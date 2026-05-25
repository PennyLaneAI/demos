r"""Active volume compilation
=============================

In this demo, we will explore the concept of an "active volume" of a quantum computation,
and how compilation can exploit this concept to reduce the resources required to execute the
computation on a suitable machine—an active volume quantum computer.
We will look at circuits, :doc:`ZX-diagrams <demos/tutorial_zx_calculus>`, and a new type of
circuit representation called logical networks.
This demo is directly based on a seminal paper by Litinski and Nickerson [#Litinski2022]_.
As the original work already does a great job at presenting the concepts and compilation
techniques at multiple levels of detail and with plenty of visualizations, we will aim to
take a complementary perspective, and walk slowly through a specific compilation example.

For readers unfamiliar with ZX-diagrams, it will be useful (but not necessary) to take a look at
our :doc:`introduction to ZX-calculus <demos/tutorial_zx_calculus>` first.
Further, it can be instructive to get a :doc:`primer on lattice surgery <demos/tutorial_lattice_surgery>`
and dive into the :doc:`Game of Surface Codes <demos/tutorial_game_of_surface_codes>`, but this
is not a requirement either.

.. figure:: _static/demo_thumbnails/opengraph_demo_thumbnails/pennylane-demo-active-volume-open-graph.png
    :align: center
    :width: 65%
    :target: javascript:void(0)

Introduction
------------

`Quantum compilation <https://pennylane.ai/compilation/>`__ is fundamentally about bridging the
gap between high-level descriptions of
quantum algorithms and low-level instructions that are actually executable on quantum hardware.
In addition to the bare translation between those representations, it also aims to optimize
numerous key metrics of the quantum program during the compilation process. Typically, these
metrics focus on specific properties of the program such as qubit count and various gate counts,
which often stand in contention against each other. To optimize the program cost effectively,
we'd ideally like a metric that expresses the combined cost of all resources used by a program.
One such metric is the **spacetime volume** cost, which in fault-tolerant architectures can be
understood as the total error-corrected qubits (space) taken up by the computation, times the
total error correction cycles (time) required to perform it.

To understand the impact of different quantum architectures and compilation strategies, we'll
often put an *algorithmic cost* metric in relation to its
*implementation cost* on hardware. For instance, the strategies
described by Litinski's :doc:`Game of Surface Codes <demos/tutorial_game_of_surface_codes>`
incur an implementation cost, specified by the **spacetime volume** of the compiled program,
of roughly twice the algorithmic cost, given by the **circuit volume** of the original quantum
circuit.

Intuitively, the **circuit volume** can be understood as the total “area” taken up by a
circuit, as depicted below. The crucial insight in the concept of **active volume** then is
the idea that not all of this “area” is dedicated to performing useful computation. In fact,
ladder circuits like those used in reversible arithmetic may consist in large part
(if not the majority) of idling qubits!
We can thus partition a circuit into computationally *active* volume and *idle* volume:

.. figure:: _static/demonstration_assets/active_volume/active-vs-idle.png
    :align: center
    :width: 85%
    :target: javascript:void(0)

    Active and idle volumes are represented by areas occupied by gates (green)
    and areas without gates (red), respectively, in a standard circuit diagram.
    Adapted from [1].

In this demo, we will discuss the active volume compilation framework introduced by
Litinski and Nickerson [#Litinski2022]_ that allows one to reduce the implementation cost
of an algorithm to being proportional to the active volume of the circuit. This is in contrast to
the framework of [#Litinski2018]_, where the implementation cost is proportional to the circuit
volume.
This framework combines the language of the ZX-calculus with the inherently-quantum technique of
state teleportation, as well as an abstraction of topologically error-corrected quantum computers.
Concretely, the approach assumes that we execute on a so-called **active volume computer**.

.. admonition:: Active volume computer
    :class: note

    .. _AV info box:

    The cost model used by active volume compilation is founded on an abstract error-corrected
    quantum computer, the active volume computer.
    It is made up of a fixed number of *qubit modules*, each of which stores a logical qubit.
    Half of the modules are designated computation modules, in which information is processed,
    while the other half consists of dedicated memory modules, which allow for intermediate
    storage and routing of information between computational steps.

    .. figure:: _static/demonstration_assets/active_volume/active-volume-computer.png
        :align: center
        :width: 65%
        :target: javascript:void(0)

    The following characteristics are crucial assumptions about the computer that
    determine our cost model:

    #. The information content of logarithmically-distanced qubit modules can be
       exchanged quickly/cheaply. They are "quickswappable".
    #. Individual qubit modules can be prepared quickly/cheaply in the state :math:`|0\rangle` or
       :math:`|+\rangle`.
       They can also be measured quickly/cheaply in the Pauli-:math:`X` or Pauli-:math:`Z` basis.
    #. Pairs of qubit modules within some specified range :math:`r` can be prepared quickly/cheaply
       in the Bell state :math:`|\phi\rangle = \tfrac{1}{\sqrt{2}} (|00\rangle +|11\rangle)`.
       They can also be measured quickly/cheaply in the Bell basis.
    #. Choosing the measurement basis of one or two qubits dynamically costs one *reaction time*
       unit. The relation between the reaction time and the time taken by the above cheap
       operations is a property of the computer.
    #. Qubit modules can execute so-called logical blocks, which is a slow/expensive operation.
       Computations are represented by logical networks (see below) made up of connected logical
       blocks. The qubit modules executing connected blocks must be within range.
    #. Qubit modules can be grouped and occupied for some time to distill magic states. This is
       a slow/expensive operation as well.

    A more detailed characterization is provided in Sec. 1 of [#Litinski2022]_.

    |

    Given the characterization above, active volume compilation aims to maximize the utilization
    of the computational qubit modules, because they execute the slow operations (logical blocks
    and magic state distillation). Memory qubit modules are allowed to sit idle, providing
    intermediate storage and space for routing information through quickswaps. Note that
    arbitrary long-range communication in the active volume computer is achieved through layers
    of quickswaps, which in typical scenarios remain low-depth and thus cheap in the above cost
    model.

In the following, we will demonstrate how to compile the circuit of a subroutine
step by step into a so-called **logical network**, which is a representation that allows
the active volume computer to use its computing resources with high efficiency.
For this, we will transform the circuit into a ZX-diagram, impose a set of rules that
lead us to an **oriented** ZX-diagram, and then rewrite it into the aforementioned
logical network. This representation will allow us to capture the essence of quantum information
processing without the restrictive representation as quantum circuits, while enforcing enough
structure to arrive at a program that can be executed by an active volume computer.

After compiling to an exemplary logical network, we will discuss teleportation-based
parallelization, to be used between multiple networks, as well as the **reaction time**,
another important concept to understand the interaction between logical networks effecting
non-Clifford gates.

The concepts are presented in a bottom-up approach and focus on the required steps to compile
our concrete Clifford circuit example.  For a top-down overview of the active
volume framework, Sec. 1 of [#Litinski2022]_ is a great source, while
Secs. 2-6 work through the compilation of increasingly complex circuit components.

Without further ado, let's get started!

The input: a quantum circuit
----------------------------

The active volume framework can compile arbitrary quantum circuits, which we consider to
be the input; Clifford gates are supported directly, as we will see for our example below.
Discrete non-Clifford gates such as ``T`` or ``Toffoli`` are implemented in a surface
code-corrected quantum computer by ingesting magic states via multi-qubit measurements.
And continuous non-Clifford gates such as arbitrary-angle Pauli product rotations (PPRs)
are supported via standard discretization techniques like
`Gridsynth <https://arxiv.org/abs/1403.2975v3>`__,
`repeat-until-success circuits <https://arxiv.org/abs/1404.5320>`__,
`channel mixing <https://quantum-journal.org/papers/q-2023-12-18-1208/>`__, or
`phase gradient decompositions <https://pennylane.ai/compilation/phase-gradient>`__.
While this already enables the compiler to handle universal quantum circuits, compilation
via an intermediate representation in terms of Pauli product measurements (PPMs) allows for
a smooth integration with other compilation techniques such as those in the
Game of Surface Codes by Litinski. Correspondingly, you will find many circuits re-expressed
in terms of PPMs in the active volume paper.

Our example circuit: CNOT ladder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The subroutine we will compile step by step is a ladder of CNOT gates, which may be used
to create multi-qubit GHZ states, for example. As the CNOT gate is part of the Clifford group,
so is the full ladder. This simple example will allow us to focus on the formalization of active
volume compilation into logical networks.
To make matters concrete, we will be concerned with compiling the following circuit:

.. figure:: _static/demonstration_assets/active_volume/example-circuit.png
    :align: center
    :width: 35%
    :target: javascript:void(0)

The labels :math:`|a\rangle` through :math:`|d\rangle` for the four input and output
qubits will be handy to keep track later on.

From circuits to ZX-diagrams
----------------------------

The first step we take is to transform the quantum circuit to a ZX-diagram, the object
representing linear maps in the :doc:`ZX-calculus <demos/tutorial_zx_calculus>`.
We briefly outline some key building blocks and properties of the calculus here, and defer
to the linked tutorial and its references for more details.

The elementary building blocks of a ZX-diagram are so-called spiders, drawn as a
blue (X-type spider) or orange (Z-type spider) vertex and a number of legs. Each spider
represents a linear map itself and composing spiders into diagrams is as easy as connecting their
legs, creating edges between the vertices. Unconnected legs of a
diagram symbolize inputs and outputs of the represented linear map.
Some basic examples are summarized in the following overview figure.

.. figure:: _static/demonstration_assets/active_volume/zx-calculus-overview.png
    :align: center
    :width: 100%
    :target: javascript:void(0)

    Basic building blocks and transformation rules of the ZX-calculus.
    Adapted from [1].

ZX-diagrams can, in principle, express universal quantum circuits (and even more maps that are
not circuits), but for active volume compilation we only need to express Clifford circuits
and Pauli product measurements (PPMs), whose measurement bases could be classically conditioned.
Therefore, we may restrict ourselves to phase-free spiders and in particular the building blocks
shown above. The main reason for this is that non-Clifford operations, such as T or Toffoli
gates, are realized via
:doc:`magic state injection <demos/tutorial_mcm_introduction#t-gadget-in-pennylane>`
in surface code computations, and that classical mixtures in the form of classically
conditioned operations can be translated into PPMs with a conditionally chosen basis.

ZX-diagram for the CNOT ladder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As we can see in the previous figure, the CNOT gate is easily expressed in terms of a diagram
with one X and one Z spider, each with three legs. This means that we can just as easily rewrite
our CNOT ladder into a ZX-diagram:

.. figure:: _static/demonstration_assets/active_volume/example-zx-basic.png
    :align: center
    :width: 35%
    :target: javascript:void(0)

We find six spiders, three of each type, with three legs each.

At this point, we didn't only exchange circuit symbols for coloured circles, though. In the
language of the ZX-calculus, the internal edges that connect vertices no longer have a temporal
meaning, so that the geometry of the diagram becomes irrelevant. Only the represented
graph carries meaningful information, as long as we associate the unconnected legs of the
diagram with fixed inputs and outputs.
This allows us to rewrite our diagram as follows (rotating it to make it easier to display):

.. figure:: _static/demonstration_assets/active_volume/example-zx-flat.png
    :align: center
    :width: 55%
    :target: javascript:void(0)

Note how the inherent time ordering of the CNOT ladder is no longer represented in this diagram.
While this looks like a small, parallelized implementation of the ladder, we will need to work a
little more to guarantee that the diagram actually represents a map that can be implemented
conveniently on a quantum computer.

Imposing structure: oriented ZX-diagrams
----------------------------------------

Expressing the quantum circuit as a ZX-diagram with little structure allows for
convenient manipulation, but it also causes trouble: ultimately we want to compile the circuit
for a quantum computer that operates with logical qubits implemented as surface
code corrected patches of physical qubits, so we will need to gain back some structure.
For this, Litinski and Nickerson introduce **oriented ZX-diagrams**, which must satisfy the
following rules, illustrated below:

#. Each spider must have two, three or four legs.
#. Each spider has six ports (north (N), up (U), east (E), south (S), down (D), and west (W))
   and at most one leg can be connected to each port.
   For each spider, the two ports in one of three pairs (N, S), (U, D), and (E, W) must both
   be unoccupied. This pair is called the **orientation** of the spider and we denote it
   by N, U or E accordingly.
#. Input (output) legs must connect to the down (up) port of a spider.
#. Edges (internal legs) must connect to ports from the same of the three pairs at both spiders.
   That is, they must connect to the same port of the two spiders they connect, or to
   opposite ports.
#. Edges (internal legs) must connect spiders of the same type (X/Z) and same orientation, or
   of different types and different orientations. (For Hadamarded edges, same (different) types
   with different (same) orientations must be connected.)

.. figure:: _static/demonstration_assets/active_volume/oriented-zx-rules.png
    :align: center
    :width: 100%
    :target: javascript:void(0)

These rules may seem quite artificial and confusing. This is because they impose constraints
on the—otherwise quite unstructured—ZX-diagram that originate in the structure of the surface
code and the arrangement of the overall computation in space and time.
Concretely, the ports named after cardinal directions (N, E, S, W) correspond to the four
directions in which a square patch encoding a logical surface code qubit can interact with
other patches, and the remaining directions (U, D) symbolize the time axis, motivating
the input/output constraint (3.).

The constraints on edges then encode the requirements for the oriented ZX-diagram to represent
a computation that can be arranged in (2+1) dimensions (surface+time) without collisions and
without inconsistencies in the boundaries of the logical qubits and their parity check measurements.
The achieved consistency allows us to perform the required joint measurements for
lattice surgery to implement multi-qubit operations.

Exercise: oriented ZX-diagram of a CNOT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before we turn to our example circuit, let's produce a valid oriented ZX-diagram for a single
CNOT as a warmup exercise. The standard ZX-diagram has two spiders, so we are tempted to
simply take this diagram and assign ports to each of the three legs, at each spider.
Due to the input/output constraint, the qubit state labels go into the U and D ports:

.. figure:: _static/demonstration_assets/active_volume/cnot-oriented-zx-incomplete.png
    :align: center
    :width: 20%
    :target: javascript:void(0)

Now, we may choose any of the remaining ports (N, E, S, W) of the Z spider (orange) for
the internal edge. We know that it must connect to the same or opposite port at the X spider
(blue) due to rule (4.), forcing the orientation of both spiders to be the same.
However, as they differ in type (X vs. Z), such a connection is forbidden by rule (5.).
We thus already run out of options for the internal edge. What can we do to fix this issue?
The solution lies in the identity ``─●─ = ──`` for ZX-diagrams; before we orient the diagram,
we may insert additional spiders with two legs, in place of a simple leg. This allows us to
gain more room in order to gradually change the occupied ports of the oriented spiders,
before switching between spider types.
As it turns out, inserting a single spider in the CNOT diagram is not sufficient,
and we need to insert two spiders:

.. figure:: _static/demonstration_assets/active_volume/cnot-oriented-zx-extended.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

Let us try to connect the edges again. We connect the two Z spiders with a (S-N) edge, giving
them both an E-orientation. Coming from the right side, we connect the X spiders with an
(E-W) edge, giving them an N-orientation:

.. figure:: _static/demonstration_assets/active_volume/cnot-oriented-zx-extended-1.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

Here we indicated the orientation of the spiders with internal lines, serving as a mnemonic
that these ports are "occupied" already.
Now we can use the U and D ports of the central spiders to route the last edge. We choose
both U ports, because this will be a convenient choice below, and arrive at the following
oriented ZX-diagram for a single CNOT:

.. figure:: _static/demonstration_assets/active_volume/cnot-oriented-zx-complete.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

This is a valid oriented ZX-diagram for a single CNOT gate. Note that the spiders we had
to insert above directly correspond to additional information-processing steps, and will take up
additional qubits in the execution of the CNOT. This may come as a surprise, given that
a CNOT is a simple two-qubit Clifford gate. However, note that the active volume computer
architecture operates with :doc:`lattice surgery <demos/tutorial_lattice_surgery>`, and
executing a CNOT with lattice surgery also requires two additional logical qubits as intermediate
bridge space. Thus, the oriented ZX-diagram simply reproduces the true cost for a CNOT in this
type of architecture.

Note that we could also connect the input state :math:`|b\rangle` to the D port of the left
X spider:

.. figure:: _static/demonstration_assets/active_volume/cnot-oriented-zx-alternative.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

This is because we could have inserted the additional blue spider on the output leg of the existing
one, rather than between the inserted orange and the existing blue spiders.
For this, the orientation of the spiders could be kept as-is.

Oriented ZX-diagram of the CNOT ladder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Having figured out the oriented ZX-diagram for a single CNOT gate, we turn to the ZX-diagram
of the CNOT ladder. As you may guess, compared to the unoriented diagram we arrived at above, we
will again need to insert more nodes.
One way to do this would be to simply copy the oriented diagram for the single CNOT three times
in the vertical (time) direction, and to route outputs of one copy into the inputs of the next.
This would lead to an arrangement with "depth" three.
However, since we want to maximize the qubit utilization on our computer and avoid idling qubits,
we can leverage the flexibility offered by the ZX-calculus to instead express the computation in a
single logical step, and in doing so generate an optimized, parallelized version of the CNOT ladder.
With a similar reasoning as for the single CNOT, we find that we need to insert six nodes overall,
arriving at the oriented ZX-diagram:

.. figure:: _static/demonstration_assets/active_volume/example-oriented-zx-complete.png
    :align: center
    :width: 90%
    :target: javascript:void(0)

In order to connect the “individual” CNOT networks side-by-side, we have moved some of the input
qubit legs to what was previously an “internal” node. This is allowed as there is no preference
for where to place logical inputs and outputs in a network. Note how some of the qubits appear
to be routed laterally through the network, hinting at the parallelization of what were previously
sequential gates. As for the single CNOT, this is not a unique solution.
We are almost done with the compilation of the CNOT ladder; one last step will be needed to
transform the oriented ZX-diagram into a logical network, which can then be executed on an
active volume computer.

Oriented ZX-diagram to logical network
--------------------------------------

The last step required to arrive at logical networks is comparatively small; the major effort
is already done with the orientation of the ZX-diagram.

We introduce **logical networks**, which are just oriented ZX-diagrams with spiders
drawn as hexagons to mark the six ports more clearly, with a modification of rule (4.) above:

4'. Edges between two oriented spiders must connect to **the same port** of both spiders.

From now on, we refer to these hexagonal spiders inside a logical network as **logical blocks**.

Depending on the concrete circuit we are compiling, this change might just lead to minor
modifications such as relabeling the ports of two connected blocks, or it might lead to cascades of
re-routing that eventually force us to insert a new block, even.
**Logical networks** are diagrams, or networks, of logical blocks. We label the logical blocks
uniquely and replace drawing edges between them by annotating the ports with the labels of the
blocks they connect to. This way, logical blocks can be arranged, or scheduled, more freely, as
long as they respect the connectivity specified by the range of the active volume computer model
(also see the `active volume computer info box <AV info box_>`_).

Logical network for the CNOT ladder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Looking at the oriented ZX-diagram of the CNOT ladder above, we see that the inserted spiders
with two legs are connected to the original three-legged spiders with edges that do not satisfy
the modified rule (4'.) yet, indicated in red below.

.. figure:: _static/demonstration_assets/active_volume/example-oriented-zx-redline.png
    :align: center
    :width: 90%
    :target: javascript:void(0)

However, in each case, we may simply use the opposite port of those
two-legged spiders:

.. figure:: _static/demonstration_assets/active_volume/example-oriented-zx-greenline.png
    :align: center
    :width: 90%
    :target: javascript:void(0)

We then arrive at a valid logical network by performing a few cosmetic
changes: first, change the shape of the spiders to hexagons, with each corner corresponding
to a port. Second, enumerate the hexagons for referencing. Any labels work, really. Third and last,
cut the edges between hexagons and instead draw shortened legs, annotated by the label of the
hexagon they connect to.
For our example, we find the logical network:

.. figure:: _static/demonstration_assets/active_volume/example-logical-network.png
    :align: center
    :width: 95%
    :target: javascript:void(0)

And this already concludes the compilation process of this simple Clifford operation, arriving at
an optimized construction for the CNOT ladder subroutine, which was parallelized
into a single time step and can now be used as a building block for a computation executed
on the active volume computer.

In order to schedule multiple logical networks in a resource-efficient manner, we will need a
second ingredient: state teleportation.

Parallelizing logical networks with state teleportation
-------------------------------------------------------

In the CNOT ladder example circuit, which has a fundamentally sequential appearance in the
circuit picture, logical network compilation parallelized it by exploiting the lack of time
ordering in ZX-diagrams, combining the effects of the three CNOTs into a single effect.
If we want to parallelize multiple subroutines, however,
without having to re-compile their joint logical network, we can do this procedurally through
state teleportation, using a so-called bridge qubit. This technique parallelizes non-commuting
operations without breaking physics, and it pays off because Bell state preparation and
measurements are assumed to be fast on an active volume computer
(see `info box at the top <AV info box_>`_).

Physical soundness
~~~~~~~~~~~~~~~~~~

For the first point, let's consider a somewhat more abstract setup, as is done in Fig. 10b),c)
in [#Litinski2022]_. Suppose we want to sequentially execute two two-qubit gates :math:`A` and
:math:`B` that overlap on one qubit and do not commute:

.. figure:: _static/demonstration_assets/active_volume/teleportation-start.png
    :align: center
    :width: 40%
    :target: javascript:void(0)

We also assume that we can track the impact that :math:`B` has on Pauli operators, i.e., what
happens to a Pauli operator when pulling it through :math:`B`. For our use case, this will be
easily satisfied, because :math:`B` will be a Clifford operation that simply turns Paulis into
new Pauli operators.

For the parallelization, we first insert a state teleportation circuit between :math:`A` and
:math:`B` using a pair of auxiliary qubits prepared in an entangled Bell state
:math:`|\phi\rangle=\tfrac{1}{\sqrt{2}}(|00\rangle +|11\rangle)`, see the
:doc:`demo on state teleportation <demos/tutorial_teleportation>` for details:

.. figure:: _static/demonstration_assets/active_volume/teleportation-insert-teleport.png
    :align: center
    :width: 55%
    :target: javascript:void(0)

Next, we pull the classically controlled correction gates, which are Pauli operators, through
the gate :math:`B`, and obtain new correction gates :math:`C_Z=B^\dagger X B` and
:math:`C_X=B^\dagger Z B`:

.. figure:: _static/demonstration_assets/active_volume/teleportation-commuted.png
    :align: center
    :width: 45%
    :target: javascript:void(0)

At this point, we pulled :math:`B` to the front, effectively parallelizing it with :math:`A`.
As we can see, the state teleportation together with the ability to track correction gates through
:math:`B` makes it possible to implement the two non-commuting operations :math:`A` and :math:`B`
simultaneously, without violating any physical laws or causal dependencies.

In the context of active volume compilation, this parallelization technique allows us to schedule
multiple logical networks at the same time step, even if they act on overlapping qubits and
do not commute. This is essential to remove idle volume from the computation, and it can be done
at low cost; the creation of qubit pairs in a Bell state and the Bell basis measurement are
assumed to be much cheaper than arbitrary logical operations (see `info box <AV info box_>`_),
because they can just be woven into the measurements, i.e., the code cycles of the error
correction code via lattice surgery.
Similarly, Pauli correction gates are tracked in software throughout, so that the only
true price we are paying for the parallelization is the extra qubits. Finally,
additional memory space to store the bridge qubit is usually available in the memory qubit modules
making up half of the active volume computer.

What did we gain?
-----------------

You may ask what we gained by translating a quantum circuit to the logical network representation,
and why we would prefer custom blocks—with new complicated rules governing them—over a
good old circuit diagram.
First, logical networks encode the capabilities and restrictions of the
`active volume computer <AV info box_>`_, which is protected by the surface code and
utilizes lattice surgery. This compatibility sets logical networks apart from other representations.
Second, logical networks can be parallelized via state teleportation, as we just discussed. While
this is true for components of a quantum circuit as well, there is a third feature that makes
parallelization more useful:
Logical networks can be divided into multiple networks, which then can be executed in
sequence. We will not go into detail about this, but refer to Sec. 2 and Fig. 11 of
[#Litinski2022]_. This allows us to split up a network that otherwise would be too large to be
executed in parallel with a previous network.
We may sketch the idea behind the scheduling via splitting and parallelization:

.. figure:: _static/demonstration_assets/active_volume/scheduling.png
    :align: center
    :width: 90%
    :target: javascript:void(0)

As we can see, the combined effect of these features of logical networks allows us to
(almost) fill the computational region of the active volume computer with logical blocks at
each time step, creating a dense computation pattern and thus removing (almost) all idle volume.
We thus solve the crucial shortcoming of circuit diagrams discussed at the beginning of the demo!

Another feature of logical network synthesis is the reduction of the computation to its essential
logical effects. As a consequence, re-synthesizing combinations of networks into a single new
network can reduce the number of logical blocks that need to be executed. This
already happens when compiling a Toffoli gate that is realized via CCZ state injection
(see Fig. 14 in [1]).

Reactive measurements and reaction time
---------------------------------------

While the compilation techniques presented above are very powerful, there is an important
restriction we haven't considered so far.
For Clifford operations, we will only ever encounter Pauli corrections that need to be applied
conditioned on measurement outcomes. Such corrections in
turn affect the outcome (and only the outcome) of other measurements, which
then determine whether Pauli corrections need to be applied, and so on. But since we're tracking
Pauli corrections in software [#PauliFrame]_, there is no immediate need to do anything with
measurement results, and we can happily continue to apply operations in parallel or any
other order without worrying about data dependencies between them. So do we *ever* need to affect
the quantum state using the information from measurements and Pauli corrections?

To answer this question, we need to consider non-Clifford operations. As mentioned further up,
these are implemented via `magic state injection <https://pennylane.ai/qml/glossary/what-are-magic-states>`__.
As a general rule, operations implemented via
Pauli measurements require a correction that is one level lower in the Clifford hierarchy.
So Clifford gates require a Pauli correction (similar to our corrections in the teleportation
circuit), 1st order non-Clifford gates like T and Toffoli require a Clifford correction, 2nd
order non-Clifford gates require 1st order ones, and so on.

.. figure:: _static/demonstration_assets/active_volume/pprs.png
    :align: center
    :width: 90%
    :target: javascript:void(0)

    The circuit style used in Litinski's papers colour-codes the different members of the
    Clifford hierarchy: Pauli gates (or :math:`\pi/2` rotations) in gray, (non-Pauli) Clifford gates
    (or :math:`\pi/4` rotations) in orange, (1st order) non-Clifford gates (or :math:`\pi/8`
    rotations) in green, and measurements in blue. Both diagrams show the decomposition of Pauli
    rotations into Pauli measurements plus associated corrections of the lower hierarchy orders.
    Image source: Daniel Litinski [#Litinski2018]_.

This is where we quickly run into issues, as Clifford corrections are generally already considered
too complex to be tracked classically. That means that we'll have to physically implement them,
and it is at this point that we require a concrete value of the measurement result (accounting
for any prior Pauli corrections). In this sense, the Clifford corrections impose a limit on the
ability to parallelize, as part of the computation will still need to happen sequentially.

Doing corrections in-line as depicted in the above circuits has actually been shown to be
inefficient, so a key optimization is to move the correction away from the data qubits and onto
the auxiliary magic states. This gives us significantly more flexibility to schedule computation
efficiently as we don't have to do the corrections right away. We can hold on to the consumed
magic state until it is convenient (or necessary) to apply the correction. Such gate
implementations are called “auto-corrected”, and will look as follows for non-Clifford Pauli
rotations:

.. figure:: _static/demonstration_assets/active_volume/auto-magic.png
    :align: center
    :width: 55%
    :target: javascript:void(0)

    Auto-corrected non-Clifford Pauli rotation. Adapted from [1,2].

In essence, we have swapped the Clifford correction for a multiplexed measurement, where the
measurement basis is adaptively chosen based on some classical input. A measurement whose basis
depends on prior measurement results is called a **reactive measurement**. The speed at which
these measurements, and associated classical processing, can be performed is a fundamental limiting
factor to how fast a computation can be executed, regardless of how many qubits are available
for parallelism. This limitation is captured in the **reaction time** of an (active volume) quantum
computer. An execution scheme for quantum programs that is limited by these reactive
measurements—rather than, say, the circuit depth—is thus known as implementing
**reaction-limited computation**.

.. figure:: _static/demonstration_assets/active_volume/reaction-limited-computation.png
    :align: center
    :width: 90%
    :target: javascript:void(0)

    Example circuit with four gates, translated to state injections, teleportation-based
    parallelization and Pauli product measurements. The duration of the computation is
    fundamentally limited through the reaction time needed to process classical bits into
    dynamically chosen measurement bases. The limiting path is marked in magenta.

If you are familiar with the Game of Surface Codes paper referenced earlier, you may wonder
how the scheme described there differs from what has been presented here. After all, Litinski
already used teleportation then (cf. fig. 25 in [#Litinski2018]_) to parallelize Pauli circuits
and achieve reaction-limited computation (also referred to as “Fowler's time-optimal scheme”).
Fundamentally, parallelization via teleportation always trades (execution) time for (memory) space.
The crucial difference, intuitively, is that Litinski's earlier techniques have to provide the
additional space as brand new qubits, since each operation is assumed, in principle, to span the
full width of a circuit (a characteristic that is reinforced by the techniques presented in the
paper).
Parallelization thus happens on the *layer structure*, where entire circuit layers are executed
simultaneously using the original qubit count multiple times.
Meanwhile, active volume computation tries to reuse already available, but idle, qubit space at a
fine-grained *operation level*, and in doing so maximizes the *efficiency* of the computer.

Conclusion
----------

With this, we conclude our introduction to active volume compilation.
We saw how it combines the best of ZX-diagrams and quantum circuits into a neat abstraction for
quantum computation, arriving at logical blocks and networks as the central objects to
represent the computation. The compilation is thus tailored to the second key component of
the framework, an active volume computer that is able to execute logical networks.
After compiling standard gates and subroutines to logical networks,
as we did here for the CNOT ladder, a quantum computation can be expressed by the combination of
those networks, which can be scheduled in parallel via teleportation, even if they act on the
same logical qubits and do not commute. As a consequence, the only
idle volume left in the compiled computation is due to memory modules, which take up half of
the layout of an active volume computer, and are used for intermediate caching and routing
of information only. This compression of quantum circuits reduces the cost of the computation
dramatically.
For details on the abstraction of the active volume computer, many additional
compilation examples, and details about the implementation in a photonic quantum computer,
we recommend to read the original paper as well.
Additionally, you may want to

- dive into the background of :doc:`lattice surgery <demos/tutorial_lattice_surgery>`,
- learn about Pauli-based computation without active volume considerations in the
  :doc:`game of surface codes <demos/tutorial_game_of_surface_codes>`,
- and understand Pauli frame tracking from the perspective
  of :doc:`classical simulation <demos/tutorial_clifford_circuit_simulations>`.

References
-----------

.. [#Litinski2022]

   Daniel Litinski and Naomi Nickerson.
   "Active volume: An architecture for efficient fault-tolerant quantum computers with limited non-local connections",
   `arXiv:2211.15465 <https://arxiv.org/abs/2211.15465>`__, 2022.

.. [#Litinski2018]

   Daniel Litinski
   "A Game of Surface Codes: Large-Scale Quantum Computing with Lattice Surgery"
   `arXiv:1808.02892 <https://arxiv.org/abs/1808.02892v3>`__, 2018.

.. [#Gidney2018]

   Craig Gidney. "Halving the cost of quantum addition",
   `Quantum 2, 74 <https://quantum-journal.org/papers/q-2018-06-18-74/>`__, 2018,
   `arXiv:1709.06648 <https://arxiv.org/abs/1709.06648>`__, 2017.

.. [#PauliFrame]

   Riesebos et al. "Pauli Frames for Quantum Computer Architectures",
   Proceedings of the 54th Annual Design Automation Conference 2017 (DAC '17),
   `doi:10.1145/3061639.3062300 <https://doi.org/10.1145/3061639.3062300>`__, 2017.

Attributions
------------

Some images are taken from `Game of Surface Codes <https://quantum-journal.org/papers/q-2019-03-05-128/>`__
by Daniel Litinski [#Litinski2018]_, `CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`__.
"""
