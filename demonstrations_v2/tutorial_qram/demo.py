r"""
Intro to quantum random access memory (QRAM)
============================================

Many quantum algorithm that promises advantage or speedup (e.g., quantum machine learning, search,
linear algebra, state preparation) quietly assumes that classical data can be efficiently and
cheaply loaded onto a quantum computer. For some time, this assumption has been justified by
pointing to Quantum Random Access Memory (QRAM) [#qram]_. But in examining this assumption, it turns out that
exactly how you build QRAM directly changes the cost of data loading, in both circuit depth and
qubit count, before hardware or error correction overheads are even considered.

QRAM is an architectural abstraction, where an address register coherently selects a data item and
loads it into a target register. Unlike preparing one fixed input state or compiling a single lookup
table, QRAM asks the architectural question: how can classical memory be queried coherently, in
superposition?

This question is timely; a recent successful small-scale experimental demonstration indicates that
QRAM is more feasible than previously thought.

We’ll use three implementations of QRAM in PennyLane to examine resource tradeoffs:

:class:`~.pennylane.SelectOnlyQRAM`, the direct select-style construction that is closest to
sequential :doc:`QROM <demos/tutorial_intro_qrom>`, :class:`~.pennylane.BBQRAM`, a bucket-brigade architecture that routes a bus qubit
through a binary tree, and :class:`~.pennylane.HybridQRAM`, which interpolates between the two by
combining a select prefix with a smaller bucket-brigade tree. Like classical virtual memory, this
enables querying a larger address space than the qubits allocated. For a collection of :math:`N`
bitstrings :math:`b_0, b_1, \ldots, b_{N-1}` of length :math:`m`, each construction implements the
same logical map:

.. math::

   \operatorname{QRAM}
   \lvert i\rangle
   \lvert 0\rangle^{\otimes m}
   =
   \lvert i\rangle
   \lvert b_i\rangle.

Our goal in this demo is to understand what tradeoff each one makes between qubit count, circuit
depth, and architectural complexity with executable examples in PennyLane.

"""

######################################################################
# The figure below presents a side-by-side conceptual comparison of three different constructions
# included in this demo: (a) a direct address-controlled QRAM implementation for
# ``SelectOnlyQRAM (QROM)``, (b) a binary routing tree for ``BBQRAM``, and (c) a select-prefix plus
# smaller tree for ``Hybrid QRAM``.
#

######################################################################
# .. figure::
#    ../_static/demonstration_assets/intro_to_qram/pennylane-demo-qram-hero.png
#    :alt: Three styles of QRAM compared
#
#
# To keep the circuits readable, we will use four classical records, each encoded as a 3-bit string:
#
# .. math:: [010, 111, 110, 000].
#
# This choice gives us a 2-qubit address register and a 3-qubit target register. The address ``00``
# loads ``010``, the address ``01`` loads ``111``, and so on. We will reuse the same dataset
# throughout the demo so that the differences between the three constructions are entirely due to the
# QRAM architecture itself.
# 

import pennylane as qp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

bitstrings = ["010", "111", "110", "000"]

control_wires = [0, 1]
target_wires = [2, 3, 4]

def decode_probs(probs, num_wires):
    return format(int(np.argmax(probs)), f"0{num_wires}b")


######################################################################
# Select-only QRAM
# ----------------
#
# We start with the most direct construction. :class:`~.pennylane.SelectOnlyQRAM` applies the
# appropriate bit flips to the target register, controlled on the address register. Conceptually, this
# is the QRAM analogue of the select-style QROM story: if the address is :math:`i`, we apply the
# gates that write :math:`b_i` into the target wires.
#
# More concretely, for every stored record whose :math:`j`-th data bit is :math:`1`, the circuit
# applies an address-controlled :math:`X` gate to target wire :math:`j`. This address-controlled
# operation is a multi-controlled :math:`X` gate: all address wires jointly control whether the
# target bit is flipped. Repeating this over all addresses and target positions implements the full
# lookup map.
#
# This is the simplest way to think about data loading, and it is the closest construction to the :doc:`QROM demo <demos/tutorial_intro_qrom>`. 
# The drawback is that the controls are global: every address bit participates in the selection
# logic for every stored record, so the circuit can require many costly multi-controlled operations.
#

######################################################################
# .. figure::
#    ../_static/demonstration_assets/intro_to_qram/pennylane-demo-qram-qrom.png
#    :alt: Select-Only QROM as QRAM
#


@qp.qnode(qp.device("default.qubit"))
def select_only_qram(index):
    qp.BasisEmbedding(index, wires=control_wires)
    qp.SelectOnlyQRAM(bitstrings, control_wires=control_wires, target_wires=target_wires)
    return qp.probs(wires=target_wires)

for i in range(len(bitstrings)):
    output = decode_probs(select_only_qram(i), len(target_wires))
    print(f"address {i:02b} -> {output}")
    
@partial(qp.transforms.decompose, max_expansion=1)
@qp.qnode(qp.device("default.qubit"))
def select_only_qram_draw(index):
    qp.BasisEmbedding(index, wires=control_wires)
    qp.SelectOnlyQRAM(bitstrings, control_wires=control_wires, target_wires=target_wires)
    return qp.probs(wires=target_wires)

qp.draw_mpl(select_only_qram_draw, style="pennylane")(2)
plt.show()


######################################################################
# The logical action is exactly what we want, but the implementation is still rather expensive. We can
# inspect the decomposition at the gate level by compiling the circuit to a CNOT-based gate set and
# then using PennyLane’s resource estimation tools.
#


@partial(qp.compile, basis_set="CNOT")
@qp.qnode(qp.device("default.qubit"))
def compiled_select_only_qram(index):
    qp.BasisEmbedding(index, wires=control_wires)
    qp.SelectOnlyQRAM(bitstrings, control_wires=control_wires, target_wires=target_wires)
    return qp.probs(wires=target_wires)


select_specs = qp.specs(compiled_select_only_qram)(0)["resources"]
print("Total qubits:", len(control_wires + target_wires))
print("One-qubit gates:", select_specs.gate_sizes.get(1, 0))
print("Two-qubit gates:", select_specs.gate_sizes.get(2, 0))


######################################################################
# Bucket-brigade QRAM
# -------------------
#
# The bucket-brigade idea reorganizes the problem. Instead of using one large global selection gadget,
# it stores routing information in a binary tree [#selectqram]_. At a high level, the query has three
# stages:
#
# 1. **Address loading.** The address qubits are routed into the binary tree, where they set the
#    direction information along the path corresponding to the queried address.
# 2. **Data retrieval.** A bus qubit is routed through the tree using the stored direction
#    information. At the bottom of the tree, the circuit applies gates determined by the classical
#    data stored at that leaf, then routes the bus back out to load the requested bitstring into the
#    target register.
# 3. **Address unloading.** The address-loading operation is reversed so that the routing tree is
#    restored and the work wires can be reused.
#
# In PennyLane, :class:`~.pennylane.BBQRAM` uses one bus wire plus three wires per internal node of
# the routing tree:
#
# - one direction wire,
# - one left-port wire, and
# - one right-port wire.
#
# If the address register has :math:`n` wires, then the work register must contain
#
# .. math:: 1 + 3(2^n - 1)
#
# additional wires. For our 2-qubit address register, that means ten work wires.
#
# Here, we present a bucket-brigade tree diagram for a 3-bit address, with one address entry
# highlighted. The address qubits are routed into the QRAM tree and temporarily stored for data
# retrieval in the subsequent step.
#

######################################################################
# .. figure::
#    ../_static/demonstration_assets/intro_to_qram/pennylane-demo-qram-bbqram.png
#    :alt: Bucket brigade QRAM (BBQRAM)
#

bb_num_work_wires = 1 + 3 * ((1 << len(control_wires)) - 1)
bb_work_wires = list(range(5, 5 + bb_num_work_wires))

@qp.qnode(qp.device("default.qubit"))
def bucket_brigade_qram(index):
    qp.BasisEmbedding(index, wires=control_wires)
    qp.BBQRAM(
        bitstrings,
        control_wires=control_wires,
        target_wires=target_wires,
        work_wires=bb_work_wires,
    )
    return qp.probs(wires=target_wires)

print(f"BBQRAM uses {len(bb_work_wires)} work wires.")
for i in range(len(bitstrings)):
    output = decode_probs(bucket_brigade_qram(i), len(target_wires))
    print(f"address {i:02b} -> {output}")


######################################################################
# The tradeoff is now clear. :class:`~.pennylane.BBQRAM` replaces large multi-controlled target
# updates with local routing operations, but it needs a substantial auxiliary memory architecture to
# do so. This is precisely the kind of width-depth tradeoff that motivates QRAM design: depending on
# the hardware model, extra qubits may be preferable to deeper control logic.
#


@partial(qp.compile, basis_set="CNOT")
@qp.qnode(qp.device("default.qubit"))
def compiled_bucket_brigade_qram(index):
    qp.BasisEmbedding(index, wires=control_wires)
    qp.BBQRAM(
        bitstrings,
        control_wires=control_wires,
        target_wires=target_wires,
        work_wires=bb_work_wires,
    )
    return qp.probs(wires=target_wires)


bb_specs = qp.specs(compiled_bucket_brigade_qram)(0)["resources"]
print("Total qubits:", len(control_wires + target_wires + bb_work_wires))
print("One-qubit gates:", bb_specs.gate_sizes.get(1, 0))
print("Two-qubit gates:", bb_specs.gate_sizes.get(2, 0))


######################################################################
# Hybrid QRAM
# -----------
#
# :class:`~.pennylane.HybridQRAM` combines the previous two ideas. We split the address into a
# select prefix of size :math:`k` and a bucket-brigade suffix of size :math:`n-k`. The prefix
# chooses one block of the classical data, and a smaller bucket-brigade tree is reused inside that
# block. The PennyLane template follows the circuit-level select/bucket-brigade hybridization idea in
# [#hybridqram]_, while hybrid QRAM also appears in hardware-oriented architectures such as
# [#hardwareefficient]_.
#
# This gives us a tunable family of constructions:
#
# - small :math:`k` means more bucket-brigade behavior and a larger routing tree,
# - large :math:`k` means more select-style behavior and less routing overhead.
#
# Notably, both :class:`~.pennylane.SelectOnlyQRAM` and :class:`~.pennylane.BBQRAM` are two special cases (extreme cases) for hybrid QRAM,
# with k=n and k=0, respectively.
#
# For our 2-qubit address register, the only nontrivial choice is :math:`k=1`: one address bit acts
# as the select prefix, while the remaining bit routes through a depth-1 bucket-brigade tree.
#
# The following figure illustrates the hybrid decomposition using a minimal :math:`k=1, n=2`
# example. The high-order address bit acts as a block selector, partitioning the memory into two
# blocks, while the remaining address bits are routed through a shared bucket-brigade subtree.
# Operationally, this construction can be viewed as reusing a smaller :math:`n=2` bucket-brigade
# QRAM query twice, once for each block, with the :math:`k=1` selector determining which block
# output is activated. This example makes explicit how the hybrid design trades replicated block-level
# structure for a reusable intra-block QRAM query path.
#

######################################################################
# .. figure::
#    ../_static/demonstration_assets/intro_to_qram/pennylane-demo-qram-hybrid-qram.png
#    :alt: Hybrid QRAM
#

k = 1
n_tree = len(control_wires) - k
hybrid_num_work_wires = 2 + 3 * ((1 << n_tree) - 1)
hybrid_work_wires = list(range(5, 5 + hybrid_num_work_wires))

@qp.qnode(qp.device("default.qubit"))
def hybrid_qram(index):
    qp.BasisEmbedding(index, wires=control_wires)
    qp.HybridQRAM(
        bitstrings,
        control_wires=control_wires,
        target_wires=target_wires,
        work_wires=hybrid_work_wires,
        k=k,
    )
    return qp.probs(wires=target_wires)

print(f"HybridQRAM uses k = {k} and {len(hybrid_work_wires)} work wires.")
for i in range(len(bitstrings)):
    output = decode_probs(hybrid_qram(i), len(target_wires))
    print(f"address {i:02b} -> {output}")


######################################################################
# Because :class:`~.pennylane.HybridQRAM` exposes the parameter :math:`k`, it gives us a continuum
# between the other two constructions. In larger examples, this can be a practical design knob: we can
# spend more auxiliary qubits to shrink the effective routing problem, or keep the work register
# smaller and accept a deeper bucket-brigade component.
#


@partial(qp.compile, basis_set="CNOT")
@qp.qnode(qp.device("default.qubit"))
def compiled_hybrid_qram(index):
    qp.BasisEmbedding(index, wires=control_wires)
    qp.HybridQRAM(
        bitstrings,
        control_wires=control_wires,
        target_wires=target_wires,
        work_wires=hybrid_work_wires,
        k=k,
    )
    return qp.probs(wires=target_wires)


hybrid_specs = qp.specs(compiled_hybrid_qram)(0)["resources"]
print("Total qubits:", len(control_wires + target_wires + hybrid_work_wires))
print("One-qubit gates:", hybrid_specs.gate_sizes.get(1, 0))
print("Two-qubit gates:", hybrid_specs.gate_sizes.get(2, 0))


######################################################################
# The tunable parameter :math:`k` is more meaningful once the address register is larger. To keep
# the demo lightweight, the next cell does not compile a larger circuit. It only uses the work-wire
# formula
#
# .. math:: 2 + 3(2^{n-k} - 1)
#
# to show how the bucket-brigade part shrinks as more address bits are handled by the select prefix.
# Full gate-level compilation is still useful, but for larger examples it can dominate the runtime of
# a tutorial notebook.
#
num_address_wires = 4
num_target_wires = len(target_wires)

print("k | tree depth | work wires | total wires")
print("--|------------|------------|------------")
for k_value in range(num_address_wires):
    tree_depth = num_address_wires - k_value
    num_work_wires = 2 + 3 * ((1 << tree_depth) - 1)
    total_wires = num_address_wires + num_target_wires + num_work_wires
    print(f"{k_value} | {tree_depth:10d} | {num_work_wires:10d} | {total_wires:10d}")


######################################################################
# Comparing the three constructions
# ---------------------------------
#
# At the logical level, all three templates implement the same map. What changes is the mechanism used
# to realize it, and therefore the asymptotic scaling. Let :math:`N=2^n` be the number of stored
# records, where :math:`n` is the number of address wires, and let :math:`m` be the bitstring
# length.
#
# +------------------------+------------------------+------------------------+------------------------+
# | Construction           | Width scaling          | Depth / gate-cost      | Main tradeoff          |
# |                        |                        | intuition              |                        |
# +========================+========================+========================+========================+
# | :class:`~.pennylane.SelectOnlyQRAM`     | :math:`O(n+m)`         | :math:`O(2^n m)`       | Minimal width;         |
# |                        |                        | multi-controlled       | exponential select     |
# |                        |                        | writes                 | cost                   |
# +------------------------+------------------------+------------------------+------------------------+
# | :class:`~.pennylane.BBQRAM`             | :math:`O(2^n+n+m)`     | Active routing path    | Large width; local     |
# |                        |                        | length :math:`O(n)`,   | routing architecture   |
# |                        |                        | plus address           |                        |
# |                        |                        | load/unload overhead   |                        |
# +------------------------+------------------------+------------------------+------------------------+
# | :class:`~.pennylane.HybridQRAM`         | :math:`O(2^{n-k}+n+m)` | Select over            | Tunable interpolation  |
# |                        |                        | :math:`2^k` blocks and | controlled by          |
# |                        |                        | route through a tree   | :math:`k`              |
# |                        |                        | of depth :math:`n-k`   |                        |
# +------------------------+------------------------+------------------------+------------------------+
#
# Here, “depth / gate-cost intuition” is meant asymptotically: exact compiled depths depend on the
# target gate set, decomposition choices, and how much parallelism the hardware model allows. For our
# toy example, we can still place the compiled resource counts side by side. These numbers should be
# read as a small-instance comparison rather than an asymptotic benchmark: with only four records,
# constant overheads can dominate, especially for the bucket-brigade and hybrid constructions.
#

resource_table = {
    "SelectOnlyQRAM": {
        "total_qubits": len(control_wires + target_wires),
        "one_qubit_gates": select_specs.gate_sizes.get(1, 0),
        "two_qubit_gates": select_specs.gate_sizes.get(2, 0),
    },
    "BBQRAM": {
        "total_qubits": len(control_wires + target_wires + bb_work_wires),
        "one_qubit_gates": bb_specs.gate_sizes.get(1, 0),
        "two_qubit_gates": bb_specs.gate_sizes.get(2, 0),
    },
    "HybridQRAM": {
        "total_qubits": len(control_wires + target_wires + hybrid_work_wires),
        "one_qubit_gates": hybrid_specs.gate_sizes.get(1, 0),
        "two_qubit_gates": hybrid_specs.gate_sizes.get(2, 0),
    },
}

for name, summary in resource_table.items():
    print(name)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print()


######################################################################
# Even in this small example, the qualitative pattern is already visible.
# :class:`~.pennylane.SelectOnlyQRAM` keeps the qubit count low but leans on address-wide control
# logic. :class:`~.pennylane.BBQRAM` introduces a dedicated memory architecture that can replace
# some of that global control with local routing. :class:`~.pennylane.HybridQRAM` then turns this
# into a tunable tradeoff through the parameter :math:`k`. For very small tables, this extra
# structure can look expensive; the purpose of the hybrid construction is to expose a design knob that
# becomes more useful as the address space grows.
#
# Conclusion
# ----------
#
# QRAM is not a single circuit pattern, but a family of architectures for loading classical data into
# quantum registers. :class:`~.pennylane.SelectOnlyQRAM`, :class:`~.pennylane.BBQRAM`, and :class:`~.pennylane.HybridQRAM` all implement the same
# abstract operation,
#
# .. math::
#
#    \lvert i\rangle \lvert 0\rangle
#    \mapsto
#    \lvert i\rangle \lvert b_i\rangle,
#
# but they do so with different architectural tradeoffs. The direct select-style construction keeps
# the width small, but pays for it with exponentially many address-wide controlled operations.
# Bucket-brigade QRAM moves in the opposite direction: it spends :math:`O(2^n)` auxiliary wires on a
# routing tree so that a query follows a local path through memory. Hybrid QRAM sits between these
# extremes, using :math:`k` select-prefix bits to shrink the routing tree while increasing the
# select-style part of the computation.
#
# The main takeaway is that the logical data-loading task does not determine a unique circuit
# architecture. Instead, QRAM design is a choice between different kinds of resource tradeoff: narrow
# circuits with global controls, wide circuits with local routing, or hybrid circuits that interpolate
# between the two. This is precisely why QRAM is an interesting topic for quantum software: once the
# logical task is fixed, the implementation details become a question of architecture and resources
# rather than correctness alone.
#
# References
# ----------
#
# .. [#qram] Vittorio Giovannetti, Seth Lloyd, and Lorenzo Maccone,
#    "Quantum random access memory",
#    `arXiv:0708.1879 <https://arxiv.org/abs/0708.1879>`__, 2007.
#
# .. [#expBBQRAM] Fanhao Shen, Yujie Ji, Debin Xiang, Yanzhe Wang, Ke Wang, Chuanyu Zhang, Aosai Zhang, Yiren Zou, Yu Gao, Zhengyi Cui, Gongyu Liu, Jianan Yang, Yihang Han, Jinfeng Deng, Anbang Wang, Zhihong Zhang, Hekang Li, Qiujiang Guo, Pengfei Zhang, Chao Song, Liqiang Lu, Zhen Wang, and Jianwei Yin,
#    "Experimental realization of the bucket-brigade quantum random access memory",
#    `arXiv:2506.16682 <https://arxiv.org/abs/2506.16682>`__, 2025.
#
# .. [#selectqram] Connor T. Hann, Gideon Lee, S. M. Girvin, and Liang Jiang,
#    "Resilience of quantum random access memory to generic noise",
#    `arXiv:2012.05340 <https://arxiv.org/abs/2012.05340>`__, 2012.
#
# .. [#hybridqram] Shifan Xu, Connor T. Hann, Ben Foxman, Steven M. Girvin, and Yongshan Ding,
#    "Systems Architecture for Quantum Random Access Memory",
#    `arXiv:2306.03242 <https://arxiv.org/abs/2306.03242>`__, 2023.
#
# .. [#hardwareefficient] Connor T. Hann, Chang-Ling Zou, Yaxing Zhang, Yiwen Chu,
#    Robert J. Schoelkopf, Steven M. Girvin, and Liang Jiang,
#    "Hardware-efficient quantum random access memory with hybrid quantum acoustic systems",
#    `https://arxiv.org/abs/1906.11340 <https://arxiv.org/abs/1906.11340>`__, 2019.
#
