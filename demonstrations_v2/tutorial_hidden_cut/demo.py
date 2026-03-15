r"""
The hidden cut problem for locating unentanglement
==================================================
"""

######################################################################
# One of the most provocative and counterintuitive features of quantum physics is *entanglement*, a
# form of correlation that can exist between quantum systems. To illustrate the striking nature of
# entanglement, imagine two entangled qubits, with the first one located at Xanadu’s headquarters in
# Toronto, and the second located in the
# `JADES-GS-z13-0 <https://en.wikipedia.org/wiki/JADES-GS-z13-0>`__ galaxy, the furthest galaxy ever
# measured. The proper cosmological distance between these qubits is about 33 billion light-years.
# Nevertheless, because they are entangled, the measurement outcome of the qubit in Toronto will
# necessarily be correlated with the measurement of the qubit in JADES-GS-z13-0! How this is possible
# when it takes light itself 33 billion years to travel between the qubits is one of the most
# philosophically loaded questions at the heart of quantum foundations.
# 
# Despite entanglement being so philosophically provocative, it’s somewhat surprising that it is so
# ubiquitous: given a random state of a two-component quantum system, it’s almost certain that the
# two components will be entangled. For this reason, it’s sometimes more interesting when a state is
# *not* entangled rather than when it is! For example, when `building a quantum computer at
# Xanadu <https://www.xanadu.ai/photonics>`__, we spend a ton of effort to ensure that our qubits are
# as *unentangled* as possible with their environment!
# 
# In this demo we’ll investigate *unentanglement* more closely. More specifically, we’ll consider a
# problem related to unentanglement, called the *hidden cut problem*. In this problem we assume that
# we’re given a state consisting of many components. As we discussed, it’ll generally be the case that
# most of these components are entangled with one another. However, in the hidden cut problem we are
# guaranteed that it’s possible to split the components into two groups, so that between the two
# groups there is *no* entanglement. The problem asks us to find this “hidden cut” that splits the
# state up into two *unentangled* pieces.
# 
# Let’s define the hidden cut problem a bit more precisely. First we need to define *unentanglement*.
# We say that a quantum state :math:`|\psi\rangle` describing a system with two parts, :math:`A` and
# :math:`B`, is *unentangled*, if it can be written as a tensor product
# 
# .. math::
# 
# 
#    |\psi\rangle = |\psi_A\rangle\otimes |\psi_B\rangle,
# 
# where :math:`|\psi_A\rangle` is a state of system :math:`A` and :math:`|\psi_B\rangle` is a state of
# system :math:`B`. We also use the term *separable* or *factorizable* to describe an unentagled
# state. We’ll usually not bother writing the tensor product sign and just write
# :math:`|\psi\rangle = |\psi_A\rangle |\psi_B\rangle`.
# 
# Now let’s suppose :math:`|\psi\rangle` is a state of :math:`n`-qubits. We’re told it’s possible to split
# the qubits into two unentangled subsets, :math:`S` and :math:`\bar S`,
# 
# .. math::
# 
# 
#    |\psi\rangle = |\psi_S\rangle |\psi_{\bar S}\rangle,
# 
# but we aren’t told what :math:`S` and :math:`\bar S` are. The hidden cut problem asks us to determine :math:`S` and :math:`\bar S`, given access to :math:`|\psi\rangle`.
# Following Bouland *et al.* [#Bouland2024]_, in this demo
# we’ll develop a quantum algorithm that solves this problem!
# 

######################################################################
# Creating an unentangled state
# -----------------------------
# 
# Before we can solve the hidden cut problem, we first need a state :math:`|\psi\rangle` to solve it on!
# First we define a function ``random_state()`` that creates a random state with a specified number of
# qubits. We do this by creating a :math:`2^n` by :math:`2^n` random unitary and taking the first row.
# Because all the rows (and columns) in a unitary matrix have norm equal to 1, this defines a valid
# quantum state.
# 

# imports
import galois
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
qml.capture.enable()
from scipy.stats import unitary_group

# set random seed
np.random.seed(123)

# function to create random state
def random_state(n_qubits):
    dim = 2**n_qubits
    return unitary_group.rvs(dim)[0]

######################################################################
# However we can’t just use this function to construct our state :math:`|\psi\rangle`, because the random
# state created by the function will almost certainly *not* be unentagled. Instead, we’ll define a function
# ``separable_state()`` that takes a list of qubit partitions as input, creates a random state for
# each partition, and tensors them together into a separable state.
# 

def separable_state(partitions):
    # Number of qubits
    n_qubits = sum(len(part) for part in partitions)

    # Sort partitions
    partitions = [sorted(part) for part in partitions]

    # Create random state for each partition
    partition_states = [(part, random_state(len(part))) for part in partitions]
    
    # Initialize full state
    full_state = np.zeros(2**n_qubits, dtype=complex)
    
    # Fill in amplitudes
    for idx in range(2**n_qubits):
        # Convert idx to binary string
        bits = format(idx, f'0{n_qubits}b')
        
        # Calculate amplitude as product of partition amplitudes
        amplitude = 1.0
        for part, state in partition_states:
            # Extract partition bits, convert to decimal, update amplitude
            part_bits = ''.join(bits[q] for q in part)
            part_idx = int(part_bits, 2)
            amplitude *= state[part_idx]
        
        full_state[idx] = amplitude
    
    return full_state

######################################################################
# We’ll use this to create a 5-qubit state :math:`|\psi\rangle` in which qubits :math:`S=\{0,1\}` are
# unentangled with qubits :math:`\bar S=\{2,3,4\}`.
# 

partitions = [[0,1], [2,3,4]]
state = separable_state(partitions)
n = int(np.log2(len(state)))
print(f'Created {n} qubit state with qubits {partitions[0]} unentangled from {partitions[1]}.')

######################################################################
# Now imagine we’re given :math:`|\psi\rangle` but aren’t told that qubits 0,1 are unentangled from
# qubits 2,3,4. How could we figure this out? This is the hidden cut problem: given a many-qubit
# quantum state, identify which qubits are unentangled with which other qubits. Now we’ll develop a
# quantum algorithm that solves this problem, and then we’ll implement it in Pennylane and see that it
# works!
# 

######################################################################
# Hidden cut problem as a hidden subgroup problem
# -----------------------------------------------
# 

######################################################################
# The key to solving the hidden cut problem is to recast it as a *hidden symmetry problem*, or in more
# mathematical language a *hidden subgroup problem* (HSP). Then we can use a famous quantum algorithm
# for solving HSPs to solve the hidden cut problem. The traditional HSP algorithm is useful for
# finding symmetries of functions :math:`f(x)`, i.e. figuring out for what values of :math:`a` we have
# :math:`f(x+a) = f(x)` for all :math:`x`. However, we’re interested in a *state* :math:`|\psi\rangle` and
# not a *function* :math:`f(x)`, so we’ll instead use a modified version of the HSP algorithm, called
# StateHSP, which finds *symmetries of states*.
# 
# We’ll explain the StateHSP algorithm below, but first let’s see how we can recast the hidden cut
# problem as one of finding a hidden symmetry of a state. To see this, remember our example state
# :math:`|\psi\rangle=|\psi_{01}\rangle |\psi_{234}\rangle`. Now consider two copies :math:`|\psi\rangle |\psi\rangle`
# of this state. We can visualize this as
#
# .. figure:: ../_static/demonstration_assets/hidden_cut/qubits.png
#    :align: center
#    :width: 80%
#
#    Figure 2. A schematic of the quantum state :math:`|\psi\rangle |\psi\rangle`, where :math:`|\psi\rangle=|\psi_{01}\rangle |\psi_{234}\rangle`.
#
# The top row corresponds to the first copy of :math:`|\psi\rangle`, and the bottom row to the second
# copy. In each row, qubits 0 and 1 are disconnected from qubits 2, 3, and 4. This schematically
# indicates the fact that in each :math:`|\psi\rangle` qubits 0, 1 are unentangled from qubits 2, 3, 4.
# 
# Now consider what happens when we swap some qubits in the top row with the corresponding qubits in
# the bottom row. We can denote which pairs of qubits we’re swapping with a 5-bit string. For example,
# the bitstring 10101 corresponds to swapping the qubits in positions 0, 2, and 4 in the top row with
# qubits 0, 2, and 4 in the bottom row. Because there are :math:`2^5=32` possible 5-bit strings, there are 32
# possible swap operations we can perform. Interestingly, the set of all 32 5-bit strings forms a
# mathematical *group* under bitwise addition.
#
# .. admonition:: Group
#     :class: note
#
#     A group is a set of elements that has:
#       1. an operation that maps two elements a and b of the set into a third element of the set, for example c = a + b,
#       2. an "identity element" e such that e + a = a for any element a, and
#       3. an inverse -a for every element a, such that a + (-a) = e.
#
#     A group is called "Abelian" if a + b = b + a for all a and b, otherwise it is called non-Abelian.
#
# We’ll call the group of 5-bit strings :math:`G`. We can now ask: which elements of :math:`G`
# correspond to swap operations that leave the state :math:`|\psi\rangle|\psi\rangle` invariant? These
# operations are the symmetries of :math:`|\psi\rangle|\psi\rangle` and the corresponding bitstrings form a
# subgroup :math:`H` of :math:`G`. For example the identity element 00000 corresponds to performing no
# swaps at all. This is clearly a symmetry, so 00000 is in :math:`H`. On the other hand 11111 swaps
# *all* the qubits in the top row with the corresponding qubits in the bottom row, so in effect it
# just swaps the entire first copy of :math:`|\psi\rangle` with the entire second copy of
# :math:`|\psi\rangle`. This is clearly also a symmetry, so 11111 is also in :math:`H`. Are there any
# other elements in :math:`H`? Stop and think about it!
# 
# In fact, there are two more elements in :math:`H`: 11000 and 00111. 11000 corresponds to swapping
# the :math:`|\psi_{01}\rangle` component of the first copy of :math:`|\psi\rangle` with the same component
# of the second copy of :math:`|\psi\rangle`, and 00111 corresponds to swapping the
# :math:`|\psi_{234}\rangle` components between the two copies. Because in each copy of :math:`|\psi\rangle`
# the :math:`|\psi_{01}\rangle` and :math:`|\psi_{234}\rangle` components are completely unentangled,
# after either of these swaps the full state remains unchanged, namely :math:`|\psi\rangle|\psi\rangle`. So the
# symmetry subgroup is :math:`H = {00000, 11111, 11000, 00111}`. We’ll call this a *hidden* symmetry
# subgroup because it wasn’t given to us - we had to find it!
# 
# Now, a shorthand way to write any group is to specify a set of *generators* - group elements that can
# be added together to generate any other element of the group. For :math:`H` the generators are 11000
# and 00111: we can add either generator to itself to get the identity 00000, and we can add the
# generators to each other to get 11111. Here’s the important point: notice that the generators of
# :math:`H` *directly* tell us the unentangled components of
# :math:`|\psi\rangle=|\psi_{01}\rangle|\psi_{234}\rangle`! The first generator 11000 has 1s in bits 0 and 1:
# this corresponds to the first unentangled component :math:`|\psi_{01}\rangle`. And the second
# generator 00111 has 1s in bits 2, 3, 4: this corresponds to the second unentangled component
# :math:`|\psi_{234}\rangle`. So finding the hidden subgroup :math:`H` gives us the unentangled
# components - it solves the hidden cut problem!
# 
# Now that we've recast the hidden cut problem as a problem of finding a hidden subgroup :math:`H`,
# let's see how the StateHSP algorithm can be used to find :math:`H`. The general algorithm works for
# any abelian group :math:`G`, but here we’ll just focus on the case where :math:`G` is the group of
# :math:`n`-bit strings, since this is the case that’s relevant to solving the hidden cut problem.
# 
# The algorithm involves running a quantum circuit, taking measurements, and postprocessing the
# measurements. The circuit involves three :math:`n`-qubit registers. Registers 2 and 3 are each
# initialized to :math:`|\psi\rangle`, and register 1 is initialized to the all-:math:`|0\rangle` state. We
# call register 1 the *group register* because we’ll use it to encode elements of the group :math:`G`.
# For example if :math:`n=5` the group element 10101 of :math:`G` would be encoded as
# :math:`|10101\rangle`.
# 
# After this register initialization, the StateHSP circuit involves three steps:
#
# 1. Apply a **Hadamard** to each qubit in the group register. This puts the group register in a uniform superposition of all group elements, which up to normalization we can write as :math:`\sum_{g\in G} |g\rangle`.
#
# 2. Apply a **controlled SWAP** operator. This acts on all 3 registers by mapping :math:`|g\rangle|\psi\rangle|\psi\rangle` to :math:`|g\rangle\text{SWAP}_g(|\psi\rangle|\psi\rangle)`. Here, :math:`\text{SWAP}_g` performs swaps at the positions indicated by :math:`g`; for example if :math:`g=10101` then qubits 0, 2 and 4 in the first copy of :math:`|\psi\rangle` will get swapped with the corresponding qubits in the second copy of :math:`|\psi\rangle`.
#
# 3. Apply a **Hadamard** to each qubit in the group register.
# 
# Finally we measure the group register. Here’s the circuit diagram:
#
# .. figure:: ../_static/demonstration_assets/hidden_cut/circuit.png
#    :align: center
#    :width: 80%
#
#    Figure 2. Hidden cut circuit
#
# We’ll implement this in Pennylane, and then we’ll show how the measurement results can be
# postprocessed to find the hidden subgroup :math:`H` that encodes the hidden cut!
# 

######################################################################
# Solving the hidden cut problem in Pennylane
# -------------------------------------------
# 

######################################################################
# Let’s implement this circuit in Pennylane! We’ll use a device with ``shots=100``: this will run the
# circuit 100 times and record a 5-bit measurement for each run. We’ll store these measurements in an
# array ``M``:
# 

dev = qml.device('default.qubit')

@qml.set_shots(100)
@qml.qnode(dev)
def circuit():
    # Initialize psi x psi in registers 2 and 3
    qml.StatePrep(state, wires=range(n, 2*n))
    qml.StatePrep(state, wires=range(2*n, 3*n))
                            
    # Hadamards
    for a in range(n):
        qml.Hadamard(a)

    # Controlled swaps
    for c in range(n):
        a = c + n
        b = c + 2*n
        qml.ctrl(qml.SWAP, c, control_values=1)(wires=(a,b))

    # Hadamards
    for a in range(n):
        qml.Hadamard(a)

    # Measure
    return qml.sample(wires=range(n))

M = circuit()

print(f'The shape of M is {M.shape}.')
print(f'The first 3 rows of M are:\n{M[:3]}')

######################################################################
# Now let’s process the measurement results :math:`M` to determine the hidden subgroup :math:`H`! This
# postprocessing step is common to all hidden subgroup algorithms. The key fact that connects the
# measurement results :math:`M` to the hidden subgroup :math:`H` is this: the elements of :math:`H`
# are the vectors that are orthogonal to all measurements (i.e. rows) in :math:`M`. Since we’re
# working with the group of bitstrings, two bitstrings are orthogonal if their dot product mod 2 is
# equal to 0. For example 10101 and 11100 are orthogonal since their dot product is
# :math:`2\equiv0\mod 2`, while 10101 and 11111 are *not* orthogonal, since their dot product is
# :math:`3\equiv1\mod 2`.
# 
# So to get :math:`H` we just have to find the binary vectors :math:`\vec b` orthogonal to every row
# of :math:`M`. Mathematically we write this as :math:`M\vec b = 0`, where all operations are assumed
# to be performed mod 2. In linear algebra lingo we say that the solutions :math:`\vec b` to this
# equation form the *nullspace* of :math:`M`. We can find this nullspace using basic
# linear algebra techniques.
# 
# Instead of doing the algebra by hand though, here we’ll use the ``galois`` python library, which can
# perform linear algebra mod 2. To ensure that operations on :math:`M` are performed mod 2, we first
# convert it to a ``galois.GF2`` array. The “GF2” stands for `“Galois field of order
# 2” <https://en.wikipedia.org/wiki/Finite_field>`__, which is a fancy way of saying that all
# operations are performed mod 2.
# 

M = galois.GF2(M)

print(f'The shape of M is {M.shape}.')
print(f'The first 3 rows of M are:\n{M[:3]}')

######################################################################
# So ``M`` is the same array as before. Let’s check that addition is performed mod 2 by adding rows 1
# and 2 of ``M``:
# 

r1 = M[1]
r2 = M[2]

print(f'     r1 = {r1}')
print(f'     r2 = {r2}')
print(f'r1 + r2 = {r1 + r2}')

######################################################################
# Looking at the middle column we see that :math:`1+1=0`, so addition is mod 2 as desired!
# 
# Now we can finally compute the nullspace of ``M``, which will give us the hidden subgroup :math:`H`.
# We can do this easily now that ``M`` is a ``galois.GF2`` array just by calling ``M.null_space()``.
# In fact this method doesn’t return all of the bitstrings in the nullspace, but instead saves space
# by only returning the generators of the nullspace.
# 

M.null_space()

######################################################################
# Because the nullspace of :math:`M` equals :math:`H`, we conclude that the generators of :math:`H`
# are 11000 and 00111. If we didn’t know that :math:`|\psi\rangle` could be factored as
# :math:`|\psi\rangle=|\psi_{01}\rangle|\psi_{234}\rangle`, the generators would directly tell us the factors!
# So we have solved the hidden cut problem for our state :math:`|\psi\rangle`!
# 

######################################################################
# The power of hidden symmetries
# ------------------------------
# 

######################################################################
# We solved the hidden cut problem—finding the factors of a multi-qubit quantum state—by approaching
# it from the perspective of symmetries. The key insight was to recognize that the question
# *What are the unentangled factors of the state?* can be rephrased as *What is the hidden symmetry of the state?*.
# This rephrasing transformed the hidden cut problem into a *hidden subgroup problem*, which we could solve using a modification of the standard algorithm for HSPs that
# allows for finding symmetries of *states* rather than functions.
# 
# In fact, many of the most well-known problems that benefit from access to a quantum computer are
# also instances of an HSP! In some cases, like with the hidden cut problem, it isn’t obvious at first glance
# that the problem involves finding a hidden symmetry. The most famous example of this
# is the problem of factoring large integers, a problem with fundamental importance in cryptography.
# It’s not at all obvious that this problem is related to finding the symmetry of a function, but with
# some clever algebra it can be phrased in this way, and hence solved efficiently using the HSP
# algorithm. As a speculative side comment, it’s interesting that the problem of *factoring states* and
# the problem of *factoring integers* can both be phrased as HSPs! Is there something about
# *factoring* problems that enables them to be expressed as HSPs? Are there other important factoring
# problems that can also be recast as HSPs and solved on a quantum computer? Or is this just a
# complete coincidence? I invite you to think about this - maybe you'll discover a deep
# connection!
# 
# Less speculatively, there definitely *is* one deep and generalizable lesson that we should take away
# from this hidden cut demo: the *power of looking for hidden symmetries*. This goes well
# beyond quantum computing. In fact, some of the most significant discoveries in physics are just
# recognitions of a hidden symmetry! For example: recognizing the symmetries of fundamental particles
# led to the development of the standard model of particle physics; recognizing the symmetry of
# systems in different inertial reference frames led to discovery of special relativity; and
# recognizing the symmetry between freefalling and accelerating objects led to the discovery of
# general relativity. It's clear that looking for hidden symmetries is a very powerful approach, both in
# quantum computing and beyond!
# 
#
# References
# ------------
#
# .. [#Bouland2024] 
#    
#     Adam Bouland, Tudor Giurgică-Tiron, John Wright, "The State Hidden Subgroup Problem and an Efficient Algorithm for Locating Unentanglement",  
#     `STOC '25 p.463-470. <https://doi.org/10.1145/3717823.3718118>`__, 2025
#