r"""How to use wire registers
====================================
A register of wires represents a collection of wires that serve a purpose (e.g., an "estimation" register in :doc:`quantum phase estimation <demos/tutorial_qpe>`) and abstract away the finer details of running quantum
algorithms. In this tutorial, we will explore how you can construct and use wire registers
in PennyLane.
"""

######################################################################
# How to create a wire register
# -----------------------------
#
# The way to construct a wire register is simple --- just use :func:`~.pennylane.registers`. We need to
# pass a dictionary where the keys are the register names and the values are the number of
# wires in each register:
# 

import pennylane as qml
qml.capture.enable()

register = qml.registers({"alice": 1, "bob": 2, "charlie": 3})
print(register)

######################################################################
# The output is a dictionary where the keys are the names of the registers and the
# values are :class:`~.pennylane.wires.Wires` instances.
#
# You can also pass in a dictionary that has nested dictionaries as its values.

nested_register = qml.registers(
    {
        "all_registers": {
            "alice": 1,
            "bob": {"bob1": {"bob1a": 1, "bob1b": 2}, "bob2": 1},
            "charlie": 1,
        }
    }
)
print(nested_register)

######################################################################
# Note that :func:`~.pennylane.registers` flattens any nested dictionaries, and the order of the
# elements is based on the order of appearance and the level of nestedness. For more details on ordering, refer
# to the documentation for :func:`~pennylane.registers`.
#
# Accessing a particular register is the same as accessing any element in a dictionary…

print(nested_register["alice"], nested_register["bob1a"])

######################################################################
# …and you can access a specific wire index via its index in a register.

print(nested_register["all_registers"][2], nested_register["bob1a"][0])

######################################################################
# You can also combine registers using set operations. Here, we use the pipe operator ``|`` to
# perform the union operation on the ``alice`` register and the ``charlie`` register.

new_register = nested_register["alice"] | nested_register["charlie"]
print(new_register)

######################################################################
# For more details on what set operations are supported, refer to the documentation of
# :class:`~.pennylane.wires.Wires`.
#
# A simple example
# ----------------
#
# In this example, we demonstrate how one can implement the swap test with registers.
# The `swap test <https://en.wikipedia.org/wiki/Swap_test>`_
# is an algorithm that calculates the squared inner
# product of two input states. It requires one auxiliary qubit and takes two input states :math:`|\psi\rangle`
# and :math:`|\phi\rangle.` We can think of these components as three registers. Suppose states
# :math:`|\psi\rangle` and :math:`|\phi\rangle` are each represented with 3 wires. In PennyLane
# code, that would be:


import numpy as np

swap_register = qml.registers({"auxiliary": 1, "psi": 3, "phi": 3})

dev = qml.device("default.qubit")

@qml.qnode(dev)
def swap_test():
    # Make "psi" and "phi" state orthogonal to each other
    qml.RX(np.pi/2, swap_register["phi"][0])

    qml.Hadamard(swap_register["auxiliary"])
    for i in range(len(swap_register["psi"])):
        # We can use the union operation to assemble our registers on the fly
        qml.CSWAP(
            swap_register["auxiliary"]
            | swap_register["psi"][i]
            | swap_register["phi"][i]
        )
    qml.Hadamard(swap_register["auxiliary"])
    return qml.expval(qml.Z(wires=swap_register["auxiliary"]))


print(swap_test())

######################################################################
# An advanced example
# -------------------
#
# Using registers can greatly streamline the process of modifying a workflow
# by simplifying wire management. In this example, we use :doc:`quantum phase estimation (QPE) <demos/tutorial_qpe>` to
# calculate the eigenvalues of a Hamiltonian.
# Generally, QPE is described with two sets of registers. One register is known as the
# "estimation" or "measurement" register, and the other is the state register where we apply our
# unitary operators :math:`U.` We can define these registers in PennyLane code:

register = qml.registers({"state": 4, "estimation": 6})

######################################################################
# To build our unitary operator :math:`U,` there are a variety of options. We can opt to use a
# straight-forward block encoding, or choose to use a subroutine like qubitization. Let's opt for
# :class:`~.pennylane.Qubitization`, which means we have to define another preparation register.
# Our registers now look like this:

register = qml.registers({"state": 4, "estimation": 6, "prep": 4})

######################################################################
# Finally, let's define our Hamiltonian. We'll use the `transverse-field Ising model <https://pennylane.ai/datasets/qspin/transverse-field-ising-model>`_ from
# `PennyLane Datasets <https://pennylane.ai/datasets/>`_,
# but feel free to try this with any other Hamiltonian.


[dataset] = qml.data.load(
    "qspin", sysname="Ising", periodicity="open", lattice="chain", layout="1x4"
)
H = dataset.hamiltonians[0]
print(H)

######################################################################
# For QPE to work, we need to initialize the "state" register with an initial state that has good
# overlap with the eigenstate we want the eigenvalue of.

initial_state = dataset.ground_states[0]

######################################################################
# With this, we can now define our QPE circuit:

dev = qml.device("lightning.qubit", wires=14)

@qml.qnode(dev)
def circuit():
    # Initialize state register to initial state
    qml.StatePrep(initial_state, wires=register["state"])

    # Apply Hadamard gate to all wires in estimation register
    for wire in register["estimation"]:
        qml.Hadamard(wires=wire)

    qml.ControlledSequence(
        qml.Qubitization(H, register["prep"]),
        control=register["estimation"],
    )

    qml.adjoint(qml.QFT)(wires=register["estimation"])

    return qml.probs(wires=register["estimation"])

######################################################################
# We'll run our circuit and do some post-processing to get the energy eigenvalue:

output = circuit()
lamb = sum([abs(c) for c in H.terms()[0]])
print("Eigenvalue: ", lamb * np.cos(2 * np.pi * (np.argmax(output)) / 2 ** len(register["estimation"])))

######################################################################
# Changing the number of wires in your estimation register is very easy with registers. 
# The complexity of wire management only gets more difficult
# as you start working with more and more registers. As you start building bigger and more complex
# algorithms, this can quickly become a serious issue!
#
# Conclusion
# ----------
# In this demo, we showed how to construct wire registers and use them to implement quantum
# algorithms. Wire registers provide an elegant way of organizing and managing wires. Often, 
# algorithms are described as acting upon registers and sub-registers; using wire registers
# can greatly streamline the implementation from theory to code. 
#
