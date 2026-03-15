r"""Using the BlueQubit (CPU) device with PennyLane
=============================================================

Running large-scale simulations usually requires lots of memory and compute power, regular laptops already struggle above 20 qubits 
and most of the time 30+ qubits are a no-go.
Using the `BlueQubit device <https://app.bluequbit.io/sdk-docs/index.html>`_, PennyLane users can now run circuit simulations on souped up machines and go up to 33 qubits!
Also, Bluequbit uses a custom build of `PennyLane-Lightning <https://docs.pennylane.ai/projects/lightning/en/stable/index.html>`__ that 
enables multi-threading and other configurations to achieve the best possible performance.

Below we will show 2 examples of how to use BlueQubit with PennyLane.
The first is a very simple example building a Bell pair, and the second one is a large 26-qubit circuit that demonstrates the central limit theorem using quantum arithmetic.


.. note::

    To follow along with this tutorial on your own computer, you will need the
    `BlueQubit SDK <https://app.bluequbit.io/sdk-docs/index.html>`_. It can be installed via pip:

    .. code-block:: bash

        pip install bluequbit


.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_bluequbit-pennylane_device.png
    :align: center
    :width: 70%
    :target: javascript:void(0)



Build your PennyLane circuit
----------------------------

Here we will build a simple :doc:`Bell pair </glossary/what-are-bell-states>` and simulate it on the BlueQubit backend.
Later in this tutorial we will show a larger example — a 26-qubit circuit that demonstrates the central limit theorem using a `Draper QFT adder <https://pennylane.ai/qml/demos/tutorial_qft_arithmetics>`__.

Here is the example circuit we will be simulating:
"""

import pennylane as qml
qml.capture.enable()
import matplotlib.pyplot as plt
import numpy as np


def bell_pair():
    qml.Hadamard(0)
    qml.CNOT(wires=(0, 1))
    return qml.probs()

fig = qml.draw_mpl(bell_pair)()


##############################################################################
# Use the BlueQubit device
# ------------------------
# Here are the 3 easy steps to simulate the above circuit on the BlueQubit backend:
#
# 1. Open an account at `app.bluequbit.io <https://app.bluequbit.io/>`__ to get a token. You can also view your submitted jobs here later.
# 2. Initialize the `bluequbit` device with your token.
# 3. Submit the circuit for simulation!

import bluequbit
bluequbit.logger.setLevel("ERROR")

# STEP 2: Initialize the bluequbit device!
# Using a guest token here. Replace it with your own token for a better experience.
bq_dev = qml.device("bluequbit.cpu", wires=2, token="3hmIGLWGKzKdWmxLoJ5F24P3rivGL04d")

bell_qnode = qml.QNode(bell_pair, bq_dev)

# STEP 3: Simulate the circuit!
result = bell_qnode()
print(result)


#############################################
# And that's it! Circuit details and visualizations will also appear in your BlueQubit account after this run.
#
# Now we can run even larger (up to 33-qubit) circuit simulations the same way!
#
# Larger workloads: 26 qubits
# ---------------------------
# Here we will see a much larger example — a 26-qubit circuit.
# Inspired by `Guillermo Allonso's <https://www.pennylane.ai/profile/ketpuntog>`__ PennyLane Demo `Basic arithmetic with the quantum Fourier transform (QFT) <https://pennylane.ai/qml/demos/tutorial_qft_arithmetics>`__,
# which implements a quantum adder in PennyLane, we build our own adder and use it to add together quantum registers.
# 
# In the quantum world we can use the idea of superposition to add multiple numbers at the same time.
# Furthermore, since each number in the superposition can have its own weight, we can use this adder to sum together distributions!
# Below we will use that idea to demonstrate the `central limit theorem <https://en.wikipedia.org/wiki/Central_limit_theorem>`__: we will add together a couple of sequences of independent and identically distributed variables (namely uniformly distributed) 
# and see what their outcome will look like.
# 
# It should take approximately 1 minute to run the code below.



def draper_adder(wires_a, wires_b, kind="fixed", include_qft=True, include_iqft=True):
    """
    Implement the Draper adder for qubit registers of different sizes using PennyLane.

    Args:
        wires_a (list): Wires for the first register (smaller or equal size).
        wires_b (list): Wires for the second register (larger or equal size).
        kind (str): The kind of adder, can be 'half' or 'fixed' (default: 'fixed'). if kind='half' min(wires_b)-1 is the additional qubit of second register
        include_qft (bool): Whether to include the QFT part (default: True).
        include_iqft (bool): Whether to include the inverse QFT part (default: True).
    """
    m = len(wires_a)
    n = len(wires_b)
    if kind == "half":
        wires_sum = [min(wires_b) - 1] + wires_b
    else:
        wires_sum = wires_b
    # QFT part
    if include_qft:
        qml.QFT(wires=wires_sum)
    # Controlled rotations
    for j in range(m):
        for k in range(n - j):
            lam = np.pi / (2**k)
            qml.ControlledPhaseShift(lam, wires=[wires_a[-j-1], wires_sum[j+k]])
    if kind == "half":
        for j in range(m):
            lam = np.pi / (2 ** (j + 1 + n - m))
            qml.ControlledPhaseShift(lam, wires=[wires_a[j], wires_sum[-1]])
    # Inverse QFT part
    if include_iqft:
        qml.adjoint(qml.QFT)(wires=wires_sum)

# Using a guest token here. Replace it with your own token for a better experience.
dev = qml.device("bluequbit.cpu", wires = 26, token="3hmIGLWGKzKdWmxLoJ5F24P3rivGL04d")

@qml.qnode(dev)
def add_4_6qubit_uniforms():
    regs = [list(range(0,6)),
           list(range(6,12)),
           list(range(12,18)),
           list(range(18,26))]
    # make each register uniform 0-63
    for reg in regs:
        for j in range(6):
            qml.Hadamard(reg[-j-1])
    # calcualte sum
    draper_adder(regs[0], regs[3][-6:], kind="half")
    draper_adder(regs[1], regs[3][-7:], kind="half", include_iqft=False)
    draper_adder(regs[2], regs[3], include_qft=False) # skip I=QFT+iQFT, a small optimization
    return qml.probs(wires=regs[3])

res = add_4_6qubit_uniforms()
plt.figure(figsize=(32, 8))
bar = plt.bar(np.arange(len(res)), res)
plt.tick_params(axis='x', labelsize=30)
plt.tick_params(axis='y', labelsize=30)

#############################################
# Wow, that looks like a Gaussian distribution! 
#
# That's exactly what's expected from the central limit theorem — adding together multiple sequences of independent and identically distributed variables approximates the normal distribution.

#############################################
# Conclusion
# ----------
#
# In this tutorial we saw how PennyLane users can run large circuit simulations on BlueQubit's souped up machines.
# We demonstrated this both on a small example, as well as a large 26-qubit simulation where we added together uniform 
# distributions to approximate a normal distribution.
#
# PennyLane users can now simulate large circuits of up to 33 qubits for free using `BlueQubit <https://app.bluequbit.io/sdk-docs/index.html>`_ — we are looking forward to seeing the  
# creative and innovative ways researchers and quantum enthusiasts will be using this capability!

#############################################
# References
# ----------
#
# .. [#Draper2000]
#
#     Thomas G. Draper, "Addition on a Quantum Computer". `arXiv:quant-ph/0008033 <https://arxiv.org/abs/quant-ph/0008033>`__.
#
