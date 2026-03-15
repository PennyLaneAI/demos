r"""
How to use quantum arithmetic operators
=======================================

Classical computers handle arithmetic operations like addition, subtraction, multiplication, and exponentiation with ease. 
For instance, you can multiply large numbers on your phone in milliseconds!
`Quantum computers <https://pennylane.ai/qml/what-is-quantum-computing/>`__ can handle these operations too, but their true value lies beyond basic calculations.

Quantum arithmetic plays a crucial role in more advanced quantum algorithms, 
serving as fundamental building blocks in their design and execution. For example:

1. In `Shor's algorithm <https://pennylane.ai/codebook/10-shors-algorithm/>`__ quantum arithmetic is crucial for performing modular exponentiation [#shor_exp]_. 

2. :doc:`Grover's algorithm <demos/tutorial_grovers_algorithm>` might need to use quantum arithmetic to construct oracles, as shown in `this related demo <https://pennylane.ai/qml/demos/tutorial_qft_arithmetics/>`_.

3. Loading data or preparing initial states on a quantum computer often requires several quantum arithmetic operations [#sanders]_.

With PennyLane, you will see how easy it is to build quantum arithmetic operations as subroutines for your algorithms. 
Knowing your way around these operators could be just the thing to streamline your algorithm design!

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_how_to_use_quantum_arithmetic_operators.png
    :align: center
    :width: 70%
    :target: javascript:void(0)

In-place and out-place arithmetic operations
------------------------------------------
Let's begin by defining the terms *in-place* and *out-place* in the context of arithmetic operators. 
In-place operators, like the :class:`~.pennylane.Adder` and :class:`~.pennylane.Multiplier`, directly modify the original quantum state by updating a 
specific register's state. In contrast, out-place operators, such as the :class:`~.pennylane.OutAdder` and :class:`~.pennylane.OutMultiplier`, 
combine multiple states and store the result in a new register, leaving the original states unchanged. Both kinds of operators are
illustrated in the following figure.

.. figure:: ../_static/demonstration_assets/how_to_use_quantum_arithmetic_operators/in_outplace.png
  :align: center
  :width: 90%

In quantum computing, all arithmetic operations are inherently `modular <https://en.wikipedia.org/wiki/Modular_arithmetic>`_. 
The default behavior in PennyLane is to perform operations modulo :math:`2^n`, 
where :math:`n` is the number of wires in the register. For example, if :math:`n=6`, we have :math:`(32 + 43) = 75 = 11 \mod 64`,  (since :math:`2^6 = 64`). 
That means that quantum registers of :math:`n` wires can  represent numbers up to :math:`2^n`. 
However, users can specify a custom value smaller than this default. It's important to keep this modular behavior 
in mind when working with quantum arithmetic, as using 
too few qubits in a quantum register could lead to overflow issues. We will come back to this point later. 

Addition operators
~~~~~~~~~~~~~~~~~~

There are two addition operators in PennyLane: :class:`~.pennylane.Adder` and :class:`~.pennylane.OutAdder`.

The :class:`~.pennylane.Adder` operator performs an in-place operation, adding an integer value :math:`k` to the state of the wires :math:`|x \rangle`. It is defined as:

.. math::

   \text{Adder}(k) |x \rangle = | x+k \rangle.

On the other hand, the :class:`~.pennylane.OutAdder` operator performs an out-place operation, where the states of two 
wires, :math:`|x \rangle` and :math:`|y \rangle`, are 
added together and the result is stored in a third register:

.. math::

   \text{OutAdder} |x \rangle |y \rangle |0 \rangle = |x \rangle |y \rangle |x + y \rangle.

To implement these operators in PennyLane, the first step is to define the `registers of wires <https://pennylane.ai/qml/demos/tutorial_how_to_use_registers/>`_
we will work with. We define wires for input registers, the output register, and also additional ``work_wires`` that will be important when we later discuss the :class:`~.pennylane.Multiplier` operator.
"""

import pennylane as qml
qml.capture.enable()

# we indicate the name of the registers and their number of qubits. 
wires = qml.registers({"x": 4, "y":4, "output":6,"work_wires": 4})

######################################################################
# Now, we write a circuit to prepare the state :math:`|x \rangle|y \rangle|0 \rangle`, which will be needed for the out-place
# operation, where we initialize specific values to :math:`x` and :math:`y`. Note that in this example we use computational basis states, but
# you could introduce any quantum state as input.

def product_basis_state(x=0,y=0):
    qml.BasisState(x, wires=wires["x"])
    qml.BasisState(y, wires=wires["y"])

dev = qml.device("default.qubit")
@qml.set_shots(1)
@qml.qnode(dev)
def circuit(x,y):
    product_basis_state(x, y)
    return {name: qml.sample(wires=wires[name]) for name in ["x", "y", "output"]}

######################################################################
# Since the arithmetic operations are deterministic, a single shot is enough to sample 
# from the circuit and extract the expected state in the output register.
# Next, for understandability, we will use an auxiliary function that will 
# take one sample from the circuit and return the associated decimal number.

def state_to_decimal(binary_array):
    # Convert a binary array to a decimal number
    return sum(bit * (2 ** idx) for idx, bit in enumerate(reversed(binary_array)))

######################################################################
# In this example we are setting :math:`x=1` and :math:`y=4` and checking that the results are as expected.

output = circuit(x=1,y=4)
print("x register: ", output["x"][0]," (binary) ---> ", state_to_decimal(output["x"][0]), " (decimal)")
print("y register: ", output["y"][0]," (binary) ---> ", state_to_decimal(output["y"][0]), " (decimal)")
print("output register: ", output["output"][0]," (binary) ---> ", state_to_decimal(output["output"][0]), " (decimal)")

######################################################################
# Now we can implement an example for the :class:`~.pennylane.Adder` operator. We will add the integer :math:`5` to the ``wires["x"]`` register
# that stores the state :math:`|x \rangle=|1 \rangle`.

@qml.set_shots(1)
@qml.qnode(dev)
def circuit(x):

    product_basis_state(x)          # |x> 
    qml.Adder(5, wires["x"])        # |x+5> 

    return qml.sample(wires=wires["x"])

print(circuit(x=1), " (binary) ---> ", state_to_decimal(circuit(x=1)[0])," (decimal)")

######################################################################
# We obtained the result :math:`5+1=6`, as expected. At this point, it's worth taking a moment to look
# at the decomposition of the circuit into quantum gates and operators. 

fig, _ = qml.draw_mpl(circuit, decimals = 2, style = "pennylane", level='device')(x=1)
fig.show()

######################################################################
# From the :class:`~.pennylane.Adder` circuit we can see that the addition is performed 
# in the Fourier basis. This includes a quantum Fourier transformation (QFT) followed by rotations to perform the addition, and 
# concludes with an inverse QFT transformation. A more detailed explanation on the decomposition of arithmetic operators can be found in
# `the PennyLane Demo on quantum arithmetic with the QFT <https://pennylane.ai/qml/demos/tutorial_qft_arithmetics/>`_. 
#
# Now, let's see an example for the :class:`~.pennylane.OutAdder` operator to add the states 
# :math:`|x \rangle` and :math:`|y \rangle` to the output register.

@qml.set_shots(1)
@qml.qnode(dev)
def circuit(x,y):

    product_basis_state(x, y)                                  #    |x> |y> |0>
    qml.OutAdder(wires["x"], wires["y"], wires["output"])      #    |x> |y> |x+y>

    return qml.sample(wires=wires["output"])

print(circuit(x=2,y=3), " (binary) ---> ", state_to_decimal(circuit(x=2,y=3)[0]), " (decimal)")

######################################################################
# We obtained the result :math:`2+3=5`, as expected.
# 
# Multiplication  operators
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# There are two multiplication operators in PennyLane: the :class:`~.pennylane.Multiplier` and the :class:`~.pennylane.OutMultiplier`.
# The class :class:`~.pennylane.Multiplier` performs an in-place operation, multiplying the state of the wires :math:`|x \rangle` by an integer :math:`k`. It is defined as:
#
# .. math::
#
#   \text{Multiplier}(k) |x \rangle = | kx \rangle.
#
# The :class:`~.pennylane.OutMultiplier` performs an out-place operation, where the states of two 
# registers, :math:`|x \rangle` and :math:`|y \rangle`, 
# are multiplied together and the result is stored in a third register:
#
# .. math::
#
#   \text{OutMultiplier} |x \rangle |y \rangle |0 \rangle = |x \rangle |y \rangle |xy \rangle.
#  
# We proceed to implement these operators in PennyLane. First, let's see an example for the 
# :class:`~.pennylane.Multiplier` operator. We will multiply the state  :math:`|x \rangle=|2 \rangle` by 
# the integer :math:`k=3`:

@qml.set_shots(1)
@qml.qnode(dev)
def circuit(x):

    product_basis_state(x)                                           #    |x>                                    
    qml.Multiplier(3, wires["x"], work_wires=wires["work_wires"])    #    |3x> 

    return qml.sample(wires=wires["x"])

print(circuit(x=2), " (binary) ---> ", state_to_decimal(circuit(x=2)[0])," (decimal)")

######################################################################
# We got the expected result of :math:`3 \cdot 2 = 6`.
#
# Now, let's look at an example using the :class:`~.pennylane.OutMultiplier` operator to multiply the states :math:`|x \rangle` and
# :math:`|y \rangle`, storing the result in the output register.

@qml.set_shots(1)
@qml.qnode(dev)
def circuit(x,y):

    product_basis_state(x, y)                                     #    |x> |y> |0>
    qml.OutMultiplier(wires["x"], wires["y"], wires["output"])    #    |x> |y> |xy>

    return qml.sample(wires=wires["output"])

print(circuit(x=4,y=2), " (binary) ---> ", state_to_decimal(circuit(x=4,y=2)[0])," (decimal)")

######################################################################
# Nice! Modular subtraction and division can also be implemented as the inverses of addition and 
# multiplication, respectively. The inverse of a quantum circuit can be implemented with the 
# :func:`~.pennylane.adjoint` operator. Let's see an example of modular subtraction.

@qml.set_shots(1)
@qml.qnode(dev)
def circuit(x):

    product_basis_state(x)                     # |x> 
    qml.adjoint(qml.Adder(3, wires["x"]))      # |x-3>  

    return qml.sample(wires=wires["x"])

print(circuit(x=6), " (binary) ---> ", state_to_decimal(circuit(x=6)[0]), " (decimal)")

######################################################################
# As predicted, this gives us :math:`6-3=3`.
#
# Implementing a polynomial on a quantum computer
# --------------------------------------------
#
# Now that you are familiar with these operations, let's take it a step further and see how we can use them for something more complicated. 
# We will explore how to implement a polynomial function on a quantum computer using basic arithmetic.
# In particular, we will use :math:`f(x,y)= 4 + 3xy + 5 x+ 3 y`  as an example, where the variables :math:`x` and
# :math:`y` are integer values. Therefore, the operator we want to build is:
#
# .. math::
# 
#    U|x\rangle |y\rangle |0\rangle = |x\rangle |y\rangle |4 + 3xy + 5x + 3y\rangle.
#
# We will show how to implement this circuit in two different ways: first, by concatenating simple modular arithmetic operators,
# and finally, using the :class:`~.pennylane.OutPoly` operator.
#
# Concatenating arithmetic operations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Let's start by defining the arithmetic operations to apply the function :math:`f(x,y) = 4 + 3xy + 5x + 3y` into a quantum state.
#
# First, we need to define a function that will add the term :math:`3xy` to the output register. We will use
# the :class:`~.pennylane.Multiplier` and :class:`~.pennylane.OutMultiplier` operators for this. Also, we will employ the
# adjoint function to undo certain multiplications, since :math:`U` does not modify the input registers.

def adding_3xy():
    # |x> --->  |3x>
    qml.Multiplier(3, wires["x"], work_wires=wires["work_wires"])

    # |3x>|y>|0> ---> |3x>|y>|3xy>
    qml.OutMultiplier(wires["x"], wires["y"], wires["output"])

    # We return the x-register to its original value using the adjoint operation
    # |3x>|y>|3xy>  ---> |x>|y>|3xy>
    qml.adjoint(qml.Multiplier)(3, wires["x"], work_wires=wires["work_wires"])

######################################################################
# Then we need to add the term :math:`5x + 3y` to the output register, which can be done by using the
# :class:`~.pennylane.Multiplier` and :class:`~.pennylane.OutAdder` operators.

def adding_5x_3y():

    # |x>|y> --->  |5x>|3y>
    qml.Multiplier(5, wires["x"], work_wires=wires["work_wires"])
    qml.Multiplier(3, wires["y"], work_wires=wires["work_wires"])

    # |5x>|3y>|0> --->  |5x>|3y>|5x + 3y>
    qml.OutAdder(wires["x"], wires["y"], wires["output"])

    # We return the x and y registers to their original value using the adjoint operation
    # |5x>|3y>|5x + 3y> --->  |x>|y>|5x + 3y>
    qml.adjoint(qml.Multiplier)(5, wires["x"], work_wires=wires["work_wires"])
    qml.adjoint(qml.Multiplier)(3, wires["y"], work_wires=wires["work_wires"])

######################################################################
# Now we can combine all these circuits to implement the transformation by the polynomial  :math:`f(x,y)= 4 + 3xy + 5 x+ 3 y`.

@qml.set_shots(1)
@qml.qnode(dev)
def circuit(x,y):

    product_basis_state(x, y)      #    |x> |y> |0>
    qml.Adder(4, wires["output"])  #    |x> |y> |4>
    adding_3xy()                   #    |x> |y> |4 + 3xy>
    adding_5x_3y()                 #    |x> |y> |4 + 3xy + 5x + 3y>

    return qml.sample(wires=wires["output"])

print(circuit(x=1,y=4), " (binary) ---> ", state_to_decimal(circuit(x=1,y=4)[0]), " (decimal)")

######################################################################
# Cool, we get the correct result, :math:`f(1,4)=33`.
#
# At this point, it's interesting to consider what would happen if we had chosen a smaller number of wires for the output.
# For instance, if we had selected one wire fewer, we would have obtained the result :math:`33 \mod 2^5 = 1`.

wires = qml.registers({"x": 4, "y": 4, "output": 5,"work_wires": 4})

print(circuit(x=1,y=4), " (binary) ---> ", state_to_decimal(circuit(x=1,y=4)[0]), " (decimal)")

######################################################################
# With one fewer wire, we get :math:`1`, just like we predicted. Remember, we are working with modular arithmetic!
#
# Using the OutPoly function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# There is a more direct method to apply polynomial transformations in PennyLane: 
# using :class:`~.pennylane.OutPoly`. 
# This operator automatically takes care of all the arithmetic under the hood. 
# Let's check out how to apply a function like :math:`f(x, y)` using :class:`~.pennylane.OutPoly`.
#
# We will start by explicitly defining our function.

def f(x, y):
   return 4 + 3*x*y + 5*x + 3*y

######################################################################
# Now, we create a quantum circuit using :class:`~.pennylane.OutPoly`.

######################################################################

wires = qml.registers({"x": 4, "y":4, "output":6})
@qml.set_shots(1)
@qml.qnode(dev)
def circuit_with_Poly(x,y):

   product_basis_state(x, y)                         #    |x> |y> |0>
   qml.OutPoly(
       f, 
       input_registers= [wires["x"], wires["y"]],
       output_wires = wires["output"])               #    |x> |y> |4 + 3xy + 5x + 3y>
   
   return qml.sample(wires = wires["output"])

print(circuit_with_Poly(x=1,y=4), " (binary) ---> ", state_to_decimal(circuit_with_Poly(x=1,y=4)[0]), " (decimal)")

######################################################################
# You can decide, depending on the problem you are tackling, whether to go for the versatility 
# of defining your own arithmetic operations or the convenience of using the :class:`~.pennylane.OutPoly` function.

######################################################################
# Conclusion 
# ------------------------------------------
# Understanding and implementing quantum arithmetic is a key step toward unlocking the full potential
# of quantum computing. By leveraging quantum arithmetic operations in PennyLane, you can streamline 
# the coding of your quantum algorithms. So, 
# whether you choose to customize your arithmetic operations or take advantage of the built-in 
# convenience offered by PennyLane 
# operators, you are now equipped to tackle the exciting quantum challenges ahead.
#
# References
# ----------
# 
# .. [#shor_exp]
#
#     Robert L Singleton Jr
#     "Shor's Factoring Algorithm and Modular Exponentiation Operators.",
#     `arXiv:2306.09122 <https://arxiv.org/abs/2306.09122/>`__, 2023.
#
# .. [#sanders]
#
#     Yuval R. Sanders, Guang Hao Low, Artur Scherer, Dominic W. Berry
#     "Black-box quantum state preparation without arithmetic.",
#     `arXiv:1807.03206 <https://arxiv.org/abs/1807.03206/>`__, 2018.
#
