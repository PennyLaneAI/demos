"""
.. _pytorch_noise:

PyTorch and noisy devices
=========================

.. meta::
    :property="og:description": Extend PyTorch with real quantum computing power,
        by using it to optimize a noisy quantum hardware device.
    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets/bloch.gif

.. related::

   tutorial_noisy_circuit_optimization Optimizing noisy circuits with Cirq

.. warning::
    This demo is only compatible with PennyLane v0.40 or below and Braket v1.31.0. To use Rigetti hardware with newer versions of PennyLane please use the `PennyLane-Braket plugin <https://amazon-braket-pennylane-plugin-python.readthedocs.io/en/stable/index.html>`__ instead.
    

Let's revisit the original :doc:`qubit rotation <demos/tutorial_qubit_rotation>` tutorial, but instead of
using the default NumPy/autograd QNode interface, we'll use the :doc:`introduction/interfaces/torch`.
We'll also replace the ``default.qubit`` device with a noisy ``rigetti.qvm``
device, to see how the optimization responds to noisy qubits. At the end of the
demonstration, we will also show a way of how Rigetti's QPU can be used via
Amazon Braket.

To follow along with this tutorial on your own computer, you will require the
following dependencies:

* Rigetti's QVM and Quil Compiler services. One option for setting this up is the
  `Rigetti SDK <https://qcs.rigetti.com/sdk-downloads>`_, which contains the quantum virtual
  machine (QVM) and quilc quantum compiler. Once installed, the QVM and quilc can be
  started by running the commands ``quilc -S`` and ``qvm -S`` in separate terminal windows.
  Alternatively, for users with Docker, the QVM and Quil Compiler services can be run with commands:

  .. code-block:: bash

      docker run -d -p 5555:5555 rigetti/quilc -R -p 5555
      docker run -d -p 5000:5000 rigetti/qvm -S -p 5000

* `PennyLane-Rigetti plugin <https://docs.pennylane.ai/projects/rigetti/en/latest/>`_, in order
  to access the QVM as a PennyLane device. This can be installed via pip:

  .. code-block:: bash

      pip install pennylane-rigetti

* `PennyLane-Braket plugin <https://amazon-braket-pennylane-plugin-python.readthedocs.io/en/latest/>`_, in order
  to access the Rigetti QPU as a PennyLane device. This can be installed via
  pip:

  .. code-block:: bash

      pip install amazon-braket-pennylane-plugin

* `PyTorch <https://pytorch.org/get-started/locally/>`_, in order to access the PyTorch
  QNode interface. Follow the link for instructions on the best way to install PyTorch
  for your system.

Setting up the device
---------------------

Once the dependencies above are installed, let's begin importing the required packages
and setting up our quantum device.

To start with, we import PennyLane, and, as we are using the PyTorch interface,
PyTorch as well:
"""

import pennylane as qml
qml.capture.enable()
import torch
from torch.autograd import Variable

##############################################################################
# Note that we do not need to import the wrapped version of NumPy provided by PennyLane,
# as we are not using the default QNode NumPy interface. If NumPy is needed, it is fine to
# import vanilla NumPy for use with PyTorch and TensorFlow.
#
# Next, we will create our device:

dev = qml.device("rigetti.qvm", device="2q", noisy=True)

##############################################################################
# Here, we create a noisy two-qubit system, simulated via the QVM. If we wish, we could
# also build the model on a physical device, such as the ``Aspen-M-2`` QPU which
# can be accessed through Amazon Braket (more details on that will follow).


##############################################################################
# Constructing the QNode
# ----------------------
#
# Now that we have initialized the device, we can construct our quantum node. Like the
# other tutorials, we use the :mod:`~.pennylane.qnode` decorator to convert
# our quantum function (encoded by the circuit above) into a quantum node
# running on the QVM.


@qml.qnode(dev, interface="torch")
def circuit(phi, theta):
    qml.RX(theta, wires=0)
    qml.RZ(phi, wires=0)
    return qml.expval(qml.PauliZ(0))


##############################################################################
# To make the QNode 'PyTorch aware', we need to specify that the QNode interfaces
# with PyTorch. This is done by passing the ``interface='torch'`` keyword argument.
#
# As a result, this QNode will be set up to accept and return PyTorch tensors, and will
# also automatically calculate any analytic gradients when PyTorch performs backpropagation.


##############################################################################
# Optimization
# ------------
#
# We can now create our optimization cost function. To introduce some additional
# complexity into the system, rather than simply training the variational circuit
# to 'flip a qubit' from state :math:`\left|0\right\rangle` to state :math:`\left|1\right\rangle,` let's also
# modify the target state every 100 steps. For example, for the first 100 steps,
# the target state will be :math:`\left|1\right\rangle;` this will then change to :math:`\left|0\right\rangle`
# for steps 100 and 200, before changing back to state :math:`\left|1\right\rangle` for steps 200
# to 300, and so on.


def cost(phi, theta, step):
    target = -(-1) ** (step // 100)
    return torch.abs(circuit(phi, theta) - target) ** 2


##############################################################################
# Now that the cost function is defined, we can begin the PyTorch optimization.
# We create two variables, representing the two free parameters of the variational
# circuit, and initialize an Adam optimizer:

phi = Variable(torch.tensor(1.0), requires_grad=True)
theta = Variable(torch.tensor(0.05), requires_grad=True)
opt = torch.optim.Adam([phi, theta], lr=0.1)

##############################################################################
# As we are using the PyTorch interface, we must use PyTorch optimizers,
# *not* the built-in optimizers provided by PennyLane. The built-in optimizers
# only apply to the default NumPy/autograd interface.
#
# Optimizing the system for 400 steps:

for i in range(400):
    opt.zero_grad()
    loss = cost(phi, theta, i)
    loss.backward()
    opt.step()

##############################################################################
# We can now check the final values of the parameters, as well as the final
# circuit output and cost function:

print(phi)
print(theta)
print(circuit(phi, theta))
print(cost(phi, theta, 400))

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#
#  .. code-block:: none
#
#    tensor(-0.7055, requires_grad=True)
#    tensor(6.1330, requires_grad=True)
#    tensor(0.9551, dtype=torch.float64, grad_fn=<SqueezeBackward0>)
#    tensor(3.7162, dtype=torch.float64, grad_fn=<PowBackward0>)


##############################################################################
# As the cost function is step-dependent, this does not provide enough detail to
# determine if the optimization was successful; instead, let's plot the output
# state of the circuit over time on a Bloch sphere:
#
# .. figure:: ../_static/demonstration_assets/pytorch_noise/bloch.gif
#     :align: center
#     :target: javascript:void(0);
#
# Here, the red x is the target state of the variational circuit, and the arrow is
# the variational circuit output state. As the target state changes, the circuit
# learns to produce the new target state!


##############################################################################
# Hybrid GPU-QPU optimization
# ---------------------------
#
# As PyTorch natively supports GPU-accelerated classical processing, and Amazon
# Braket provides quantum hardware access in the form of QPUs, we can run the above code 
# as a hybrid GPU-QPU optimization with very little modification.
#
# Note that to run the following script, you will need access to Rigetti's QPU.
# To connect to a QPU, we can use Amazon Braket. For a dedicated demonstration
# on using Amazon Braket, see our tutorial on
# :doc:`Computing gradients in parallel with Amazon Braket <demos/braket-parallel-gradients>`.

import pennylane as qml
qml.capture.enable()
import torch
from torch.autograd import Variable

my_bucket = "amazon-braket-Your-Bucket-Name"  # the name of the bucket
my_prefix = "Your-Folder-Name"  # the name of the folder in the bucket
s3_folder = (my_bucket, my_prefix)

device_arn = "arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3"

qpu = qml.device(
    "braket.aws.qubit",
    device_arn=device_arn,
    wires=32,
    s3_destination_folder=s3_folder,
)

# Note: swap dev to qpu here to use the QPU
# Warning: check the pricing of Aspen-M-3 on Braket to make
# sure you are aware of the costs associated with running the
# optimization below.
@qml.qnode(dev, interface="torch")
def circuit(phi, theta):
    qml.RX(theta, wires=0)
    qml.RZ(phi, wires=0)
    return qml.expval(qml.PauliZ(0))


def cost(phi, theta, step):
    target = -(-1) ** (step // 100)
    return torch.abs(circuit(phi, theta) - target) ** 2


phi = Variable(torch.tensor(1.0, device="cuda"), requires_grad=True)
theta = Variable(torch.tensor(0.05, device="cuda"), requires_grad=True)
opt = torch.optim.Adam([phi, theta], lr=0.1)

for i in range(400):
    opt.zero_grad()
    loss = cost(phi, theta, i)
    loss.backward()
    opt.step()

##############################################################################
# When using a classical interface that supports GPUs, the QNode will automatically
# copy any tensor arguments to the CPU, before applying them on the specified quantum
# device. Once done, it will return a tensor containing the QNode result, and
# automatically copy it back to the GPU for any further classical processing.
#
# .. note:: For more details on the PyTorch interface, see :doc:`introduction/interfaces/torch`.
#
