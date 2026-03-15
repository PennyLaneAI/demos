r"""
How to use Catalyst with Lightning-GPU
=======================================

We have been sharing details about how to use
`Catalyst <https://docs.pennylane.ai/projects/catalyst>`__,
our quantum just-in-time (QJIT) compiler framework,
to optimize and compile hybrid quantum-classical programs
with different `PennyLane devices <https://pennylane.ai/plugins>`__.
Here, we will demonstrate how to leverage the power of
`Lightning-GPU <https://docs.pennylane.ai/projects/lightning/en/stable/lightning_gpu/device.html>`__
to accelerate your quantum simulations.


Lightning-GPU is our high-performance CUDA-backed state-vector simulator for PennyLane.
This device enables fast quantum circuit simulations on NVIDIA GPUs
and has been recently integrated with Catalyst.


.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_Lightning_GPU_Catalyst.png
     :align: center
     :width: 80%
     :target: javascript:void(0)
     :alt: How to use Catalyst with Lightning-GPU


Let's get started with setting up the environment and running a simple example
to highlight Catalyst with Lightning-GPU features.

"""
##############################################################################
# Setting up your environment
# ----------------------------
#
# Using Lightning-GPU is particularly beneficial when dealing with large quantum circuits or
# when running simulations that require significant computational resources.
# By offloading the heavy linear algebra operations to GPUs,
# we can achieve substantial performance improvements.
#
# Lightning-GPU can be used on systems equipped with NVIDIA GPUs of generation SM 7.0 (Volta) and greater,
# specifically, it is optimized for Linux X86-64 or ARM64 machines with CUDA-12 installed.
#
# To use Catalyst with Lightning-GPU, you need to install PennyLane,
# the :func:`~pennylane.qjit` compiler framework, and the Lightning-GPU plugin (``lightning.gpu``).
# These packages can be installed from the Python Package Index (PyPI) using the following command:
#
# .. code-block:: none
#
#     pip install pennylane pennylane-catalyst pennylane-lightning-gpu
#
# If you are using a different hardware configuration or operating system,
# you may need to install these packages from source.
# Detailed instructions for source installation can be found
# in the respective `documentation <https://pennylane.ai/install>`__ of each package.

##############################################################################
# Compile and run a circuit
# --------------------------
#
# Here, we use a simple PennyLane circuit to demonstrate
# how to compile and run a quantum program on Lightning-GPU with Catalyst.
#
# First, we create a ``lightning.gpu`` device with 28 qubits using the :func:`~pennylane.device` function.
# We then define a quantum circuit that applies layers of :func:`~pennylane.RZ` and :func:`~pennylane.RY`
# rotations to each wire and returns the expectation value as the measurement result.
#
# To compile the circuit with Catalyst, we use the :func:`~pennylane.qjit` decorator.
# The ``autograph=True`` compilation option compiles the circuit with for-loops.
# This allows Catalyst to efficiently
# :doc:`capture control-flow operations <catalyst:dev/autograph>`,
# reduce the compilation time, and generate a more efficient program.
#
# .. code-block:: python
#
#     import pennylane as qml
qml.capture.enable()
#     import jax.numpy as jnp
#
#     import jax
#
#     # Set number of wires
#     num_wires = 28
#
#     # Set a random seed
#     key = jax.random.PRNGKey(0)
#
#     dev = qml.device("lightning.gpu", wires=num_wires)
#
#     @qml.qjit(autograph=True)
#     @qml.qnode(dev)
#     def circuit(params):
#
#         # Apply layers of RZ and RY rotations
#         for i in range(num_wires):
#             qml.RZ(params[3*i], wires=[i])
#             qml.RY(params[3*i+1], wires=[i])
#             qml.RZ(params[3*i+2], wires=[i])
#
#         return qml.expval(qml.PauliZ(0) + qml.PauliZ(num_wires-1))
#
#     # Initialize the weights
#     weights = jax.random.uniform(key, shape=(3 * num_wires,), dtype=jnp.float32)
#
#     >>> circuit(weights)

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  .. code-block:: none
#
#    1.2454125296886156
#
# Lightning-GPU has feature parity with `the rest of Lightning state-vector simulators <https://docs.pennylane.ai/projects/lightning/en/stable>`__,
# providing native support for many PennyLane's operations and measurement processes with
# a fast adjoint-Jacobian differentiation implementation.
#
# In the next section, we demonstrate how to compute the gradient of your quantum programs
# using the native adjoint-Jacobian method on this device.

##############################################################################
# Compute gradients
# ------------------
#
# We use :func:`~pennylane.grad` to compute the gradient of the ``circuit`` with respect to
# a set of trainable arguments.
# This method is particularly useful for variational quantum algorithms
# where the gradient of the cost function with respect to the circuit parameters is required.
# You can check out `this demo <https://pennylane.ai/qml/demos/tutorial_how_to_quantum_just_in_time_compile_vqe_catalyst>`__
# for more information on compiling and optimizing Variational Quantum Eigensolver (VQE) with Catalyst.
#
# .. code-block:: python
#
#     @qml.qjit(autograph=True)
#     def workflow(params):
#         g = qml.grad(circuit)
#         return g(params)
#
#     >>> workflow(weights)

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  .. code-block:: none
#
#    [ -1.1102230e-16 ... -7.2645772e-01  0.0000000e+00]
#
#
# Note that in the above example, we didn't use ``method="adjoint"``.
# The adjoint-Jacobian is the default gradient method when you compile
# and run your program on Lightning-GPU.
#
# Optimize parameters
# --------------------
#
# Now that we have seen how to leverage the device differentiation capabilities,
# we can move on to further optimizing the circuit parameters using Catalyst.
# We follow the same steps as in
# `the VQE demo <https://pennylane.ai/qml/demos/tutorial_how_to_quantum_just_in_time_compile_vqe_catalyst>`__,
# to QJIT-compile the entire optimization workflow.
#
# .. code-block:: python
#
#     import catalyst
#     import optax
#
#     opt = optax.sgd(learning_rate=0.4)
#
#     def update_step(i, params, opt_state):
#         energy, grads = catalyst.value_and_grad(circuit)(params)
#         updates, opt_state = opt.update(grads, opt_state)
#         params = optax.apply_updates(params, updates)
#         catalyst.debug.print("Step = {i}", i=i)
#         return (params, opt_state)
#
#     @qml.qjit
#     def optimization(params):
#         opt_state = opt.init(params)
#         (params, opt_state) = qml.for_loop(0, 10, 1)(update_step)(params, opt_state)
#         return params
#
#     >>> optimization(weights)

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  .. code-block:: none
#
#    Step = 0
#    Step = 1
#    Step = 2
#    Step = 3
#    Step = 4
#    Step = 5
#    Step = 6
#    Step = 7
#    Step = 8
#    Step = 9
#    [0.947667 ... 3.093198 0.8401809 ]
#
#
# We used `Optax <https://github.com/google-deepmind/optax>`__,
# a library designed for optimization using JAX,
# as well as the QJIT-compatible :func:`~pennylane.for_loop` function
# to compile the entire optimization loop,
# leading to further performance improvements.

##############################################################################
# Précis
# -------
#
# In this tutorial, we have shown how to use Catalyst with Lightning-GPU
# to accelerate quantum simulations on NVIDIA GPUs.
# We have compiled a simple quantum circuit with Catalyst,
# computed the gradient of the circuit using the adjoint-Jacobian method,
# and optimized the circuit parameters using Catalyst and Optax.
#
# To learn more about all the integrated devices,
# check out the :doc:`Catalyst Devices <catalyst:dev/devices>` guide.
#