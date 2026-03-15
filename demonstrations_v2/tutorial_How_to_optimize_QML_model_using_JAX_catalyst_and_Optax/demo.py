r"""
How to optimize a QML model using Catalyst and quantum just-in-time (QJIT) compilation
======================================================================================

Once you have set up your quantum machine learning model (which typically includes deciding on your
circuit architecture/ansatz, determining how you embed or integrate your data, and creating your
cost function to minimize a quantity of interest), the next step is **optimization**. That is,
setting up a classical optimization loop to find a minimal value of your cost function.

In this example, we’ll show you how to use `JAX <https://jax.readthedocs.io>`__, an
autodifferentiable machine learning framework, and `Optax <https://optax.readthedocs.io/>`__, a
suite of JAX-compatible gradient-based optimizers, to optimize a PennyLane quantum machine learning
model which has been quantum just-in-time compiled using the :func:`~.pennylane.qjit` decorator and
`Catalyst <https://github.com/pennylaneai/catalyst>`__.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_large_how-to-optimize-qjit-optax_2024-04-23.png
    :align: center
    :width: 60%
    :target: javascript:void(0)

Set up your model, data, and cost
---------------------------------

Here, we will create a simple QML model for our optimization. In particular:

-  We will embed our data through a series of rotation gates.
-  We will then have an ansatz of trainable rotation gates with parameters ``weights``; it is these
   values we will train to minimize our cost function.
-  We will train the QML model on ``data``, a ``(5, 4)`` array, and optimize the model to match
   target predictions given by ``target``.
"""

import pennylane as qml
qml.capture.enable()
from jax import numpy as jnp
import optax
import catalyst

n_wires = 5
data = jnp.sin(jnp.mgrid[-2:2:0.2].reshape(n_wires, -1)) ** 3
targets = jnp.array([-0.2, 0.4, 0.35, 0.2])

dev = qml.device("lightning.qubit", wires=n_wires)

@qml.qnode(dev)
def circuit(data, weights):
    """Quantum circuit ansatz"""

    @qml.for_loop(0, n_wires, 1)
    def data_embedding(i):
        qml.RY(data[i], wires=i)

    data_embedding()

    @qml.for_loop(0, n_wires, 1)
    def ansatz(i):
        qml.RX(weights[i, 0], wires=i)
        qml.RY(weights[i, 1], wires=i)
        qml.RX(weights[i, 2], wires=i)
        qml.CNOT(wires=[i, (i + 1) % n_wires])

    ansatz()

    # we use a sum of local Z's as an observable since a
    # local Z would only be affected by params on that qubit.
    return qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(n_wires)]))

######################################################################
# The :func:`catalyst.vmap` function allows us to specify that the first argument to circuit (``data``)
# contains a batch dimension. In this example, the batch dimension is the second axis (axis 1).
#

circuit = qml.qjit(catalyst.vmap(circuit, in_axes=(1, None)))

######################################################################
# We will define a simple cost function that computes the overlap between model output and target
# data:
#

def my_model(data, weights, bias):
    return circuit(data, weights) + bias

@qml.qjit
def loss_fn(params, data, targets):
    predictions = my_model(data, params["weights"], params["bias"])
    loss = jnp.sum((targets - predictions) ** 2 / len(data))
    return loss

######################################################################
# Note that the model above is just an example for demonstration – there are important considerations
# that must be taken into account when performing QML research, including methods for data embedding,
# circuit architecture, and cost function, in order to build models that may have use. This is still
# an active area of research; see our `demonstrations <https://pennylane.ai/qml/demonstrations>`__ for
# details.
#
# Initialize your parameters
# --------------------------
#
# Now, we can generate our trainable parameters ``weights`` and ``bias`` that will be used to train
# our QML model.
#

weights = jnp.ones([n_wires, 3])
bias = jnp.array(0.)
params = {"weights": weights, "bias": bias}

######################################################################
# Plugging the trainable parameters, data, and target labels into our cost function, we can see the
# current loss as well as the parameter gradients:
#

loss_fn(params, data, targets)

print(qml.qjit(catalyst.grad(loss_fn, method="fd"))(params, data, targets))

######################################################################
# Create the optimizer
# --------------------
#
# We can now use Optax to create an Adam optimizer, and train our circuit.
#
# We first define our ``update_step`` function, which needs to do a couple of things:
#
# -  Compute the gradients of the loss function. We can
#    do this via the :func:`catalyst.grad` function.
#
# -  Apply the update step of our optimizer via ``opt.update``
#
# -  Update the parameters via ``optax.apply_updates``
#

opt = optax.adam(learning_rate=0.3)

@qml.qjit
def update_step(i, args):
    params, opt_state, data, targets = args

    grads = catalyst.grad(loss_fn, method="fd")(params, data, targets)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return (params, opt_state, data, targets)

loss_history = []

opt_state = opt.init(params)

for i in range(100):
    params, opt_state, _, _ = update_step(i, (params, opt_state, data, targets))
    loss_val = loss_fn(params, data, targets)

    if i % 5 == 0:
        print(f"Step: {i} Loss: {loss_val}")

    loss_history.append(loss_val)

######################################################################
# JIT-compiling the optimization
# ------------------------------
#
# In the above example, we just-in-time (JIT) compiled our cost function ``loss_fn``. However, we can
# also JIT compile the entire optimization loop; this means that the for-loop around optimization is
# not happening in Python, but is compiled and executed natively. This avoids (potentially costly)
# data transfer between Python and our JIT compiled cost function with each update step.
#

params = {"weights": weights, "bias": bias}

@qml.qjit
def optimization(params, data, targets):
    opt_state = opt.init(params)
    args = (params, opt_state, data, targets)
    (params, opt_state, _, _) = qml.for_loop(0, 100, 1)(update_step)(args)
    return params

######################################################################
# Note that we use :func:`~pennylane.for_loop` rather than a standard Python for loop, to allow the control
# flow to be JIT compatible.
#

final_params = optimization(params, data, targets)

print(final_params)

######################################################################
# Timing the optimization
# -----------------------
#
# We can time the two approaches (JIT compiling just the cost function, vs JIT compiling the entire
# optimization loop) to explore the differences in performance:
#

from timeit import repeat

opt = optax.adam(learning_rate=0.3)

def optimization_noqjit(params):
    opt_state = opt.init(params)

    for i in range(100):
        params, opt_state, _, _ = update_step(i, (params, opt_state, data, targets))

    return params

reps = 5
num = 2

times = repeat("optimization_noqjit(params)", globals=globals(), number=num, repeat=reps)
result = min(times) / num

print(f"Quantum jitting just the cost (best of {reps}): {result} sec per loop")

times = repeat("optimization(params, data, targets)", globals=globals(), number=num, repeat=reps)
result = min(times) / num

print(f"Quantum jitting the entire optimization (best of {reps}): {result} sec per loop")
