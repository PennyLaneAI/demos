r"""How to optimize a QML model using JAX and Optax
====================================================================

Once you have set up a quantum machine learning model, data to train with and
cost function to minimize as an objective, the next step is to **perform the optimization**. That is,
setting up a classical optimization loop to find a minimal value of your cost function.

In this example, we’ll show you how to use `JAX <https://jax.readthedocs.io>`__, an
autodifferentiable machine learning framework, and `Optax <https://optax.readthedocs.io/>`__, a
suite of JAX-compatible gradient-based optimizers, to optimize a PennyLane quantum machine learning
model.

.. figure:: ../_static/demonstration_assets/How_to_optimize_QML_model_using_JAX_and_Optax/socialsthumbnail_large_How_to_optimize_QML_model_using_JAX_and_Optax_2024-01-16.png
    :align: center
    :width: 50%

"""

######################################################################
# Set up your model, data, and cost
# ---------------------------------
#

######################################################################
# Here, we will create a simple QML model for our optimization. In particular:
#
# -  We will embed our data through a series of rotation gates.
# -  We will then have an ansatz of trainable rotation gates with parameters ``weights``; it is these
#    values we will train to minimize our cost function.
# -  We will train the QML model on ``data``, a ``(5, 4)`` array, and optimize the model to match
#    target predictions given by ``target``.
#

import pennylane as qml
qml.capture.enable()
import jax
from jax import numpy as jnp
import optax

n_wires = 5
data = jnp.sin(jnp.mgrid[-2:2:0.2].reshape(n_wires, -1)) ** 3
targets = jnp.array([-0.2, 0.4, 0.35, 0.2])

dev = qml.device("default.qubit", wires=n_wires)

@qml.qnode(dev)
def circuit(data, weights):
    """Quantum circuit ansatz"""

    # data embedding
    for i in range(n_wires):
        # data[i] will be of shape (4,); we are
        # taking advantage of operation vectorization here
        qml.RY(data[i], wires=i)

    # trainable ansatz
    for i in range(n_wires):
        qml.RX(weights[i, 0], wires=i)
        qml.RY(weights[i, 1], wires=i)
        qml.RX(weights[i, 2], wires=i)
        qml.CNOT(wires=[i, (i + 1) % n_wires])

    # we use a sum of local Z's as an observable since a
    # local Z would only be affected by params on that qubit.
    return qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(n_wires)]))

def my_model(data, weights, bias):
    return circuit(data, weights) + bias

######################################################################
# We will define a simple cost function that computes the overlap between model output and target
# data, and `just-in-time (JIT) compile <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html>`__ it:
#

@jax.jit
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

######################################################################
# Initialize your parameters
# --------------------------
#

######################################################################
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

print(loss_fn(params, data, targets))

print(jax.grad(loss_fn)(params, data, targets))

######################################################################
# Create the optimizer
# --------------------
#

######################################################################
# We can now use Optax to create an optimizer, and train our circuit.
# Here, we choose the Adam optimizer, however
# `other available optimizers <https://optax.readthedocs.io/en/latest/api.html>`__
# may be used here.
#

opt = optax.adam(learning_rate=0.3)
opt_state = opt.init(params)

######################################################################
# We first define our ``update_step`` function, which needs to do a couple of things:
#
# -  Compute the loss function (so we can track training) and the gradients (so we can apply an
#    optimization step). We can do this in one execution via the ``jax.value_and_grad`` function.
#
# -  Apply the update step of our optimizer via ``opt.update``
#
# -  Update the parameters via ``optax.apply_updates``
#

def update_step(opt, params, opt_state, data, targets):
    loss_val, grads = jax.value_and_grad(loss_fn)(params, data, targets)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_val

loss_history = []

for i in range(100):
    params, opt_state, loss_val = update_step(opt, params, opt_state, data, targets)

    if i % 5 == 0:
        print(f"Step: {i} Loss: {loss_val}")

    loss_history.append(loss_val)

######################################################################
# Jitting the optimization loop
# -----------------------------
#

######################################################################
# In the above example, we JIT compiled our cost function ``loss_fn``. However, we can
# also JIT compile the entire optimization loop; this means that the for-loop around optimization is
# not happening in Python, but is compiled and executed natively. This avoids (potentially costly) data
# transfer between Python and our JIT compiled cost function with each update step.
#

# Define the optimizer we want to work with
opt = optax.adam(learning_rate=0.3)

@jax.jit
def update_step_jit(i, args):
    params, opt_state, data, targets, print_training = args

    loss_val, grads = jax.value_and_grad(loss_fn)(params, data, targets)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    def print_fn():
        jax.debug.print("Step: {i}  Loss: {loss_val}", i=i, loss_val=loss_val)

    # if print_training=True, print the loss every 5 steps
    jax.lax.cond((jnp.mod(i, 5) == 0) & print_training, print_fn, lambda: None)

    return (params, opt_state, data, targets, print_training)

@jax.jit
def optimization_jit(params, data, targets, print_training=False):

    opt_state = opt.init(params)
    args = (params, opt_state, data, targets, print_training)
    (params, opt_state, _, _, _) = jax.lax.fori_loop(0, 100, update_step_jit, args)

    return params

######################################################################
# Note that we use ``jax.lax.fori_loop`` and ``jax.lax.cond``, rather than a standard Python for loop
# and if statement, to allow the control flow to be JIT compatible. We also
# use ``jax.debug.print`` to allow printing to take place at function run-time,
# rather than compile-time.
#

params = {"weights": weights, "bias": bias}
optimization_jit(params, data, targets, print_training=True)

######################################################################
# Appendix: Timing the two approaches
# -----------------------------------
#
# We can time the two approaches (JIT compiling just the cost function, vs JIT compiling the entire
# optimization loop) to explore the differences in performance:
#

from timeit import repeat

def optimization(params, data, targets):
    opt = optax.adam(learning_rate=0.3)
    opt_state = opt.init(params)

    for i in range(100):
        params, opt_state, loss_val = update_step(opt, params, opt_state, data, targets)

    return params

reps = 5
num = 2

times = repeat("optimization(params, data, targets)", globals=globals(), number=num, repeat=reps)
result = min(times) / num

print(f"Jitting just the cost (best of {reps}): {result} sec per loop")

times = repeat("optimization_jit(params, data, targets)", globals=globals(), number=num, repeat=reps)
result = min(times) / num

print(f"Jitting the entire optimization (best of {reps}): {result} sec per loop")


######################################################################
# In this example, JIT compiling the entire optimization loop
# is significantly more performant.
#
