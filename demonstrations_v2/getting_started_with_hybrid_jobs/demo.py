r"""

Getting started with Amazon Braket Hybrid Jobs
=======================================


This tutorial provides an introduction to running hybrid quantum-classical algorithms using
PennyLane on `Amazon Braket <https://aws.amazon.com/braket/>`__. With Amazon Braket, you gain access to both real quantum devices and
scalable classical compute, enabling you to push the boundaries of your algorithm.

In this tutorial, we'll walk through how to create your first hybrid quantum-classical algorithms on AWS.
With a single-line-of-code, we'll see how to scale from PennyLane simulators on your laptop to running full-scale experiments on AWS that leverage both powerful classical compute and quantum devices.
You'll gain understanding of the hybrid jobs queue, including QPU priority queuing, and learn how to scale classical resources for resource-intensive tasks.
We hope these tools will empower you to start experimenting today with hybrid quantum algorithms!

.. figure:: /_static/demonstration_assets/getting_started_with_hybrid_jobs/socialthumbnail_large_getting_started_with_hybrid_jobs.png
     :align: center
     :width: 65%
     :target: javascript:void(0);


Amazon Braket Hybrid Jobs
-------------------------

Amazon Braket Hybrid Jobs offers a way for you to run hybrid quantum-classical algorithms that
require both classical resources and quantum processing units (QPUs). Hybrid Jobs is designed to
spin up the requested classical compute, run your algorithm, and release the instances after
completion so you only pay for what you use. This workflow is ideal for long-running iterative
algorithms involving both classical and quantum resources. Simply package up your code into a single
function, create a hybrid job with a single line of code, and Braket will schedule it to run as soon
as possible without interruption.

Hybrid jobs have a separate queue from quantum tasks, so once your algorithm starts running, it will
not be interrupted by variations in the quantum task queue. This helps your long-running algorithms
run efficiently and predictably. Any quantum tasks created from a running hybrid job will be run
before any other quantum tasks in the queue. This is particularly beneficial for iterative hybrid
algorithms where subsequent tasks depend on the outcomes of prior quantum tasks. Examples of such
algorithms include the Quantum Approximate Optimization Algorithm (QAOA), Variational Quantum
Eigensolver (VQE), or Quantum Machine Learning (QML). You can also monitor your algorithm's progress in near-real
time, enabling you to keep track of costs, budget, or custom metrics such as training loss or
expectation values.

Importantly, on specific QPUs, running your algorithm in Hybrid Jobs benefits from `parametric compilation <https://docs.aws.amazon.com/braket/latest/developerguide/braket-jobs-parametric-compilation.html>`__.
This reduces the overhead associated with the computationally expensive compilation step by compiling a circuit only once and not for every iteration in your hybrid algorithm.
This dramatically reduces the total runtime for many variational algorithms.
For long-running hybrid jobs, Braket automatically uses the updated calibration data from the hardware provider when compiling your circuit to ensure the highest quality results.

Getting started with PennyLane
------------------------------

Let’s setup an algorithm that makes use of both classical and quantum resources. We adapt the :doc:`PennyLane qubit rotation tutorial<demos/tutorial_qubit_rotation>`.

.. warning::

    The following demo is only compatible with Python version 3.10. Consequently, the latest supported PennyLane
    version is ``pennylane==0.42``.

"""

######################################################################
# First, we define a quantum simulator to run the algorithm on. In this example, we will use the Braket
# local simulator before moving onto a QPU.
#

import pennylane as qml
qml.capture.enable()
from pennylane import numpy as np

device = qml.device("braket.local.qubit", wires=1)

######################################################################
# Now we define a circuit with two rotation gates and measure the expectation value in the
# :math:`Z`-basis.
#


@qml.qnode(device)
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    return qml.expval(qml.PauliZ(0))


######################################################################
# Finally, we create a classical-quantum loop that uses gradient descent to minimize the expectation value.
#
# We add the ``log_metric`` function from Braket to record the training progress (see `metrics
# documentation <https://amazon-braket-sdk-python.readthedocs.io/en/stable/_apidoc/braket.jobs.metrics.html>`__).
# When running on AWS, ``log_metric`` records the metrics in `Amazon CloudWatch <https://aws.amazon.com/cloudwatch/>`__, which is accessible
# through the Braket console page or the Braket SDK. When running locally on your laptop,
# ``log_metric`` prints the iteration numbers.
#

from braket.jobs.metrics import log_metric


def qubit_rotation(num_steps=10, stepsize=0.5):
    opt = qml.GradientDescentOptimizer(stepsize=stepsize)
    params = np.array([0.5, 0.75])

    for i in range(num_steps):
        # update the circuit parameters
        params = opt.step(circuit, params)
        expval = circuit(params)

        log_metric(metric_name="expval", iteration_number=i, value=expval)

    return params


######################################################################
# To run the entire algorithm, we call the qubit rotation function to see that it runs correctly.
#

qubit_rotation(5, stepsize=0.5)

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  .. code-block:: none
#
#       Metrics - timestamp=1695043649.822114; expval=0.38894534132396147; iteration_number=0;
#       Metrics - timestamp=1695043649.8757634; expval=0.12290715413453956; iteration_number=1;
#       Metrics - timestamp=1695043649.9254549; expval=-0.09181374013482171; iteration_number=2;
#       Metrics - timestamp=1695043649.973501; expval=-0.2936094099948542; iteration_number=3;
#       Metrics - timestamp=1695043650.020348; expval=-0.5344079938678078; iteration_number=4;
#       [0.6767967215302757, 2.3260934173312653]

######################################################################
# Great! We see the expectation value change with each iteration number and the final parameters were
# returned as a list. Now, instead of running on our laptop, let’s submit this same function to be run
# on AWS cloud.
#

######################################################################
# Running as a hybrid job
# -----------------------
#
# To run our algorithm for a long time, we can run it asynchronously with Amazon Braket Hybrid Jobs,
# which fully manages the classical infrastructure so you can focus on the algorithm. For example, you
# can train a larger circuit, or increase the number of iterations. Note that each hybrid job has
# at least a one minute startup time since it creates a containerized environment on Amazon EC2. So
# for very short workloads, such as a single circuit or a batch of circuits, it may suffice for you to
# use quantum tasks.
#
# We now show how you can go from running your local Python function to running it as a hybrid job.
# Note that only Python 3.10 is supported by default. For custom environments, you can opt to hybrid
# job scripts, or a custom container from Amazon Elastic Container Registry (ECR) (see `containers
# documentation <https://docs.aws.amazon.com/braket/latest/developerguide/braket-jobs-byoc.html>`__).
#
# The first step to creating a hybrid job is to annotate which function you want to run with the
# ``@hybrid_job`` decorator. Then you create the job by invoking the function as you would for normal
# Python functions. However, the decorated function returns the hybrid job handle rather than the
# result of the function. To retrieve the results after it has been completed, use ``job.result()``.
#
# The required device argument in the ``@hybrid_job`` decorator specifies the QPU that the hybrid job
# will have priority access to.
# The device string you give is accessible in the hybrid job instance as the environment variable ``AMZN_BRAKET_DEVICE_ARN``.
#
# When using on-demand simulators or `embedded simulators <https://docs.aws.amazon.com/braket/latest/developerguide/pennylane-embedded-simulators.html>`__,
# you may provide the device argument as string of the form: ``"local:<provider>/<simulator_name>"`` or simply ``None``.
# For example, you may set ``"local:pennylane/lightning.qubit"`` for the `PennyLane lightning simulator <https://pennylane.ai/performance>`__.
#
# In the following code, we annotate the ``qubit_rotation`` function from above. It is redefined inside the
# `hybrid_job` context as the job only has access to local variables and functions.
#

from braket.jobs import hybrid_job


@hybrid_job(device="local:pennylane/lightning.qubit")
def qubit_rotation_hybrid_job(num_steps=1, stepsize=0.5):
    device = qml.device("lightning.qubit", wires=1)

    @qml.qnode(device)
    def circuit(params):
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=0)
        return qml.expval(qml.PauliZ(0))

    opt = qml.GradientDescentOptimizer(stepsize=stepsize)
    params = np.array([0.5, 0.75])

    for i in range(num_steps):
        # update the circuit parameters
        params = opt.step(circuit, params)
        expval = circuit(params)

        log_metric(metric_name="expval", iteration_number=i, value=expval)

    return params


######################################################################
# Now we create a hybrid job by calling the function as usual. This returns an ``AwsQuantumJob`` object
# that contains the device ARN, region, and job name.
#

job = qubit_rotation_hybrid_job(num_steps=10, stepsize=0.5)
print(job)

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  .. code-block:: none
#
#       AwsQuantumJob('arn':'arn:aws:braket:<aws-region>:<account_id>:job/qubit-rotation-hybrid-job-1695044583')

######################################################################
# The hybrid job automatically captures the function arguments as hyperparameters.
# In this case, we set ``num_steps = 10`` and ``stepsize = 0.5`` as the hyperparameters.
#
# We can check the status with:
#

job.state()

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  .. code-block:: none
#
#       'QUEUED'

######################################################################
# Once the hybrid job starts, it will change the status to ``RUNNING``. We can also check the hybrid
# job status in the Braket console.
#
# After the hybrid job completes, we can get the results with ``job.result()``. For this example, it
# should take approximately 2 minutes.
#

job.result()

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  .. code-block:: none
#
#       {'result': [0.036420360224358496, 3.1008192940506736]}

######################################################################
# Any objects in the return statement will be returned in the hybrid job results. Note that the objects
# returned by the function must be a tuple with each element being serializable.
# serializable.
#
# Additionally, we can plot the metrics recorded during the algorithm. Below we show the expectation
# value decreases with each iteration as expected.
#
# .. note::
#
#     ``job.metrics()`` will be an empty dictionary while the job is running, and likely also for
#     a few minutes afterwards. To make sure that the below plot generates correctly, it is recommended
#     to block execution until the metrics are correctly reported. This can be done by:
#
#     * using a ``while`` loop to manually block execution until the metrics are available.
#     * using a Python debugger and setting a breakpoint before the line that uses ``job.metrics()``
#       and blocking as needed.
#
#     We will use a ``while`` loop as shown below.
#

while len(job.metrics()) == 0:
    pass


import matplotlib.pyplot as plt
import pandas as pd
from braket.jobs.metrics_data.definitions import MetricType

df = pd.DataFrame(job.metrics(metric_type=MetricType.ITERATION_NUMBER))
df.sort_values(by=["iteration_number"], inplace=True)

plt.plot(df["iteration_number"], df["expval"], "-o", color="orange")
plt.xlabel("iteration number")
plt.ylabel("expectation value")
plt.title("Simulator results")

plt.show()

######################################################################
#
# .. figure:: /_static/demonstration_assets/getting_started_with_hybrid_jobs/simulator.png
#     :align: center
#     :width: 75%
#     :alt: Expectation value per iteration number on QPU.
#     :target: javascript:void(0);


######################################################################
# Running on a QPU with priority
# ----------------
#
# The next step is to see how well the qubit rotation works on a real QPU. We
# create a hybrid job with the Rigetti device as the priority QPU. We also increase the number of
# steps to 10.
#
# Using hybrid jobs for iterative algorithms is very beneficial because you retain priority access to the
# target QPU. So once your quantum tasks are created in the hybrid job, they run ahead of other tasks
# waiting in the *quantum task queue*. This means your
# algorithm will not be interrupted by other quantum tasks, so it will run more efficiently and
# predictably. Quantum tasks submitted as part of a hybrid job have priority, and are aggregated in the *priority task queue*.
#
# Hybrid jobs have their own *hybrid jobs queue* so that only a single
# hybrid job can run on a QPU at a time.
# Each QPU has its own hybrid jobs queue. Note that this is a different queue from the quantum tasks. For a single quantum circuit, or a batch of circuits, it’s
# recommended to create quantum tasks instead of hybrid jobs.
# For more information on quantum tasks and hybrid jobs queue see the `Amazon Braket documentation <https://docs.aws.amazon.com/braket/latest/developerguide/braket-task-when.html>`__.
#
# To get QPU priority, you must ensure that the device ARN used within the function matches that
# specified in the decorator. For convenience, you can use the helper function ``get_device_arn()`` to
# automatically capture the device ARN declared in ``@hybrid_job``.
#
# .. note::
#   Only hybrid jobs running on AWS receive priority. Hybrid jobs running `locally <https://docs.aws.amazon.com/braket/latest/developerguide/braket-jobs-local-mode.html>`__,
#   or with a mismatched device ARN do not get priority task queueing.
#
# In the previous example, we declared the local simulator outside the decorated function scope.
# However, for AWS devices such as QPUs or on-demand simulators, the device must be declared within the function scope.
#
# .. note::
#   AWS devices must be declared within the body of the decorated function.
#
# .. warning::
#   The Rigetti device used in this demo, AspenM3, has been retired. For an updated list of available hardware through
#   Amazon Braket, please consult the `supported regions and devices documentation <https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html>`__. The general steps outlined below still hold regardless of the choice of device, though.
#

from braket.devices import Devices

device_arn = Devices.Rigetti.AspenM3


@hybrid_job(device=device_arn)  # set priority QPU
def qpu_qubit_rotation_hybrid_job(num_steps=10, stepsize=0.5):
    # AWS devices must be declared within the decorated function.
    device = qml.device(
        "braket.aws.qubit",
        device_arn=device_arn.value,  # Make sure the device ARN matches the hybrid job device ARN
        wires=2,
    )

    @qml.set_shots(1_000)
    @qml.qnode(device)
    def circuit(params):
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=0)
        return qml.expval(qml.PauliZ(0))

    opt = qml.GradientDescentOptimizer(stepsize=stepsize)
    params = np.array([0.5, 0.75])

    for i in range(num_steps):
        # update the circuit parameters
        params = opt.step(circuit, params)
        expval = circuit(params)

        log_metric(metric_name="expval", iteration_number=i, value=expval)

    return params


######################################################################
# To get a sense of how long we will wait before the hybrid job runs, we can check the hybrid job
# queue depth with ``AwsDevice(device_arn).queue_depth().jobs``. We can also check if the device is
# currently available with ``AwsDevice(device_arn).is_available``.
#
# When there are no other hybrid jobs in the queue ahead of you, and the device is available, the
# hybrid job will start running.
#
# .. warning::
#
#    Running the following cell will only run once the QPU is available. This may take a long
#    time and will result in usage fees charged to your AWS account. Only run the cell if you
#    are comfortable with the potential wait-time and costs. We recommend monitoring the Billing &
#    Cost Management Dashboard on the AWS console.
#

qpu_job = qpu_qubit_rotation_hybrid_job(num_steps=10, stepsize=0.5)
print(qpu_job)
qpu_job.result()

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  .. code-block:: none
#
#       AwsQuantumJob('arn':'arn:aws:braket:<aws-region>:<account_id>:job/qpu-qubit-rotation-hybrid-job-1695044576')
#       {'result': [0.034321330234338991, 3.058282990501767]}


######################################################################
# Next, we plot the expectation value per iteration number below. We see that on a real QPU, the data
# is not as smooth as the simulator, but the minimum still is detected correctly!
#

df = pd.DataFrame(qpu_job.metrics(metric_type=MetricType.ITERATION_NUMBER))
df.sort_values(by=["iteration_number"], inplace=True)

plt.plot(df["iteration_number"], df["expval"], "-o", color="teal")
plt.xlabel("iteration number")
plt.ylabel("expectation value")
plt.title("QPU results")
plt.show()

######################################################################
#
# .. figure:: /_static/demonstration_assets/getting_started_with_hybrid_jobs/qpu.png
#     :align: center
#     :width: 75%
#     :alt: Expectation value per iteration number on QPU.
#     :target: javascript:void(0);


######################################################################
# Conclusion
# ------------
#
# In this tutorial, we showed how to migrate from local Python functions to running algorithms on simulators and QPUs on Amazon Braket.
# We adapted the simple example of rotating a qubit using gradient descent, running this on both a local simulator and a real QPU.
# Using Amazon Braket Hybrid Jobs allowed us to run algorithms asynchronously, scale classical compute using AWS, and obtain priority access to the selected QPU for the duration of our algorithm.
#
