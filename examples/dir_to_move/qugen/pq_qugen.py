import os

import pennylane as qml
from pilot.pilot_compute_service import PilotComputeService

from qugen.main.generator.discrete_qcbm_model_handler import (
    DiscreteQCBMModelHandler,
)
from qugen.main.data.data_handler import load_data


RESOURCE_URL_HPC = "slurm://localhost"
WORKING_DIRECTORY = os.path.join(os.environ["HOME"], "work")

pilot_compute_description_dask = {
    "resource": RESOURCE_URL_HPC,
    "working_directory": WORKING_DIRECTORY,
    "queue": "debug",
    "type": "dask",
    "walltime": 30,
    "project": "m4408",
    "number_of_nodes": 1,
    "cores_per_node": 2,
    "scheduler_script_commands": ["#SBATCH --constraint=cpu"]
}


def start_pilot():
    pcs = PilotComputeService()
    dp = pcs.create_pilot(pilot_compute_description=pilot_compute_description_dask)
    dp.wait()
    return dp

data_set_name = "X_2D"
data_set_path = f"./training_data/{data_set_name}"
data, _ = load_data(data_set_path)
model = DiscreteQCBMModelHandler()

# build a new model:

def train_model(data_set_name, data, model):
    os.environ['JAX_PLATFORMS']="cpu"
    model.build(
        "discrete",
        data_set_name,
        n_qubits=8,
        n_registers=2,
        circuit_depth=2,
        initial_sigma=0.01,
        circuit_type="copula",
        transformation="pit",
        hot_start_path="", #path to pre-trained model parameters
    )

    # train a quantum generative model:

    model.train(
    data,
    n_epochs=500,
    batch_size=200,
    hist_samples=100000)

    # evaluate the performance of the trained model:

    evaluation_df = model.evaluate(data)

    # find the model with the minimum Kullbach-Liebler divergence:

    minimum_kl_data = evaluation_df.loc[evaluation_df["kl_original_space"].idxmin()]
    minimum_kl_calculated = minimum_kl_data["kl_original_space"]
    print(f"{minimum_kl_calculated=}")

if __name__ == "__main__":
    dask_pilot, dask_client = None, None

    try:
        # Start Pilot
        dask_pilot = start_pilot()

        dask_client = dask_pilot.get_client()
        print(dask_client.scheduler_info())

        a = dask_pilot.submit_task("train_model", train_model, data_set_name, data, model)
        print(f"{a.result()}\n")
    finally:
        if dask_pilot:
            dask_pilot.cancel()
