import collections
import os
import time
from time import sleep
from tracemalloc import start

import numpy as np
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_addon_cutting import (cut_wires, expand_observables,
                                  generate_cutting_experiments,
                                  partition_problem,
                                  reconstruct_expectation_values)
from qiskit_addon_cutting.automated_cut_finding import (DeviceConstraints,
                                                        OptimizationParameters,
                                                        find_cuts)
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2
from qiskit_ibm_runtime import Batch, SamplerV2
import pdb

from qiskit.primitives import (
    SamplerResult,  # for SamplerV1
    PrimitiveResult,  # for SamplerV2
)

from qiskit.circuit.library import EfficientSU2

from pilot.pilot_compute_service import ExecutionEngine, PilotComputeService

RESOURCE_URL_HPC = "ssh://localhost"
WORKING_DIRECTORY = os.path.join(os.environ["HOME"], "work")

pilot_compute_description_ray = {
    "resource": RESOURCE_URL_HPC,
    "working_directory": WORKING_DIRECTORY,
    "type": "ray",
    "number_of_nodes": 16,
    "cores_per_node": 64,
    "gpus_per_node": 4,
    "project": "m4408",
    "queue": "premium",
    "walltime": 15,
    "scheduler_script_commands": ["#SBATCH --constraint=gpu",
                                  "#SBATCH --gpus-per-task=1",
                                  "#SBATCH --ntasks-per-node=4",
                                  "#SBATCH --gpu-bind=none"],
}

def start_pilot(pilot_compute_description_ray):
    pcs = PilotComputeService(execution_engine=ExecutionEngine.RAY, working_directory=WORKING_DIRECTORY)
    pcd = pcs.create_pilot(pilot_compute_description=pilot_compute_description_ray)
    pcd.wait()
    time.sleep(60)
    return pcs


def pre_processing(logger, scale=1, qps=2, num_samples=10):    
    base_qubuits = 7    
    circuit = EfficientSU2(base_qubuits * scale, entanglement="linear", reps=2).decompose()
    #circuit.assign_parameters([0.4] * len(circuit.parameters), inplace=True)    
    observable = SparsePauliOp([i * scale for i in ["ZIIIIII", "IIIIIZI", "IIIIIIZ"]])

    # Specify settings for the cut-finding optimizer
    optimization_settings = OptimizationParameters(seed=111)

    # Specify the size of the QPUs available
    device_constraints = DeviceConstraints(qubits_per_subcircuit=qps)

    cut_circuit, metadata = find_cuts(circuit, optimization_settings, device_constraints)
    logger.info(
        f'Found solution using {len(metadata["cuts"])} cuts with a sampling '
        f'overhead of {metadata["sampling_overhead"]}.\n'
        f'Lowest cost solution found: {metadata["minimum_reached"]}.'
    )
    for cut in metadata["cuts"]:
        logger.info(f"{cut[0]} at circuit instruction index {cut[1]}")


    qc_w_ancilla = cut_wires(cut_circuit)
    observables_expanded = expand_observables(observable.paulis, circuit, qc_w_ancilla)

    partitioned_problem = partition_problem(
        circuit=qc_w_ancilla, observables=observables_expanded
    )
    subcircuits = partitioned_problem.subcircuits
    subobservables = partitioned_problem.subobservables
    logger.info(
        f"Sampling overhead: {np.prod([basis.overhead for basis in partitioned_problem.bases])}"
    )

    subexperiments, coefficients = generate_cutting_experiments(
        circuits=subcircuits, observables=subobservables, num_samples=num_samples
    )
    sum=0
    for i in range(len(subexperiments)):
        sum+=len(subexperiments[i])
    logger.info(f"Total subexperiments to run on backend.: {sum}")
        
    return subexperiments, coefficients, subobservables, observable, circuit

def execute_sampler(sampler, label, subsystem_subexpts, shots):
    print(sampler, label, subsystem_subexpts, shots)
    submit_start = time.time()
    job = sampler.run(subsystem_subexpts, shots=shots)
    submit_end = time.time()
    result_start = time.time()
    result = job.result()    
    result_end = time.time()
    print(f"Job {label} completed with job id {job.job_id()}, submit_time: {submit_end-submit_start} and execution_time: {result_end - result_start}, type: {type(result)}")
    return (label, result)

def run_full_circuit(observable, backend_options, full_circuit_transpilation):
    estimator = EstimatorV2(options=backend_options)
    exact_expval = estimator.run([(full_circuit_transpilation, observable)]).result()[0].data.evs
    return exact_expval

if __name__ == "__main__":
    pcs = None
    num_nodes = [2]
    for nodes in num_nodes:
        start_time = time.time()
        try:
            # Start Pilot
            pilot_compute_description_ray["number_of_nodes"] = nodes
            pcs = start_pilot(pilot_compute_description_ray)
            logger = pcs.get_logger()
            
            subexperiments, coefficients, subobservables, observable, circuit = pre_processing(logger)
            
            # backend_options = {"backend_options": {"device":"GPU"}}
            backend_options = {"backend_options": {"shots": 4096, "device":"GPU", "method":"statevector", "blocking_enable":True, "batched_shots_gpu":True, "blocking_qubits":25}}
            
            backend = AerSimulator(**backend_options["backend_options"])
            logger.info("*********************************** transpiling circuits ***********************************")
            # Transpile the subexperiments to ISA circuits
            # pdb.set_trace()/10
            pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend)

            isa_subexperiments = {
                label: pass_manager.run(partition_subexpts)
                for label, partition_subexpts in subexperiments.items()
            }            
            logger.info("*********************************** transpiling done ***********************************")
            
            
            results_tuple  = None
            for i in range(3):
                tasks=[]
                i=0
                with Batch(backend=backend) as batch:
                    sampler = SamplerV2(mode=batch)
                    logger.info(f"*********************************** len of subexperiments {len(isa_subexperiments)}*************************")
                    for label, subsystem_subexpts in isa_subexperiments.items():                        
                        logger.info(f"*********************************** len of subsystem_subexpts {len(subsystem_subexpts), subsystem_subexpts}*************************")
                        for ss in subsystem_subexpts:                        
                            task_future = pcs.submit_task(execute_sampler, sampler, label, [ss], shots=2**12, resources={'num_cpus': 1, 'num_gpus': 1, 'memory': None})
                            tasks.append(task_future)
                            i=i+1

                sub_circuit_execution_time = time.time()
                results_tuple=pcs.get_results(tasks)
                # print(results_tuple)
                sub_circuit_execution_end_time = time.time()
                logger.info(f"Execution time for subcircuits: {sub_circuit_execution_end_time-sub_circuit_execution_time}")

            # Get all samplePubResults            
            samplePubResults = collections.defaultdict(list)
            for result in results_tuple:
                samplePubResults[result[0]].extend(result[1]._pub_results)
            
            
            results = {}
            
            for label, samples in samplePubResults.items():
                results[label] = PrimitiveResult(samples)
            
            reconstruction_start_time = time.time()
            # Get expectation values for each observable term
            reconstructed_expvals = reconstruct_expectation_values(
                results,
                coefficients,
                subobservables,
            )
            reconstruction_end_time = time.time()
            logger.info(f"Execution time for reconstruction: {reconstruction_end_time-reconstruction_start_time}")
            
            final_expval = np.dot(reconstructed_expvals, observable.coeffs)            
            
            
            exact_expval = 0
            transpile_full_circuit_time = time.time()
            full_circuit_transpilation = pass_manager.run(circuit)
            transpile_full_circuit_end_time = time.time()
            logger.info(f"Execution time for full Circuit transpilation: {transpile_full_circuit_end_time-transpile_full_circuit_time}")
            
            for i in range(3):
                full_circuit_estimator_time = time.time()                           
                full_circuit_task = pcs.submit_task(run_full_circuit, observable, backend_options, full_circuit_transpilation, resources={'num_cpus': 1, 'num_gpus': 4, 'memory': None})
                exact_expval = pcs.get_results([full_circuit_task])
                full_circuit_estimator_end_time = time.time()
                
                logger.info(f"Execution time for full Circuit: {full_circuit_estimator_end_time-full_circuit_estimator_time}")            
                
            logger.info(f"Reconstructed expectation value: {np.real(np.round(final_expval, 8))}")
            logger.info(f"Exact expectation value: {np.round(exact_expval, 8)}")
            logger.info(f"Error in estimation: {np.real(np.round(final_expval-exact_expval, 8))}")
            logger.info(
                f"Relative error in estimation: {np.real(np.round((final_expval-exact_expval) / exact_expval, 8))}"
            )                    
                                                    
        except Exception as e:
            print(f"Error: {e}")
        finally:
            if pcs:                
                pcs.cancel()
