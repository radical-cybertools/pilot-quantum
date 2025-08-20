import csv
import threading
import time
import uuid
from copy import copy
from datetime import datetime

# Lazily initialized, per-worker cached Q_DREAMER instance
_WORKER_QDREAMER = None


# Minimal metrics schema (mirrors PilotComputeService.METRICS)
METRICS = {
    'task_id': None,
    'pilot_scheduled': None,
    'submit_time': datetime.now(),
    'wait_time_secs': None,
    'staging_time_secs': 0,
    'input_staging_data_size_bytes': 0,
    'completion_time': None,
    'execution_secs': None,
    'status': None,
    'error_msg': None,
}
SORTED_METRICS_FIELDS = sorted(METRICS.keys())


def _get_or_create_worker_qdreamer(quantum_resources):
    global _WORKER_QDREAMER
    if _WORKER_QDREAMER is None:
        from pilot.dreamer import Q_DREAMER
        # Create QDREAMER with default high_fidelity optimization
        qdreamer_config = {"optimization_mode": "high_fidelity"}
        _WORKER_QDREAMER = Q_DREAMER(qdreamer_config, quantum_resources)
    return _WORKER_QDREAMER


def quantum_execution_remote(metrics_fn, quantum_task, quantum_resources, *args, **kwargs):
    """
    Remote-executed function that selects best resource on the worker and executes the circuit.
    Uses a per-worker cached Q_DREAMER instance to avoid repeated initialization overhead.
    """
    import logging
    
    # Generate unique task ID for correlation
    task_id = f"quantum-{uuid.uuid4()}"
    task_metrics = copy(METRICS)
    task_metrics["task_id"] = task_id
    task_metrics["submit_time"] = datetime.now()
    task_metrics["status"] = "RUNNING"

    # Create task-specific logger with correlation ID
    logger = logging.getLogger(__name__)
    
    # Log task start with correlation ID
    logger.info(f"[TASK:{task_id}] 🚀 Starting quantum task execution")
    logger.info(f"[TASK:{task_id}] 📋 Task details: {quantum_task.num_qubits} qubits, gates: {quantum_task.gate_set}")
    
    task_execution_start_time = time.time()
    result = None

    try:
        # Initialize or reuse Q_DREAMER cached in this worker process
        logger.info(f"[TASK:{task_id}] 🧠 Initializing QDREAMER for resource selection...")
        worker_qdreamer = _get_or_create_worker_qdreamer(quantum_resources)

        # Select best resource for this task
        logger.info(f"[TASK:{task_id}] 🔍 QDREAMER selecting best resource for task: {quantum_task.num_qubits} qubits, gates: {quantum_task.gate_set}")
        best_resource = worker_qdreamer.get_best_resource(quantum_task, task_id)
        if not best_resource:
            raise Exception(
                f"No suitable quantum resource found for task with {quantum_task.num_qubits} qubits and gates {quantum_task.gate_set}"
            )

        task_metrics["pilot_scheduled"] = best_resource.name
        logger.info(f"[TASK:{task_id}] ✅ QDREAMER selected: {best_resource.name}")

        # Execute the quantum task
        logger.info(f"[TASK:{task_id}] ⚡ Executing circuit on {best_resource.name}...")
        from pilot.services.quantum_execution_service import QuantumExecutionService
        execution_service = QuantumExecutionService()
        result = execution_service.execute_circuit(quantum_task, best_resource, task_id=task_id, *args, **kwargs)
        task_metrics["status"] = "SUCCESS"
        logger.info(f"[TASK:{task_id}] ✅ Task completed successfully on {best_resource.name}")

    except Exception as e:
        task_metrics["status"] = "FAILED"
        task_metrics["error_msg"] = str(e)
        task_metrics["pilot_scheduled"] = "unknown"
        logger.error(f"[TASK:{task_id}] ❌ Task failed: {str(e)}")

    task_metrics["completion_time"] = datetime.now()
    task_metrics["execution_secs"] = round((time.time() - task_execution_start_time), 4)
    
    logger.info(f"[TASK:{task_id}] 📊 Task metrics: execution_time={task_metrics['execution_secs']}s, status={task_metrics['status']}")

    # Persist metrics
    lock = threading.Lock()
    with lock:
        with open(metrics_fn, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=SORTED_METRICS_FIELDS)
            writer.writerow(task_metrics)

    return result


