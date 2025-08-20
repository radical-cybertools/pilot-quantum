import csv
import threading
import time
import uuid
from copy import copy
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import Dict, List, Optional, Any

from qiskit import QuantumCircuit

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    print("Warning: PuLP not available. Using fallback weighted approach.")

import logging

logger = logging.getLogger(__name__)

class TaskType(Enum):
    CLASSICAL = "classical"
    QUANTUM = "quantum"
    HYBRID = "hybrid"


class Coupling(Enum):
    TIGHT = "tight"
    MEDIUM = "medium"
    LOOSE = "loose"


class Task:
    def __init__(self, type: TaskType, resource_config: dict): 
        self.type = TaskType
        self.resource_config = None
        self.resource_config = resource_config
    
        
    def get_circuit(self):
        return self.circuit
    
class QuantumTask:
    """Represents a quantum task with requirements."""
    
    def __init__(self, circuit, num_qubits: int, gate_set: List[str], resource_config: Dict = None):
        self.circuit = circuit
        self.num_qubits = num_qubits
        self.gate_set = gate_set
        self.resource_config = resource_config or {}

class QuantumResource:
    """Represents a quantum resource with capabilities."""
    
    def __init__(self, name: str, qubit_count: int, gateset: List[str], 
                 error_rate: float = 0.0, noise_level: float = 0.0, 
                 quantum_config: Dict = None, pilot_name: str = None, 
                 original_name: str = None):
        self.name = name
        self.qubit_count = qubit_count
        self.gateset = gateset
        self.error_rate = error_rate
        self.noise_level = noise_level
        self.quantum_config = quantum_config or {}
        self.pilot_name = pilot_name
        self.original_name = original_name
        
    @property
    def fidelity(self) -> float:
        """Calculate fidelity as 1 - error_rate."""
        return 1.0 - self.error_rate if self.error_rate is not None else 1.0

class OptimizedResourceSelector:
    """Intelligent resource optimization using PuLP."""
    
    def __init__(self, optimization_mode: str = "multi_objective"):
        """
        Initialize the intelligent optimizer.
        
        Args:
            optimization_mode: "multi_objective", "fidelity_first", "queue_first", or "balanced"
        """
        self.optimization_mode = optimization_mode
        self.logger = logging.getLogger(__name__)
        
        if not PULP_AVAILABLE:
            raise ImportError("PuLP is required for intelligent resource optimization. Please install with: pip install pulp>=2.7.0")
    
    def optimize_resource_selection(self, task: QuantumTask, resources: Dict[str, QuantumResource], 
                                  queue_dynamics: Dict[str, float] = None, task_id: str = 'unknown') -> Optional[QuantumResource]:
        """
        Optimize resource selection using PuLP.
        
        Args:
            task: The quantum task to optimize for
            resources: Available quantum resources
            queue_dynamics: Current queue utilization for each resource
            task_id: Task ID for correlation logging
            
        Returns:
            Best resource or None if no suitable resource found
        """
        # Filter suitable resources
        suitable_resources = self._filter_suitable_resources(task, resources)
        if not suitable_resources:
            self.logger.warning(f"[TASK:{task_id}] ⚠️ No suitable resources found for task: {task.num_qubits} qubits, gates: {task.gate_set}")
            return None
        
        self.logger.info(f"[TASK:{task_id}] 🔍 Found {len(suitable_resources)} suitable resources for optimization")
        return self._solve_optimization_problem(task, suitable_resources, queue_dynamics, task_id)
    
    def _filter_suitable_resources(self, task: QuantumTask, resources: Dict[str, QuantumResource]) -> Dict[str, QuantumResource]:
        """Filter resources that can handle the task requirements."""
        suitable = {}
        
        for name, resource in resources.items():
            # Check qubit count
            if resource.qubit_count < task.num_qubits:
                continue
            
            # Check gate compatibility
            if not self._check_gate_compatibility(task.gate_set, resource.gateset):
                continue
            
            suitable[name] = resource
        
        return suitable
    
    def _check_gate_compatibility(self, task_gates: List[str], resource_gates: List[str]) -> bool:
        """Check if resource supports all required gates."""
        # Normalize gate names for comparison
        task_gates_normalized = {gate.lower().replace('cnot', 'cx').replace('cnot', 'cx') for gate in task_gates}
        resource_gates_normalized = {gate.lower().replace('cnot', 'cx').replace('cnot', 'cx') for gate in resource_gates}
        
        # Check if all task gates are supported
        return task_gates_normalized.issubset(resource_gates_normalized)
    
    def _solve_optimization_problem(self, task: QuantumTask, suitable_resources: Dict[str, QuantumResource], 
                                  queue_dynamics: Dict[str, float] = None, task_id: str = 'unknown') -> Optional[QuantumResource]:
        """Solve the resource selection optimization problem using PuLP."""
        
        # Create optimization problem
        prob = pulp.LpProblem("Quantum_Resource_Selection", pulp.LpMinimize)
        
        # Decision variables: binary variable for each resource
        resource_vars = {}
        for name in suitable_resources.keys():
            resource_vars[name] = pulp.LpVariable(f"resource_{name}", cat=pulp.LpBinary)
        
        # Objective function components
        fidelity_terms = []
        queue_terms = []
        
        for name, resource in suitable_resources.items():
            # Fidelity component (higher fidelity = lower cost)
            fidelity_cost = 1.0 - resource.fidelity
            fidelity_terms.append(fidelity_cost * resource_vars[name])
            
            # Queue component (lower queue = lower cost)
            queue_util = queue_dynamics.get(name, 0.0) if queue_dynamics else 0.0
            queue_terms.append(queue_util * resource_vars[name])
        
        # Objective function based on optimization mode
        if self.optimization_mode == "high_fidelity":
            # Prioritize high fidelity (for hardware/fake backends where fidelity varies)
            objective = (
                0.8 * pulp.lpSum(fidelity_terms) +      # 80% weight on fidelity
                0.2 * pulp.lpSum(queue_terms)           # 20% weight on queue
            )
        elif self.optimization_mode == "high_speed":
            # Prioritize high speed (for simulators where fidelity is perfect)
            objective = (
                0.2 * pulp.lpSum(fidelity_terms) +      # 20% weight on fidelity (simulators are perfect)
                0.8 * pulp.lpSum(queue_terms)           # 80% weight on queue (speed matters more)
            )
        else:  # balanced (default)
            # Equal weights for fidelity and queue
            objective = (
                0.5 * pulp.lpSum(fidelity_terms) +      # 50% weight on fidelity
                0.5 * pulp.lpSum(queue_terms)           # 50% weight on queue
            )
        
        prob += objective
        
        # Constraint: exactly one resource must be selected
        prob += pulp.lpSum(resource_vars.values()) == 1
        
        # Solve the problem
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # Extract solution
        if prob.status == pulp.LpStatusOptimal:
            for name, var in resource_vars.items():
                if var.value() == 1:
                    selected_resource = suitable_resources[name]
                    queue_util = queue_dynamics.get(name, 0.0) if queue_dynamics else 0.0
                    self.logger.info(f"[TASK:{task_id}] 🎯 Optimization selected: {name} (fidelity={selected_resource.fidelity:.3f}, "
                                   f"queue={queue_util:.3f}, noise={selected_resource.noise_level:.3f})")
                    return selected_resource
        
        self.logger.warning(f"[TASK:{task_id}] ⚠️ Optimization problem could not be solved optimally")
        return None
    

    


import csv
import threading
import time
import uuid
from copy import copy
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional, Any

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    print("Warning: PuLP not available. Using fallback weighted approach.")

class Q_DREAMER:
    def __init__(self, qdreamer_config_or_resources, quantum_resources_or_config=None, simulation=True, cache_ttl_seconds=30):
        """
        QDREAMER framework provides intelligent resource selection for quantum tasks.
        
        Supports both parameter orders for backward compatibility:
        - Old: Q_DREAMER(qdreamer_config, quantum_resources, ...)
        - New: Q_DREAMER(quantum_resources, qdreamer_config, ...)
        
        :param qdreamer_config_or_resources: Either qdreamer_config dict or quantum_resources dict
        :param quantum_resources_or_config: Either quantum_resources dict or qdreamer_config dict
        :param simulation: Whether to run in simulation mode
        :param cache_ttl_seconds: How long to cache queue information (default: 30 seconds)
        """
        # Handle parameter order detection - now supports optimization_mode
        if isinstance(qdreamer_config_or_resources, dict) and ('load_balancing' in qdreamer_config_or_resources or 'optimization_mode' in qdreamer_config_or_resources):
            # First parameter is qdreamer_config
            self.qdreamer_config = qdreamer_config_or_resources
            self.quantum_resources = quantum_resources_or_config or {}
        else:
            # First parameter is quantum_resources
            self.quantum_resources = qdreamer_config_or_resources
            self.qdreamer_config = quantum_resources_or_config or {}
        
        self.simulation = simulation
        self.cache_ttl_seconds = cache_ttl_seconds
        
        # Initialize intelligent optimizer
        optimization_mode = self.qdreamer_config.get('optimization_mode', 'multi_objective')
        self.optimizer = OptimizedResourceSelector(optimization_mode)
        

        
        # Cache for queue information
        self._queue_cache = {}
        self._cache_timestamps = {}
        self._cache_lock = Lock()
        
        # Background monitoring
        self._monitoring_active = False
        self._monitor_thread = None
        self._monitor_interval = 60  # seconds
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize queue dynamics
        self.queue_dynamics = self.qdreamer_config.get("queue_dynamics", {})
        
        # Start background monitoring
        self._start_background_monitoring()
    
    @property
    def config(self):
        """Get the QDREAMER configuration."""
        return {
            'qdreamer_config': self.qdreamer_config,
            'optimization_mode': self.optimizer.optimization_mode,
            'simulation': self.simulation,
            'cache_ttl_seconds': self.cache_ttl_seconds
        }
    
    def get_best_resource(self, quantum_task: QuantumTask, task_id: str = 'unknown') -> Optional[QuantumResource]:
        """
        Get the best quantum resource for a given task using intelligent optimization.
        
        Args:
            quantum_task: The quantum task to find resources for
            task_id: Task ID for correlation logging
            
        Returns:
            Best quantum resource or None if no suitable resource found
        """
        self.logger.info(f"[TASK:{task_id}] 🔍 QDREAMER selecting best resource for task: {quantum_task.num_qubits} qubits, gates: {quantum_task.gate_set}")
        
        # Get current queue dynamics
        current_queue_dynamics = self._get_current_queue_dynamics()
        
        # Use intelligent optimization
        best_resource = self.optimizer.optimize_resource_selection(
            quantum_task, self.quantum_resources, current_queue_dynamics, task_id
        )
        
        if best_resource:
            self.logger.info(f"[TASK:{task_id}] ✅ QDREAMER selected: {best_resource.name}")
            return best_resource
        else:
            self.logger.warning(f"[TASK:{task_id}] ⚠️ No suitable quantum resource found!")
            return None
    
    def _get_current_queue_dynamics(self) -> Dict[str, float]:
        """Get current queue utilization for all resources from executors."""
        current_time = time.time()
        
        with self._cache_lock:
            # Check if cache is still valid
            if (current_time - self._cache_timestamps.get('queue_dynamics', 0)) < self.cache_ttl_seconds:
                return self._queue_cache.get('queue_dynamics', {})
            
            # Update queue dynamics from executors
            queue_dynamics = {}
            
            # Group resources by executor type
            executor_resources = {}
            for resource_name, resource in self.quantum_resources.items():
                # Extract executor type from resource name (e.g., "pilot1_qiskit_aer" -> "qiskit")
                if hasattr(resource, 'pilot_name'):
                    # Use pilot name to identify executor
                    executor_key = resource.pilot_name
                else:
                    # Fallback: extract from resource name
                    parts = resource_name.split('_')
                    executor_key = parts[1] if len(parts) > 1 else 'unknown'
                
                if executor_key not in executor_resources:
                    executor_resources[executor_key] = []
                executor_resources[executor_key].append((resource_name, resource))
            
            # Get queue information from each executor
            for executor_key, resources in executor_resources.items():
                try:
                    # Get queue lengths from the executor
                    # This would need to be implemented in the quantum execution service
                    # For now, use the existing queue_dynamics as fallback
                    for resource_name, resource in resources:
                        original_name = getattr(resource, 'original_name', resource_name)
                        queue_util = self.queue_dynamics.get(original_name, 0.0)
                        queue_dynamics[resource_name] = queue_util
                        
                except Exception as e:
                    self.logger.warning(f"Failed to get queue dynamics for executor {executor_key}: {e}")
                    # Fallback to existing queue_dynamics
                    for resource_name, resource in resources:
                        original_name = getattr(resource, 'original_name', resource_name)
                        queue_util = self.queue_dynamics.get(original_name, 0.0)
                        queue_dynamics[resource_name] = queue_util
            
            # Update cache
            self._queue_cache['queue_dynamics'] = queue_dynamics
            self._cache_timestamps['queue_dynamics'] = current_time
            
            return queue_dynamics
    
    def _start_background_monitoring(self):
        """Start background monitoring of queue dynamics."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_queues, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Background queue monitoring started")
    
    def _monitor_queues(self):
        """Background thread to monitor queue dynamics."""
        while self._monitoring_active:
            try:
                # Update queue dynamics (in a real implementation, this would query actual queue status)
                if self.simulation:
                    # Simulate queue changes
                    for resource_name in self.quantum_resources.keys():
                        original_name = getattr(self.quantum_resources[resource_name], 'original_name', resource_name)
                        if original_name in self.queue_dynamics:
                            # Simulate some queue variation
                            current = self.queue_dynamics[original_name]
                            variation = (hash(f"{original_name}_{int(time.time() / 60)}") % 100) / 1000.0
                            self.queue_dynamics[original_name] = max(0.0, min(1.0, current + variation))
                
                time.sleep(self._monitor_interval)
                
            except Exception as e:
                self.logger.error(f"Error in queue monitoring: {e}")
                time.sleep(self._monitor_interval)
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("Stopped background queue monitoring")


 