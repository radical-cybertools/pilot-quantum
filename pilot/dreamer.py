import logging
import uuid
from enum import Enum
from typing import Dict, List, Optional
from qiskit import QuantumCircuit

logger = logging.getLogger('pilot.pcs_logger')


class TaskType(Enum):
    CLASSICAL = "classical"
    QUANTUM = "quantum"
    HYBRID = "hybrid"

class Coupling(Enum):
    TIGHT = "tight"
    MEDIUM = "medium"
    LOOSE = "loose"

class Task:
    def __init__(self, type: TaskType, resource_config: dict, args: tuple = None, kwargs: Dict = None): 
        self.type = type
        self.resource_config = resource_config
        self.task_id = kwargs.get('task_id') if kwargs and kwargs.get('task_id') else str(uuid.uuid4())
        self.args = args or ()
        self.kwargs = kwargs or {}
    
class QuantumTask(Task):
    """Represents a quantum task with requirements."""
    
    def __init__(self, circuits: List[QuantumCircuit], resource_config: Dict = None, args: tuple = None, kwargs: Dict = None):
        """
        Initialize a QuantumTask.
        
        Args:
            circuit: List of quantum circuits
            resource_config: Resource configuration dictionary
            args: Task arguments tuple
            kwargs: Task keyword arguments dictionary
        """
        super().__init__(TaskType.QUANTUM, resource_config, args, kwargs)
        self._circuits = circuits
        
    @property
    def circuits(self):
        return self._circuits

class QuantumResource:
    """Represents a quantum resource with capabilities."""
    
    def __init__(self, name: str, qubit_count: int, gateset: List[str], 
                 error_rate: float = 0.0, noise_level: float = 0.0, 
                 quantum_config: Dict = None, pilot_name: str = None):
        self.name = name
        self.qubit_count = qubit_count
        self.gateset = gateset
        self.error_rate = error_rate
        self.noise_level = noise_level
        self.quantum_config = quantum_config or {}
        self.pilot_name = pilot_name

class DreamerStrategyType(Enum):
    LEAST_ERROR_RATE = "least_error_rate"
    ROUND_ROBIN = "round_robin"
    LEAST_BUSY = "least_busy"


class StrategySelector:
    """Strategy selector for resource selection."""
    def __init__(self, strategy_type: DreamerStrategyType):
        self.strategy_type = strategy_type        
        self.resources = []
    
    def select_resource(self, task: QuantumTask, resources: Dict[str, QuantumResource]) -> Optional[QuantumResource]:
        raise NotImplementedError("Strategy must implement this method")

class RoundRobinStrategy:
    """Round Robin strategy for resource selection."""
    def __init__(self):
        self.current_index = 0
    
    def select_resource(self, task: QuantumTask, resources: Dict[str, QuantumResource]) -> Optional[QuantumResource]:
        if not resources:
            return None
        resource_list = list(resources.values())
        resource = resource_list[self.current_index % len(resource_list)]
        self.current_index += 1
        return resource

class LeastErrorRateStrategy:
    """Least Error Rate strategy for resource selection."""
    def select_resource(self, task: QuantumTask, resources: Dict[str, QuantumResource]) -> Optional[QuantumResource]:
        if not resources:
            return None
        return min(resources.values(), key=lambda x: x.error_rate)

class LeastBusyStrategy:
    """Least Busy strategy for resource selection."""
    def select_resource(self, task: QuantumTask, resources: Dict[str, QuantumResource]) -> Optional[QuantumResource]:
        if not resources:
            return None
        # For now, return first available resource (can be enhanced with actual queue data)
        return list(resources.values())[0]


class Q_DREAMER:
    def __init__(self, quantum_resources, dreamer_type: DreamerStrategyType = DreamerStrategyType.LEAST_ERROR_RATE):
        """ QDREAMER framework provides intelligent resource selection for quantum tasks. """
        self.quantum_resources = quantum_resources
        self.strategy_type = dreamer_type
        self.strategy_selector = self.get_strategy()

    def get_strategy(self) -> StrategySelector:
        if self.strategy_type == DreamerStrategyType.LEAST_ERROR_RATE:
            return LeastErrorRateStrategy()
        elif self.strategy_type == DreamerStrategyType.ROUND_ROBIN:
            return RoundRobinStrategy()
        elif self.strategy_type == DreamerStrategyType.LEAST_BUSY:
            return LeastBusyStrategy()
        else:
            raise ValueError(f"Invalid strategy type: {self.strategy_type}")        
    
    def get_best_resource(self, quantum_task: QuantumTask) -> Optional[QuantumResource]:
        """
        Get the best quantum resource for a given task
        
        Args:
            quantum_task: The quantum task to find best resource for
            
        Returns:
            Best quantum resource or None if no suitable resource found
        """

        best_resource = self.strategy_selector.select_resource(quantum_task, self.quantum_resources)
        print(f"Best resource selected: {best_resource.name}")
        return best_resource


 