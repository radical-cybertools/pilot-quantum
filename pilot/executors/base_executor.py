"""
Base Quantum Executor

Defines the interface that all quantum executors must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger('pilot.pcs_logger')

class BaseExecutor(ABC):
    """Base class for all quantum executors."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger('pilot.pcs_logger')
    
    @abstractmethod
    def execute_circuit(self, circuit, resource_name: str, **kwargs) -> Any:
        """Execute a quantum circuit on the specified resource."""
        pass
    
    @abstractmethod
    def get_available_resources(self) -> Dict[str, Any]:
        """Get available quantum resources for this executor."""
        pass
    
    def get_queue_lengths(self) -> Dict[str, float]:
        """
        Get current queue lengths for all backends managed by this executor.
        
        Returns:
            Dict mapping backend names to queue utilization (0.0 to 1.0)
        """
        # Default implementation returns empty dict
        # Subclasses should override to provide real queue information
        return {}
    
    def get_backend_status(self, backend_name: str) -> Dict[str, Any]:
        """
        Get detailed status information for a specific backend.
        
        Args:
            backend_name: Name of the backend
            
        Returns:
            Dict containing status information (queue_length, status, etc.)
        """
        # Default implementation returns basic info
        return {
            "name": backend_name,
            "queue_length": 0.0,
            "status": "unknown"
        }
    
    def is_simulator(self) -> bool:
        """
        Check if this executor primarily uses simulators.
        
        Returns:
            True if this executor uses simulators, False for real hardware
        """
        # Default implementation - subclasses should override
        return True
    
    def calculate_execution_fidelity(self, circuit, result, expected_state=None):
        """
        Calculate the execution fidelity of a circuit by comparing actual vs expected results.
        
        Args:
            circuit: The executed quantum circuit
            result: The execution result from the backend
            expected_state: Optional expected state vector for comparison
            
        Returns:
            float: Execution fidelity (0.0 to 1.0), or None if calculation fails
        """
        # Default implementation - subclasses should override
        self.logger.warning("Execution fidelity calculation not implemented for this executor")
        return None