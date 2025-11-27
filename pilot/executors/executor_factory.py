"""
Quantum Executor Factory

A clean, maintainable factory for creating quantum executors.
Adding a new executor requires only adding it to the EXECUTOR_REGISTRY.
"""

from typing import Dict, Any, Optional, Type
from .base_executor import BaseExecutor
from .qiskit_executor import QiskitExecutor
from .pennylane_executor import PennylaneExecutor
from .braket_executor import BraketExecutor
from .ibmq_executor import IBMQExecutor


# Registry of all available executors
# To add a new executor, simply add it here:
# 
# Example: Adding a new 'cirq' executor
# 1. Create your executor class: class CirqExecutor(BaseExecutor): ...
# 2. Import it above: from .cirq_executor import CirqExecutor
# 3. Add it to the registry below:
#    'cirq': CirqExecutor,
#
EXECUTOR_REGISTRY = {
    'qiskit': QiskitExecutor,
    'pennylane': PennylaneExecutor,
    'braket': BraketExecutor,
    'ibmq': IBMQExecutor,
}


class QuantumExecutorFactory:
    """
    Factory for creating quantum executors.
    
    Usage:
        # Create an executor
        executor = QuantumExecutorFactory.create_executor('qiskit', config)
        
        # Check if supported
        if QuantumExecutorFactory.is_supported('pennylane'):
            executor = QuantumExecutorFactory.create_executor('pennylane')
            
        # Get all supported types
        executors = QuantumExecutorFactory.get_supported_types()
    """
    
    @classmethod
    def create_executor(cls, executor_type: str, config: Optional[Dict[str, Any]] = None) -> BaseExecutor:
        """
        Create a quantum executor.
        
        Args:
            executor_type: Type of executor ('qiskit', 'pennylane', 'braket', 'ibmq')
            config: Optional configuration dictionary
            
        Returns:
            Configured executor instance
            
        Raises:
            ValueError: If executor type is not supported
        """
        if not cls.is_supported(executor_type):
            supported = ', '.join(cls.get_supported_types())
            raise ValueError(f"Unsupported executor type: '{executor_type}'. Supported: {supported}")
        
        executor_class = EXECUTOR_REGISTRY[executor_type]
        return executor_class(executor_type, config)
    
    @classmethod
    def is_supported(cls, executor_type: str) -> bool:
        """Check if executor type is supported."""
        return executor_type in EXECUTOR_REGISTRY
    
    @classmethod
    def get_supported_types(cls) -> list:
        """Get list of all supported executor types."""
        return list(EXECUTOR_REGISTRY.keys())
    
    @classmethod
    def get_executor_info(cls, executor_type: str) -> Dict[str, Any]:
        """
        Get detailed information about an executor type.
        
        Args:
            executor_type: Type of executor to get info for
            
        Returns:
            Dictionary with executor information
        """
        if not cls.is_supported(executor_type):
            return {
                'supported': False,
                'error': f"Executor type '{executor_type}' not supported"
            }
        
        executor_class = EXECUTOR_REGISTRY[executor_type]
        
        # Check availability
        try:
            temp_executor = executor_class("temp", {})
            available = temp_executor.is_available()
        except Exception as e:
            available = False
            error = str(e)
        
        return {
            'supported': True,
            'available': available,
            'class_name': executor_class.__name__,
            'description': executor_class.__doc__ or 'No description available',
            'error': error if not available else None
        }
    
    @classmethod
    def get_all_executor_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all supported executors.
        
        Returns:
            Dictionary mapping executor types to their info
        """
        return {
            executor_type: cls.get_executor_info(executor_type)
            for executor_type in cls.get_supported_types()
        }


# Convenience functions for easier access
def create_executor(executor_type: str, config: Optional[Dict[str, Any]] = None) -> BaseExecutor:
    """Convenience function to create an executor."""
    return QuantumExecutorFactory.create_executor(executor_type, config)


def is_executor_supported(executor_type: str) -> bool:
    """Convenience function to check if executor is supported."""
    return QuantumExecutorFactory.is_supported(executor_type)


def get_supported_executors() -> list:
    """Convenience function to get supported executor types."""
    return QuantumExecutorFactory.get_supported_types()
