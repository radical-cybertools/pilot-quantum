"""
Quantum Execution Service

Service for executing quantum circuits using the executor pattern.
"""

import logging
from typing import Dict, Any, Optional
from ..executors.executor_factory import QuantumExecutorFactory
from ..executors.base_executor import BaseExecutor


class QuantumExecutionService:
    """
    Service for executing quantum circuits.
    
    Manages the execution of quantum circuits using appropriate executors
    based on the resource type and configuration.
    """
    
    def __init__(self):
        """Initialize the quantum execution service."""
        self.logger = logging.getLogger(__name__)
        self._executors: Dict[str, BaseQuantumExecutor] = {}
    
    def execute_circuit(self, quantum_task, resource, *args, **kwargs):
        """
        Execute a quantum circuit using the appropriate executor.
        
        Args:
            quantum_task: Quantum task containing the circuit
            resource: Quantum resource information
            *args: Additional arguments for circuit execution
            **kwargs: Additional keyword arguments for circuit execution
            
        Returns:
            Circuit execution result
        """
        # Extract task_id for correlation if provided
        task_id = kwargs.pop('task_id', 'unknown')
        
        try:
            # Determine executor type from resource
            executor_type = self._get_executor_type_from_resource(resource)
            self.logger.info(f"[TASK:{task_id}] 🔧 Using {executor_type} executor for {resource.name}")
            
            # Get or create executor
            executor = self._get_executor(executor_type, resource, task_id)
            
            # Execute circuit
            self.logger.info(f"[TASK:{task_id}] ⚡ Executing circuit on {resource.name}...")
            result = executor.execute_circuit(quantum_task.circuit, *args, **kwargs)
            
            self.logger.info(f"[TASK:{task_id}] ✅ Circuit executed successfully using {executor_type} executor")
            return result
            
        except Exception as e:
            self.logger.error(f"[TASK:{task_id}] ❌ Circuit execution failed: {str(e)}")
            raise
    
    def _get_executor_type_from_resource(self, resource) -> str:
        """
        Determine executor type from resource information.
        
        Args:
            resource: Quantum resource information
            
        Returns:
            Executor type string
        """
        resource_name = resource.name.lower()
        
        if 'qiskit' in resource_name:
            return 'qiskit'
        elif 'pennylane' in resource_name:
            return 'pennylane'
        elif 'braket' in resource_name:
            return 'braket'
        elif 'ibmq' in resource_name:
            return 'ibmq'
        else:
            # Default to qiskit for unknown resources
            return 'qiskit'
    
    def _get_executor(self, executor_type: str, resource, task_id: str = 'unknown') -> BaseExecutor:
        """
        Get or create an executor for the given type.
        
        Args:
            executor_type: Type of executor
            resource: Quantum resource information
            task_id: Task ID for correlation logging
            
        Returns:
            Quantum executor instance
        """
        # Create cache key
        cache_key = f"{executor_type}_{resource.name}"
        
        if cache_key not in self._executors:
            # Create executor configuration from resource
            config = self._create_executor_config(executor_type, resource)
            
            # Create executor
            executor = QuantumExecutorFactory.create_executor(executor_type, config)
            self._executors[cache_key] = executor
            
            self.logger.info(f"[TASK:{task_id}] 🔧 Created executor for {executor_type} with resource {resource.name}")
        else:
            self.logger.info(f"[TASK:{task_id}] 🔄 Reusing cached executor for {executor_type} with resource {resource.name}")
        
        return self._executors[cache_key]
    
    def _create_executor_config(self, executor_type: str, resource) -> Dict[str, Any]:
        """
        Create executor configuration from resource information.
        
        Args:
            executor_type: Type of executor
            resource: Quantum resource information
            
        Returns:
            Executor configuration dictionary
        """
        config = {
            'shots': 1000,  # Default shots
        }
        
        # Add executor-specific configuration
        if executor_type == 'qiskit':
            # Read backend from resource's quantum_config or fallback to resource name
            quantum_config = getattr(resource, 'quantum_config', {})
            config['backend'] = quantum_config.get('backend', resource.name)
            # Add custom backend parameters for noise modeling
            if hasattr(resource, 'error_rate') and resource.error_rate is not None:
                config['error_rate'] = resource.error_rate
            if hasattr(resource, 'noise_level') and resource.noise_level is not None:
                config['noise_level'] = resource.noise_level
            if hasattr(resource, 'qubit_count') and resource.qubit_count is not None:
                config['qubit_count'] = resource.qubit_count
        elif executor_type == 'pennylane':
            # Read device directly from resource's quantum_config
            quantum_config = getattr(resource, 'quantum_config', {})
            config['device'] = quantum_config.get('device', 'default.qubit')
            config['wires'] = resource.qubit_count
        elif executor_type == 'braket':
            config['device_arn'] = resource.name
        elif executor_type == 'ibmq':
            config['backend'] = resource.name
            # Token should be provided in environment or config
        
        return config
    
    def get_executor_info(self, executor_type: str) -> Dict[str, Any]:
        """
        Get information about an executor type.
        
        Args:
            executor_type: Type of executor
            
        Returns:
            Executor information dictionary
        """
        return QuantumExecutorFactory.get_executor_info(executor_type)
    
    def get_supported_executors(self) -> list:
        """
        Get list of supported executor types.
        
        Returns:
            List of supported executor types
        """
        return QuantumExecutorFactory.get_supported_types()
    
    def clear_executors(self):
        """Clear all cached executors."""
        self._executors.clear()
        self.logger.info("Cleared all cached executors")
