"""
IBM Quantum Runtime Executor

Handles execution of quantum circuits using IBM Quantum Runtime service.
"""

from typing import Any, Dict, Optional
from .base_executor import BaseExecutor


class IBMQExecutor(BaseExecutor):
    """
    Executor for IBM Quantum Runtime circuits.
    
    Supports IBM Quantum hardware and simulators through the Runtime service.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize IBM Quantum executor.
        
        Args:
            name: Executor name
            config: Configuration dictionary with backend settings
        """
        super().__init__(name, config)
        self.backend_name = config.get('backend', 'ibmq_qasm_simulator') if config else 'ibmq_qasm_simulator'
        self.token = config.get('token') if config else None
        self.shots = config.get('shots', 1000) if config else 1000
        self._service = None
        self._backend = None
    
    def execute_circuit(self, circuit, *args, **kwargs):
        """
        Execute an IBM Quantum circuit.
        
        Args:
            circuit: Qiskit QuantumCircuit or circuit function
            *args: Additional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Circuit execution result
        """
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
            
            # Get the circuit
            if callable(circuit):
                qiskit_circuit = circuit(*args, **kwargs)
            else:
                qiskit_circuit = circuit
            
            # Get service and backend
            service = self._get_service()
            backend = self._get_backend(service)
            
            # Execute circuit
            sampler = Sampler(session=backend)
            job = sampler.run(qiskit_circuit, shots=self.shots)
            result = job.result()
            
            return result
            
        except Exception as e:
            raise Exception(f"IBM Quantum execution failed: {str(e)}")
    
    def get_backend_info(self) -> Dict[str, Any]:
        """
        Get information about the IBM Quantum backend.
        
        Returns:
            Dictionary containing backend information
        """
        try:
            service = self._get_service()
            backend = self._get_backend(service)
            
            return {
                'name': backend.name,
                'configuration': backend.configuration().to_dict(),
                'properties': backend.properties().to_dict() if backend.properties() else None
            }
        except Exception:
            return {
                'name': self.backend_name,
                'configuration': None,
                'properties': None
            }
    
    def is_available(self) -> bool:
        """
        Check if IBM Quantum Runtime is available.
        
        Returns:
            True if IBM Quantum Runtime is available, False otherwise
        """
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            return True
        except ImportError:
            return False
    
    def get_available_resources(self) -> Dict[str, Any]:
        """
        Get available quantum resources for this executor.
        
        Returns:
            Dictionary containing backend information
        """
        try:
            service = self._get_service()
            backend = self._get_backend(service)
            
            return {
                'name': backend.name,
                'configuration': backend.configuration().to_dict(),
                'properties': backend.properties().to_dict() if backend.properties() else None
            }
        except Exception:
            return {
                'name': self.backend_name,
                'configuration': None,
                'properties': None
            }
    
    def get_queue_lengths(self) -> Dict[str, float]:
        """
        Get current queue lengths for IBM Quantum backends.
        
        This queries the actual backend status to get real queue information.
        
        Returns:
            Dict mapping backend names to queue utilization (0.0 to 1.0)
        """
        try:
            service = self._get_service()
            backend = self._get_backend(service)
            
            # Get backend status
            status = backend.status()
            
            # Calculate queue utilization based on pending jobs
            pending_jobs = status.pending_jobs
            max_jobs = getattr(status, 'max_jobs', 100)  # Default max jobs
            
            queue_utilization = min(1.0, pending_jobs / max_jobs) if max_jobs > 0 else 0.0
            
            return {self.backend_name: queue_utilization}
            
        except Exception as e:
            self.logger.warning(f"Failed to get queue length for {self.backend_name}: {e}")
            return {self.backend_name: 0.0}
    
    def get_backend_status(self, backend_name: str) -> Dict[str, Any]:
        """
        Get detailed status information for an IBM Quantum backend.
        
        Args:
            backend_name: Name of the backend
            
        Returns:
            Dict containing status information
        """
        if backend_name != self.backend_name:
            return {
                "name": backend_name,
                "queue_length": 0.0,
                "status": "unknown"
            }
        
        try:
            service = self._get_service()
            backend = self._get_backend(service)
            status = backend.status()
            
            return {
                "name": backend_name,
                "queue_length": min(1.0, status.pending_jobs / getattr(status, 'max_jobs', 100)),
                "status": status.status.value,
                "type": "hardware" if not "simulator" in backend_name.lower() else "simulator",
                "pending_jobs": status.pending_jobs,
                "max_jobs": getattr(status, 'max_jobs', 100)
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to get backend status for {backend_name}: {e}")
            return {
                "name": backend_name,
                "queue_length": 0.0,
                "status": "unknown"
            }
    
    def _get_service(self):
        """
        Get the IBM Quantum Runtime service.
        
        Returns:
            QiskitRuntimeService instance
        """
        if self._service is None:
            from qiskit_ibm_runtime import QiskitRuntimeService
            
            if self.token:
                self._service = QiskitRuntimeService(token=self.token)
            else:
                self._service = QiskitRuntimeService()
        
        return self._service
    
    def _get_backend(self, service):
        """
        Get the IBM Quantum backend.
        
        Args:
            service: QiskitRuntimeService instance
            
        Returns:
            IBM Quantum backend instance
        """
        if self._backend is None:
            self._backend = service.backend(self.backend_name)
        
        return self._backend
