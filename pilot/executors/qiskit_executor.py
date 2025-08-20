"""
Qiskit Quantum Executor

Handles execution of quantum circuits using Qiskit backends.
"""

from typing import Any, Dict, Optional
from .base_executor import BaseExecutor


class QiskitExecutor(BaseExecutor):
    """
    Executor for Qiskit quantum circuits.
    
    Supports local Qiskit Aer simulators only.
    Supports custom noise models based on resource parameters.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize Qiskit executor.
        
        Args:
            name: Executor name
            config: Configuration dictionary with backend settings
        """
        super().__init__(name, config)
        self.backend_name = config.get('backend', 'qasm_simulator') if config else 'qasm_simulator'
        self.shots = config.get('shots', 1000) if config else 1000
        
        # Custom backend parameters for noise modeling
        self.error_rate = config.get('error_rate', None)
        self.noise_level = config.get('noise_level', None)
        self.qubit_count = config.get('qubit_count', None)
        
        self._backend = None
        self._is_custom_backend = False
    
    def execute_circuit(self, circuit, *args, **kwargs):
        """
        Execute a Qiskit circuit.
        
        Args:
            circuit: Qiskit QuantumCircuit or circuit function
            *args: Additional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Circuit execution result
        """
        try:
            # Try to import Aer from qiskit_aer first (newer versions)
            try:
                from qiskit_aer import Aer
            except ImportError:
                # Fall back to importing from qiskit (older versions)
                from qiskit import Aer
            
            # Get the circuit
            if callable(circuit):
                qiskit_circuit = circuit(*args, **kwargs)
            else:
                qiskit_circuit = circuit
            
            # Check if we have custom backend parameters
            if self._has_custom_parameters():
                # Use custom backend with noise model
                return self._execute_with_custom_backend(qiskit_circuit)
            else:
                # Use standard backend approach
                return self._execute_with_standard_backend(qiskit_circuit)
            
        except Exception as e:
            raise Exception(f"Qiskit execution failed: {str(e)}")
    
    def _has_custom_parameters(self) -> bool:
        """Check if we have custom backend parameters for noise modeling."""
        return (self.error_rate is not None or 
                self.noise_level is not None or 
                self.qubit_count is not None)
    
    def _execute_with_custom_backend(self, qiskit_circuit):
        """Execute circuit using custom backend with noise model."""
        try:
            from qiskit_aer import AerSimulator
            from qiskit_aer.noise import NoiseModel, depolarizing_error
            
            # Create noise model based on custom parameters
            noise_model = self._create_noise_model()
            
            # Create custom backend with noise
            backend_options = {
                'method': 'density_matrix',  # Use density matrix method for noise simulation
                'noise_model': noise_model,
                'shots': self.shots
            }
            
            # Add qubit count if specified
            if self.qubit_count is not None:
                backend_options['n_qubits'] = self.qubit_count
            
            custom_backend = AerSimulator(**backend_options)
            
            # Execute circuit using the backend's run method (newer Qiskit API)
            job = custom_backend.run(qiskit_circuit, shots=self.shots)
            result = job.result()
            
            return result
            
        except Exception as e:
            raise Exception(f"Custom backend execution failed: {str(e)}")
    
    def _execute_with_standard_backend(self, qiskit_circuit):
        """Execute circuit using standard backend approach."""
        try:
            # Try to import execute from qiskit.primitives first (newer versions)
            try:
                from qiskit.primitives import Sampler
                use_sampler = True
            except ImportError:
                use_sampler = False
            
            # Try to import execute from qiskit (older versions)
            if not use_sampler:
                try:
                    from qiskit import execute
                    use_execute = True
                except ImportError:
                    use_execute = False
            
            # Execute circuit using appropriate method
            if use_sampler:
                # Use new Sampler API (Qiskit 1.0+)
                sampler = Sampler()
                # Sampler expects a sequence of circuits, so wrap in a list
                job = sampler.run([qiskit_circuit], shots=self.shots)
                result = job.result()
            elif use_execute:
                # Use old execute API (pre-Qiskit 1.0)
                backend = self._get_backend()
                job = execute(qiskit_circuit, backend, shots=self.shots)
                result = job.result()
            else:
                # Fallback: use backend's run method directly
                backend = self._get_backend()
                job = backend.run(qiskit_circuit, shots=self.shots)
                result = job.result()
            
            return result
            
        except Exception as e:
            raise Exception(f"Standard backend execution failed: {str(e)}")
    
    def _create_noise_model(self):
        """Create a noise model based on custom parameters."""
        try:
            from qiskit_aer.noise import NoiseModel, depolarizing_error
            
            noise_model = NoiseModel()
            
            # Add depolarizing error based on error_rate and noise_level
            if self.error_rate is not None and self.error_rate > 0:
                # Convert error rate to depolarizing error
                # For single qubit gates
                single_qubit_error = depolarizing_error(self.error_rate, 1)
                noise_model.add_all_qubit_quantum_error(single_qubit_error, ['h', 'x', 'z'])
                
                # For two qubit gates (higher error rate)
                two_qubit_error_rate = self.error_rate * 2  # Typically 2-qubit gates have higher error
                two_qubit_error = depolarizing_error(two_qubit_error_rate, 2)
                noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx'])
            
            # Add additional noise based on noise_level
            if self.noise_level is not None and self.noise_level > 0:
                # Convert noise_level to additional depolarizing error
                additional_error_rate = self.noise_level * 0.1  # Scale noise_level to reasonable error rate
                
                # For single qubit gates
                single_qubit_noise = depolarizing_error(additional_error_rate, 1)
                noise_model.add_all_qubit_quantum_error(single_qubit_noise, ['h', 'x', 'z'])
                
                # For two qubit gates
                two_qubit_noise = depolarizing_error(additional_error_rate * 2, 2)
                noise_model.add_all_qubit_quantum_error(two_qubit_noise, ['cx'])
            
            return noise_model
            
        except Exception as e:
            raise Exception(f"Failed to create noise model: {str(e)}")
    
    def get_available_resources(self) -> Dict[str, Any]:
        """
        Get information about the Qiskit backend.
        
        Returns:
            Dictionary containing backend information
        """
        try:
            if self._has_custom_parameters():
                return {
                    'name': f"custom_{self.backend_name}",
                    'configuration': {
                        'error_rate': self.error_rate,
                        'noise_level': self.noise_level,
                        'qubit_count': self.qubit_count,
                        'shots': self.shots
                    },
                    'properties': None
                }
            else:
                backend = self._get_backend()
                return {
                    'name': backend.name(),
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
        Check if Qiskit is available.
        
        Returns:
            True if Qiskit is available, False otherwise
        """
        try:
            import qiskit
            return True
        except ImportError:
            return False
    
    def _get_backend(self):
        """
        Get the Qiskit backend.
        
        Returns:
            Qiskit backend instance
        """
        if self._backend is None:
            # Try to import Aer from qiskit_aer first (newer versions)
            try:
                from qiskit_aer import Aer
            except ImportError:
                # Fall back to importing from qiskit (older versions)
                from qiskit import Aer
            
            self._backend = Aer.get_backend(self.backend_name)
        return self._backend
    
    def get_queue_lengths(self) -> Dict[str, float]:
        """
        Get current queue lengths for Qiskit backends.
        
        For local simulators, queue length is typically 0.0 (immediate execution).
        For real hardware, this would query the actual backend queue status.
        
        Returns:
            Dict mapping backend names to queue utilization (0.0 to 1.0)
        """
        # For local simulators, queue is always 0.0 (immediate execution)
        # For real hardware, this would query IBMQ backend status
        return {self.backend_name: 0.0}
    
    def get_backend_status(self, backend_name: str) -> Dict[str, Any]:
        """
        Get detailed status information for a Qiskit backend.
        
        Args:
            backend_name: Name of the backend
            
        Returns:
            Dict containing status information
        """
        if backend_name == self.backend_name:
            return {
                "name": backend_name,
                "queue_length": 0.0,  # Local simulators have no queue
                "status": "available",
                "type": "simulator" if "simulator" in backend_name.lower() else "hardware"
            }
        else:
            return {
                "name": backend_name,
                "queue_length": 0.0,
                "status": "unknown"
            }
    
    def is_simulator(self) -> bool:
        """
        Check if this executor uses simulators.
        
        Returns:
            True - QiskitExecutor is for local simulators only
        """
        return True
