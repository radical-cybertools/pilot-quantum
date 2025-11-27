"""
AWS Braket Quantum Executor

Handles execution of quantum circuits using AWS Braket devices.
"""

from typing import Any, Dict, Optional
from .base_executor import BaseExecutor


class BraketExecutor(BaseExecutor):
    """
    Executor for AWS Braket quantum circuits.
    
    Supports various Braket devices including simulators and quantum hardware.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize Braket executor.
        
        Args:
            name: Executor name
            config: Configuration dictionary with device settings
        """
        super().__init__(name, config)
        self.device_arn = config.get('device_arn') if config else None
        self.shots = config.get('shots', 1000) if config else 1000
        self._device = None
    
    def execute_circuit(self, circuit, *args, **kwargs):
        """
        Execute a Braket circuit.
        
        Args:
            circuit: Braket Circuit or circuit function
            *args: Additional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Circuit execution result
        """
        try:
            from braket.aws import AwsDevice
            
            # Get the circuit
            if callable(circuit):
                braket_circuit = circuit(*args, **kwargs)
            else:
                braket_circuit = circuit
            
            # Get device
            device = self._get_device()
            
            # Execute circuit
            task = device.run(braket_circuit, shots=self.shots)
            result = task.result()
            
            return result
            
        except Exception as e:
            raise Exception(f"Braket execution failed: {str(e)}")
    
    def get_backend_info(self) -> Dict[str, Any]:
        """
        Get information about the Braket device.
        
        Returns:
            Dictionary containing device information
        """
        try:
            device = self._get_device()
            return {
                'name': device.name,
                'arn': device.arn,
                'type': device.type,
                'provider_name': device.provider_name,
                'properties': device.properties.dict() if device.properties else None
            }
        except Exception:
            return {
                'name': 'braket_device',
                'arn': self.device_arn,
                'type': 'unknown',
                'provider_name': 'unknown',
                'properties': None
            }
    
    def is_available(self) -> bool:
        """
        Check if AWS Braket is available.
        
        Returns:
            True if Braket is available, False otherwise
        """
        try:
            from braket.aws import AwsDevice
            return True
        except ImportError:
            return False
    
    def is_simulator(self) -> bool:
        """
        Check if this executor uses simulators.
        
        Returns:
            True if using simulators, False for real hardware
        """
        # Check if backend name contains 'simulator' or if no token (fake backends)
        return ("simulator" in self.backend_name.lower() or 
                "fake" in self.backend_name.lower() or 
                self.token is None)
    
    def _get_device(self):
        """
        Get the Braket device.
        
        Returns:
            Braket device instance
        """
        if self._device is None:
            if not self.device_arn:
                raise Exception("Device ARN not provided for Braket executor")
            
            from braket.aws import AwsDevice
            self._device = AwsDevice(self.device_arn)
        
        return self._device
