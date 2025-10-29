"""
Pennylane Quantum Executor

Handles execution of quantum circuits using Pennylane devices.
"""

import logging
from typing import Any, Dict, Optional
from .base_executor import BaseExecutor


class PennylaneExecutor(BaseExecutor):
    """
    Executor for Pennylane quantum circuits.
    
    Supports various Pennylane devices including default.qubit, Lightning, and plugin devices.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize Pennylane executor.
        
        Args:
            name: Executor name
            config: Configuration dictionary with device settings
        """
        super().__init__(name, config)
        self.device_name = config.get('device', 'default.qubit') if config else 'default.qubit'
        self.wires = config.get('wires', 2) if config else 2
        self.shots = config.get('shots', None) if config else None
        self._device = None
        self.logger = logging.getLogger('pilot.pcs_logger')
    
    def execute_circuit(self, circuit, *args, **kwargs):
        """
        Execute a Pennylane circuit.
        
        Args:
            circuit: Pennylane circuit function (raw function or QNode)
            *args: Additional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Circuit execution result
        """
        try:
            import pennylane as qml
            
            # Check if circuit is already a QNode (bound to a device)
            if hasattr(circuit, 'device') and hasattr(circuit, 'func'):
                # Circuit is already a QNode - just execute it
                self.logger.info(f"Executing existing QNode with device: {circuit.device.name}")
                result = circuit(*args, **kwargs)
            else:
                # Circuit is a raw function - create QNode with executor's device
                self.logger.info(f"Creating QNode with executor device: {self.device_name}")
                device = self._get_device()
                qnode = qml.QNode(circuit, device)
                result = qnode(*args, **kwargs)
            
            return result
            
        except Exception as e:
            raise Exception(f"Pennylane execution failed: {str(e)}")
    
    def get_backend_info(self) -> Dict[str, Any]:
        """
        Get information about the Pennylane device.
        
        Returns:
            Dictionary containing device information
        """
        try:
            device = self._get_device()
            return {
                'name': device.name,
                'wires': device.num_wires,
                'shots': device.shots,
                'short_name': device.short_name,
                'device_name': self.device_name
            }
        except Exception:
            return {
                'name': self.device_name,
                'wires': self.wires,
                'shots': self.shots,
                'short_name': self.device_name,
                'device_name': self.device_name
            }
    
    def is_available(self) -> bool:
        """
        Check if Pennylane is available.
        
        Returns:
            True if Pennylane is available, False otherwise
        """
        try:
            import pennylane as qml
            return True
        except ImportError:
            return False
    
    def get_available_resources(self) -> Dict[str, Any]:
        """
        Get available Pennylane resources.
        
        Returns:
            Dictionary containing available device information
        """
        return {
            'device_name': self.device_name,
            'wires': self.wires,
            'shots': self.shots,
            'capabilities': {
                'supports_backprop': True,
                'supports_adjoint': True,
                'supports_vjp': True,
                'supports_jvp': True
            }
        }
    
    def _get_device(self):
        """
        Get the Pennylane device based on executor configuration.
        
        Returns:
            Pennylane device instance
        """
        if self._device is None:
            import pennylane as qml
            
            # Create device based on executor's configuration
            device_kwargs = {'wires': self.wires}
            if self.shots is not None:
                device_kwargs['shots'] = self.shots
            
            self._device = qml.device(self.device_name, **device_kwargs)
            self.logger.info(f"Created Pennylane device: {self.device_name} with {self.wires} wires")
        
        return self._device
