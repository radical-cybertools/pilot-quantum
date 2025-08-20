from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2
# from qiskit_braket_provider import BraketLocalBackend
# from qiskit_ionq import IonQProvider
import json
import logging
import random
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class QuantumResource:
    def __init__(self, name, qubit_count, gateset, error_rate=None, noise_level=None, pending_jobs=0, quantum_config=None):
        """
        Initializes a QuantumResource instance.

        :param name: Backend name
        :param qubit_count: Number of qubits available
        :param gateset: List of supported gates
        :param error_rate: Estimated error rate (default: None)
        :param noise_level: Noise level of the backend (default: None)
        :param pending_jobs: Number of pending jobs in the queue (default: 0)
        :param quantum_config: Original quantum configuration from pilot description
        """
        self.name = name
        self.qubit_count = qubit_count
        self.gateset = gateset or []
        self.error_rate = error_rate if error_rate is not None else float("inf")
        self.noise_level = noise_level if noise_level is not None else float("inf")
        self.available_qubits = qubit_count
        self.quantum_config = quantum_config or {}

    def to_dict(self):
        """Returns a dictionary representation of the QuantumResource."""
        return {
            "name": self.name,
            "qubit_count": self.qubit_count,
            "gateset": self.gateset,
            "error_rate": self.error_rate,
            "noise_level": self.noise_level,
            "quantum_config": self.quantum_config
        }

    def __repr__(self):
        """Returns a detailed string representation (useful for debugging)."""
        return f"QuantumResource(name={self.name}, qubit_count={self.qubit_count}, gateset={self.gateset}, " \
               f"error_rate={self.error_rate}, noise_level={self.noise_level}, quantum_config={self.quantum_config})"

    def __str__(self):
        """Returns a human-readable string representation."""
        return f"{self.name}: {self.qubit_count} qubits, Gateset: {self.gateset}, " \
               f"Error Rate: {self.error_rate:.4f}, Noise Level: {self.noise_level:.4f}, " \
               f"Config: {self.quantum_config}"
    
    @property
    def fidelity(self) -> float:
        """Calculate fidelity as 1 - error_rate."""
        return 1.0 - self.error_rate if self.error_rate is not None and self.error_rate != float("inf") else 1.0
               
               
class QuantumResourceEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, QuantumResource):
            return obj.to_dict()  # Convert QuantumResource to a dictionary
        return super().default(obj)

# Create a class that generates quantum resources backends.
class QuantumResourceGenerator:
    def __init__(self):
        self.quantum_resources = self._get_quantum_resources()

    def _get_fake_backends(self):
        backends = FakeProviderForBackendV2().backends()
        # braket_backends = [BraketLocalBackend("default")]
        # return fake_backends + braket_backends
        return backends 
    
    def get_quantum_resources(self, executor, config=None):
        """
        Get quantum resources for a specific executor.
        
        Args:
            executor: Executor name (e.g., 'qiskit_local', 'ibmq', 'pennylane')
            config: Configuration dictionary containing device/backend settings
            
        Returns:
            Dictionary of quantum resources for the executor
        """
        if executor == 'qiskit_local' or executor == 'qiskit':
            return self._get_qiskit_local_resources(config)
        elif executor.startswith('ibmq_'):
            return self._get_ibmq_resources(executor, config)
        elif executor.startswith('braket_'):
            return self._get_braket_resources(executor, config)
        elif executor == 'pennylane':
            return self._get_pennylane_resources(config)
        else:
            # Fallback to fake backends
            return self._get_fake_backend_resources()
    
    def get_quantum_resources_for_executor(self, executor, config=None):
        """
        Alias for get_quantum_resources to maintain backward compatibility.
        
        Args:
            executor: Executor name (e.g., 'qiskit_local', 'ibmq', 'pennylane')
            config: Configuration dictionary containing device/backend settings
            
        Returns:
            Dictionary of quantum resources for the executor
        """
        return self.get_quantum_resources(executor, config)

    def get_filtered_quantum_resources(self, fidelity=None, qubit_count=None):
        """
        Returns a list of quantum resources that satisfy the given fidelity and qubit_count requirements.
        Fidelity is interpreted as the maximum allowed fidelity (i.e., resource fidelity <= fidelity).
        qubit_count is interpreted as the minimum required number of qubits.
        If no filters are provided, returns all quantum resources as a list.
        """
        filtered_resources = []
        for qr in self.quantum_resources.values():
            # Fidelity filter: resource fidelity <= given fidelity
            # Fidelity is defined as (1 - error_rate)
            if fidelity is not None:
                if qr.error_rate is None:
                    continue
                resource_fidelity = 1 - qr.error_rate
                if resource_fidelity > fidelity:
                    continue
            # Qubit count filter
            if qubit_count is not None:
                if qr.qubit_count < qubit_count:
                    continue
            filtered_resources.append(qr)
        return filtered_resources
    
    def _get_quantum_resources(self):
        q_resources = self._get_fake_backends()
        q_resources_map = {}
        for q_resource in q_resources:
            config = QuantumResource(
                name=q_resource.name,
                qubit_count=self.get_qubit_count(q_resource),
                gateset=self.get_gateset_availability(q_resource),
                error_rate=self.get_error_rate(q_resource),
                noise_level=self.get_noise_level(q_resource),
            )
            q_resources_map[q_resource.name] = config
        return q_resources_map
    
    def _get_qiskit_local_resources(self, config=None):
        """Get Qiskit local simulator resources."""
        resources = {}
        
        # Check if custom backends are provided in config
        if config and 'custom_backends' in config:
            print(f"🔧 Creating custom quantum resources from configuration...")
            for backend_name, backend_config in config['custom_backends'].items():
                resources[backend_name] = QuantumResource(
                    name=backend_name,
                    qubit_count=backend_config['qubit_count'],
                    gateset=['h', 'cx', 'x', 'z', 'measure'],  # Default gateset
                    error_rate=backend_config['error_rate'],
                    noise_level=backend_config['noise_level'],
                    quantum_config=config
                )
                fidelity = 1 - backend_config['error_rate']
                print(f"   ✅ {backend_name}: {backend_config['qubit_count']} qubits, "
                      f"{fidelity:.1%} fidelity, queue_length={backend_config.get('queue_length', 0)}")
            return resources
        
        # Check if specific backends are requested
        backend_names = config.get('backend') if config else None
        
        # Default Qiskit local resources
        all_resources = {
            'qiskit_aer_simulator': QuantumResource(
                name='qiskit_aer_simulator',
                qubit_count=32,
                gateset=['h', 'cx', 'x', 'z', 'measure', 'u1', 'u2', 'u3'],
                error_rate=0.0,  # Perfect simulator
                noise_level=0.0,
                quantum_config=config or {}
            ),
            'qiskit_basicaer_simulator': QuantumResource(
                name='qiskit_basicaer_simulator',
                qubit_count=24,
                gateset=['h', 'cx', 'x', 'z', 'measure'],
                error_rate=0.0,
                noise_level=0.0,
                quantum_config=config or {}
            )
        }
        
        # Filter based on requested backends
        if backend_names:
            for backend_name in backend_names:
                for resource_name, resource in all_resources.items():
                    if resource_name == backend_name or backend_name in resource_name:
                        resources[resource_name] = resource
            
            # Log which backends were found
            if resources:
                found_backends = list(resources.keys())
                print(f"✅ Found requested Qiskit backends: {found_backends}")
                if len(found_backends) < len(backend_names):
                    missing = [b for b in backend_names if not any(b in found for found in found_backends)]
                    print(f"⚠️  Warning: Some requested Qiskit backends not found: {missing}")
            else:
                print(f"⚠️  Warning: No requested Qiskit backends {backend_names} found. Available: {list(all_resources.keys())}")
                # Fall back to all available resources
                resources = all_resources
        else:
            # Use all available resources
            resources = all_resources
        
        return resources
    
    def _get_ibmq_resources(self, executor, config):
        """Get IBM Quantum resources."""
        resources = {}
        
        # Check if backends are requested in config
        backend_names = config.get('backend') if config else None
        
        try:
            # Try to import IBM Quantum provider
            from qiskit_ibm_runtime import QiskitRuntimeService
            
            # Get token from config or environment
            token = config.get('ibmqx_token') if config else None
            if not token:
                token = os.environ.get('IBMQX_TOKEN')
            
            if token:
                # Real IBM Quantum backends
                service = QiskitRuntimeService(token=token)
                backends = service.backends()
                
                for backend in backends:
                    # Check if this backend matches any of the requested backends
                    backend_matches = False
                    if backend_names:
                        for backend_name in backend_names:
                            if backend.name == backend_name or backend_name in backend.name:
                                backend_matches = True
                                break
                    else:
                        # If no specific backends requested, match by executor name
                        if backend.name == executor or executor in backend.name:
                            backend_matches = True
                    
                    if backend_matches:
                        config = backend.configuration()
                        properties = backend.properties() if hasattr(backend, 'properties') else None
                        
                        # Extract gate set
                        gateset = getattr(config, 'basis_gates', ['h', 'cx', 'x', 'z', 'measure'])
                        
                        # Extract error rates
                        error_rate = 0.001  # Default
                        noise_level = 0.002  # Default
                        
                        if properties:
                            # Calculate average error rate from properties
                            gate_errors = []
                            for gate in properties.gates:
                                for param in gate.parameters:
                                    if param.name == 'gate_error':
                                        gate_errors.append(param.value)
                            
                            if gate_errors:
                                error_rate = sum(gate_errors) / len(gate_errors)
                        
                        resources[backend.name] = QuantumResource(
                            name=backend.name,
                            qubit_count=config.n_qubits,
                            gateset=gateset,
                            error_rate=error_rate,
                            noise_level=noise_level
                        )
                
                if not resources:
                    # Create a generic IBMQ resource if specific backend not found
                    resources[executor] = QuantumResource(
                        name=executor,
                        qubit_count=5,
                        gateset=['h', 'cx', 'x', 'z', 'measure'],
                        error_rate=0.001,
                        noise_level=0.002
                    )
            else:
                # No token provided - use fake backends
                if backend_names:
                    # If specific backends requested, try to find them in fake backends
                    fake_backends = self._get_fake_backends()
                    for backend in fake_backends:
                        # Check if this backend matches any of the requested backends
                        for backend_name in backend_names:
                            if backend.name == backend_name or backend_name in backend.name:
                                resources[backend.name] = QuantumResource(
                                    name=backend.name,
                                    qubit_count=self.get_qubit_count(backend),
                                    gateset=self.get_gateset_availability(backend),
                                    error_rate=self.get_error_rate(backend),
                                    noise_level=self.get_noise_level(backend)
                                )
                                break
                    
                    # If specific backends not found, log and create generic ones
                    if not resources:
                        print(f"⚠️  Warning: Requested backends {backend_names} not found in fake backends. Creating generic resources.")
                        for backend_name in backend_names:
                            resources[backend_name] = QuantumResource(
                                name=backend_name,
                                qubit_count=5,
                                gateset=['h', 'cx', 'x', 'z', 'measure'],
                                error_rate=0.001,
                                noise_level=0.002
                            )
                    else:
                        # Log which backends were found
                        found_backends = list(resources.keys())
                        print(f"✅ Found requested backends: {found_backends}")
                        if len(found_backends) < len(backend_names):
                            missing = [b for b in backend_names if not any(b in found for found in found_backends)]
                            print(f"⚠️  Warning: Some requested backends not found: {missing}")
                else:
                    # No specific backends requested - use all fake backends
                    fake_backends = self._get_fake_backends()
                    for backend in fake_backends:
                        resources[backend.name] = QuantumResource(
                            name=backend.name,
                            qubit_count=self.get_qubit_count(backend),
                            gateset=self.get_gateset_availability(backend),
                            error_rate=self.get_error_rate(backend),
                            noise_level=self.get_noise_level(backend)
                        )
                
        except ImportError:
            # IBMQ provider not available, use fake backends
            if backend_names:
                # Try to find specific backends in fake backends
                fake_backends = self._get_fake_backends()
                for backend in fake_backends:
                    # Check if this backend matches any of the requested backends
                    for backend_name in backend_names:
                        if backend.name == backend_name or backend_name in backend.name:
                            resources[backend.name] = QuantumResource(
                                name=backend.name,
                                qubit_count=self.get_qubit_count(backend),
                                gateset=self.get_gateset_availability(backend),
                                error_rate=self.get_error_rate(backend),
                                noise_level=self.get_noise_level(backend)
                            )
                            break
                
                # If specific backends not found, log and create generic ones
                if not resources:
                    print(f"⚠️  Warning: Requested backends {backend_names} not found in fake backends. Creating generic resources.")
                    for backend_name in backend_names:
                        resources[backend_name] = QuantumResource(
                            name=backend_name,
                            qubit_count=5,
                            gateset=['h', 'cx', 'x', 'z', 'measure'],
                            error_rate=0.001,
                            noise_level=0.002
                        )
                else:
                    # Log which backends were found
                    found_backends = list(resources.keys())
                    print(f"✅ Found requested backends: {found_backends}")
                    if len(found_backends) < len(backend_names):
                        missing = [b for b in backend_names if not any(b in found for found in found_backends)]
                        print(f"⚠️  Warning: Some requested backends not found: {missing}")
            else:
                # Use all fake backends
                fake_backends = self._get_fake_backends()
                for backend in fake_backends:
                    resources[backend.name] = QuantumResource(
                        name=backend.name,
                        qubit_count=self.get_qubit_count(backend),
                        gateset=self.get_gateset_availability(backend),
                        error_rate=self.get_error_rate(backend),
                        noise_level=self.get_noise_level(backend)
                    )
        
        return resources
    
    def _get_braket_resources(self, executor, config):
        """Get AWS Braket resources."""
        resources = {}
        
        try:
            # Try to import Braket provider
            from braket.aws import AwsDevice
            
            # Get device name from config or environment
            device_name = config.get('braket_device_arn') if config else None
            if not device_name:
                device_name = os.environ.get('BRAKET_DEVICE_ARN')
            
            if device_name:
                device = AwsDevice(device_name)
                
                # Extract gate set
                gateset = device.properties.basis_gates
                
                # Extract error rates
                error_rate = 0.001  # Default
                noise_level = 0.002  # Default
                
                # Braket properties are not directly available as a single error rate or noise level.
                # We'll use a placeholder or a default value.
                # For simplicity, we'll set a default value.
                
                resources[device_name] = QuantumResource(
                    name=device_name,
                    qubit_count=device.properties.n_qubits,
                    gateset=gateset,
                    error_rate=error_rate,
                    noise_level=noise_level
                )
            else:
                # No device ARN provided, create a generic resource
                resources[executor] = QuantumResource(
                    name=executor,
                    qubit_count=5,
                    gateset=['h', 'cx', 'x', 'z', 'measure'],
                    error_rate=0.001,
                    noise_level=0.002
                )
                
        except ImportError:
            # Braket provider not available, create generic resource
            resources[executor] = QuantumResource(
                name=executor,
                qubit_count=5,
                gateset=['h', 'cx', 'x', 'z', 'measure'],
                error_rate=0.001,
                noise_level=0.002
            )
        
        return resources
    
    def _get_pennylane_resources(self, config=None):
        """Get Pennylane resources based on device configuration."""
        resources = {}
        
        # Read device information from config
        devices = []
        
        # Check for multiple devices in config
        if config and 'devices' in config:
            devices = config['devices']
        elif config and 'quantum' in config and 'devices' in config['quantum']:
            devices = config['quantum']['devices']
        # Fallback to single device for backward compatibility
        elif config and 'device' in config:
            devices = [config['device']]
        elif config and 'quantum' in config and 'device' in config['quantum']:
            devices = [config['quantum']['device']]
        
        # If no devices specified, use default
        if not devices:
            devices = ['default.qubit']
        
        # Create resources for each device
        for device_name in devices:
            if device_name == 'default.qubit':
                resources['pennylane_default_qubit'] = QuantumResource(
                    name='pennylane_default_qubit',
                    qubit_count=30,
                    gateset=['h', 'cnot', 'x', 'z', 'measure', 'rx', 'ry', 'rz'],
                    error_rate=0.0,
                    noise_level=0.0,
                    quantum_config=config or {}
                )
            elif device_name == 'qiskit.aer':
                resources['pennylane_qiskit'] = QuantumResource(
                    name='pennylane_qiskit',
                    qubit_count=32,
                    gateset=['h', 'cx', 'x', 'z', 'measure', 'u1', 'u2', 'u3'],
                    error_rate=0.0,
                    noise_level=0.0,
                    quantum_config=config or {}
                )
            elif device_name == 'lightning.qubit':
                resources['pennylane_lightning'] = QuantumResource(
                    name='pennylane_lightning',
                    qubit_count=30,
                    gateset=['h', 'cnot', 'x', 'z', 'measure', 'rx', 'ry', 'rz'],
                    error_rate=0.0,
                    noise_level=0.0,
                    quantum_config=config or {}
                )
            elif device_name == 'braket.aws.qubit':
                resources['pennylane_braket'] = QuantumResource(
                    name='pennylane_braket',
                    qubit_count=25,
                    gateset=['h', 'cnot', 'x', 'z', 'measure', 'rx', 'ry', 'rz'],
                    error_rate=0.001,
                    noise_level=0.002,
                    quantum_config=config or {}
                )
            else:
                # Unknown device - create a generic resource
                device_key = f"pennylane_{device_name.replace('.', '_')}"
                resources[device_key] = QuantumResource(
                    name=device_key,
                    qubit_count=30,
                    gateset=['h', 'cnot', 'x', 'z', 'measure', 'rx', 'ry', 'rz'],
                    error_rate=0.0,
                    noise_level=0.0,
                    quantum_config=config or {}
                )
        
        return resources
    
    def _get_fake_backend_resources(self):
        """Get fake backend resources as fallback."""
        return self.quantum_resources
    
    @staticmethod
    def get_error_rate(backend):
        return QuantumResourceGenerator.get_backend_property(backend, "gate_error", random.uniform(0.001, 0.01))

    @staticmethod
    def get_noise_level(backend):
        return QuantumResourceGenerator.get_backend_property(backend, "readout_error", random.uniform(0.001, 0.01))

    @staticmethod
    def get_gateset_availability(backend):
        return QuantumResourceGenerator.get_backend_configuration(backend, "basis_gates", None)

    @staticmethod
    def get_qubit_count(backend):
        return QuantumResourceGenerator.get_backend_configuration(backend, "n_qubits", None)

    @staticmethod
    def get_backend_property(backend, property_name, default):
        if hasattr(backend, "properties") and backend.properties():
            properties = backend.properties()
            values = [param.value for gate in properties.gates for param in gate.parameters if param.name == property_name]
            return sum(values) / len(values) if values else default
        return default

    @staticmethod
    def get_backend_configuration(backend, config_name, default):
        if hasattr(backend, "configuration") and backend.configuration():
            config = backend.configuration()
            return getattr(config, config_name, default)
        return default    
    
    def save_quantum_resources(self, filename="quantum_resources.json"):
        with open(filename, "w") as f:
            json.dump(self.quantum_resources, f, cls=QuantumResourceEncoder, indent=4)
        logging.info(f"Backend configurations saved to {filename}")                     
    
if __name__ == "__main__":
    # Generate the quantum resources.
    q_resource_generator = QuantumResourceGenerator()
    q_resource_generator.save_quantum_resources()
    