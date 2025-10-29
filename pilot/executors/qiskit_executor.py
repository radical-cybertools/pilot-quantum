"""
Qiskit Quantum Executor

Minimal implementation for executing quantum circuits using Qiskit SamplerV2.
"""

from typing import Any, Dict, List, Union
from .base_executor import BaseExecutor


class QiskitExecutor(BaseExecutor):
    """
    Minimal executor for Qiskit quantum circuits using SamplerV2.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize Qiskit executor."""
        super().__init__(name, config)
        self.shots = config.get('shots', 1000) if config else 1000
        self.backend_options = config.get('backend_options', {
            'shots': self.shots,
            'device': 'CPU',
            'method': 'statevector'
        })
        print(f"[QiskitExecutor] Initialized with shots={self.shots}, backend_options={self.backend_options}")
    
    def execute_circuit(self, circuits: List, *args, **kwargs):
        """
        Execute list of quantum circuits using SamplerV2.
        
        Args:
            circuits: List of quantum circuits
            *args: Additional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            SamplerV2 execution result
        """
        try:
            from qiskit_aer import AerSimulator
            from qiskit_ibm_runtime import SamplerV2, Batch
            import time
            
            print(f"[QiskitExecutor] Executing {len(circuits)} circuits")
            
            # Create AerSimulator backend with configured options
            submit_start = time.time()
            backend = AerSimulator(**self.backend_options)
            print(f"[QiskitExecutor] Created backend in {time.time() - submit_start:.4f}s: {backend}")
            
            # Execute using SamplerV2 with Batch
            with Batch(backend=backend) as batch:
                sampler = SamplerV2(mode=batch)
                print(f"[QiskitExecutor] Submitting job")
                job = sampler.run(circuits)
                print(f"[QiskitExecutor] Job submitted, waiting for result...")
                result = job.result()
                print(f"[QiskitExecutor] Job completed successfully with result: {result}")

            return self.get_transformed_results(result)
            
        except Exception as e:
            print(f"[QiskitExecutor] ERROR: Execution failed - {str(e)}")
            raise Exception(f"Qiskit execution failed: {str(e)}")

    def get_transformed_results(self, result):
        # Reconstruct the PrimitiveResult object to fix serialization issues with current Qiskit versions (at the time 1.3)
        # see https://github.com/Qiskit/qiskit/issues/12787
        from qiskit.primitives.containers import (
            PrimitiveResult,
            SamplerPubResult,
            DataBin,
            BitArray,
        )
        import copy
        import numpy as np

        # Override DataBin class to fix serialization issues
        class CustomDataBin(DataBin):
            def __setattr__(self, name, value):
                super().__init__()
                self.__dict__[name] = value

        # Reconstruct the PrimitiveResult object to fix serialization issues
        new_results = []
        for pub_result in result:
            # Deep copy the metadata
            new_metadata = copy.deepcopy(pub_result.metadata)

            # Access the DataBin object
            data_bin = pub_result.data

            # Reconstruct DataBin
            new_data_bin_dict = {}

            # Explicitly copy 'observable_measurements'
            if hasattr(data_bin, "observable_measurements"):
                observable_measurements = data_bin.observable_measurements
                new_observable_array = np.copy(observable_measurements.array)
                new_observable_bitarray = BitArray(
                    new_observable_array, observable_measurements.num_bits
                )
                new_data_bin_dict["observable_measurements"] = new_observable_bitarray

            # Explicitly copy 'qpd_measurements'
            if hasattr(data_bin, "qpd_measurements"):
                qpd_measurements = data_bin.qpd_measurements
                new_qpd_array = np.copy(qpd_measurements.array)
                new_qpd_bitarray = BitArray(new_qpd_array, qpd_measurements.num_bits)
                new_data_bin_dict["qpd_measurements"] = new_qpd_bitarray

            # Copy other attributes of DataBin (e.g., 'shape')
            if hasattr(data_bin, "shape"):
                new_data_bin_dict["shape"] = copy.deepcopy(data_bin.shape)

            # Create a new DataBin instance
            new_data_bin = CustomDataBin(**new_data_bin_dict)
            # new_data_bin.__setattr__ = custom_setattr

            # Create a new SamplerPubResult
            new_pub_result = SamplerPubResult(data=new_data_bin, metadata=new_metadata)
            new_results.append(new_pub_result)

        # Create a new PrimitiveResult
        new_result = PrimitiveResult(
            new_results, metadata=copy.deepcopy(result.metadata)
        )             
        return new_result        
    
    def is_simulator(self) -> bool:
        """Check if this executor uses simulators."""
        return True
    
    def get_available_resources(self) -> Dict[str, Any]:
        """Get basic resource information."""
        return {
            'name': 'qiskit_aer_simulator',
            'shots': self.shots,
            'type': 'simulator'
        }