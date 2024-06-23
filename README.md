# Pilot-Quantum

Last Updated: 03/03/2024

# Overview:

Pilot-Quantum is presented as a Quantum-HPC middleware framework designed to address the challenges of integrating quantum and classical computing resources. It focuses on managing heterogeneous resources, including diverse Quantum Processing Unit (QPU) modalities and various integration types with classical resources, such as accelerators.
 
Requirements:

	* Currently only SLURM clusters are supported
	* Setup password-less documentation, e.g., using sshproxy on Perlmutter.

Anaconda or Miniconda is the preferred distribution


## Installation
Requirement (in case a manual installation is required):

The best way to utilize Pilot-Quantum is Anaconda, which provides an easy way to install

    pip install -r requirements.txt

To install Pilot-Quantum type:

    python setup.py install


## Hints

Your default conda environment should contain all Pilot-Quantum and application dependencies. Activate it, e.g., in the `.bashrc`.oxy
