Sample Bashrc
```
cat ~/.bashrc
#module load PrgEnv-gnu cray-mpich cudatoolkit/11.7 craype-accel-nvidia80 evp-patch gcc/11.2.0 python
module load PrgEnv-gnu/8.3.3 cray-mpich/8.1.25 cudatoolkit/11.7 craype-accel-nvidia80 evp-patch gcc/11.2.0 cmake/3.22.0
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:/opt/cray/pe/mpich/8.1.25/ofi/gnu/9.1/lib/:$LD_LIBRARY_PATH
module load python
conda activate /pscratch/sd/p/prmantha/py3117
export CUQUANTUM_SDK=/pscratch/sd/p/prmantha/py3117/lib/python3.11/site-packages/cuquantum
export LD_LIBRARY_PATH=/pscratch/sd/p/prmantha/py3117/lib/python3.11/site-packages/cuquantum/lib:$LD_LIBRARY_PATH

```

# Ensure all necessary modules are loaded up-front

module load PrgEnv-gnu/8.3.3 cray-mpich/8.1.25 cudatoolkit/11.7 craype-accel-nvidia80 evp-patch gcc/11.2.0 cmake/3.22.0

# Required due to a potentially missing lib for the CUDA-aware MPICH library

export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:/opt/cray/pe/mpich/8.1.25/ofi/gnu/9.1/lib/:$LD_LIBRARY_PATH

# Create a python virtualenv

conda create --prefix /pscratch/sd/p/<username>/py3117 python=3.11.7

# install the following dependencies

python -m pip install cmake ninja custatevec-cu11 cuquantum-python-cu11 wheel pennylane~=0.34.0 pennylane-lightning~=0.34.0

# build mpi4py against the system's CUDA-aware MPICH

MPICC="cc -shared" python -m pip install --force --no-cache-dir --no-binary=mpi4py mpi4py

# clone and checkout lightning.gpu at version 0.34.0
git clone https://github.com/PennyLaneAI/pennylane-lightning.git

cd pennylane-lightning && git checkout v0.34.0

pip install -r requirements.txt

PL_BACKEND="lightning_qubit" pip install -e . -vv

export CUQUANTUM_SDK=/pscratch/sd/p/<Username>/py3117/lib/python3.11/site-packages/cuquantum

export LD_LIBRARY_PATH=/pscratch/sd/p/<Username>/py3117/lib/python3.11/site-packages/cuquantum/lib:$LD_LIBRARY_PATH

PL_BACKEND="lightning_gpu" python -m pip install -e .

CMAKE_ARGS="-DENABLE_MPI=ON"  PL_BACKEND="lightning_gpu" python -m pip install -e .


# Grab an allocation for however many GPUs you want; I'd start with a single interactive node/4 GPUs

salloc -N 1 --qos interactive  --time 02:00:00 --gpus=4 --constraint gpu --account=m4408

# and launch over multiple processes

srun -n 4 python script.py

