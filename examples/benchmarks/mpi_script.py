import sys
from mpi4py import MPI

def mpi_task(args):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"Hello from rank {rank} of {size}")
    print("Command line arguments:", args)
    # Add your MPI logic here.

if __name__ == "__main__":
    args = sys.argv[1:]
    mpi_task(args)
