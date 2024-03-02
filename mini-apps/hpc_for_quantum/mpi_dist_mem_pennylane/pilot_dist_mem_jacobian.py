import radical.pilot as rp
import os, sys

curr_script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
SIM_SCRIPT = os.path.join(curr_script_dir, "./", "dist_mem_jacobian.py")

# Define the details of the Perlmutter resource
pd_init = {
"resource" : "nersc.perlmutter_gpu",
"nodes" : 2,
"runtime" : 10, 
"gpus_per_node": 4,
"project" : "m4408",
"queue": "regular"
}

print ("Submitting pilot with description", pd_init)

# Create a new session
session = rp.Session()

try:
    # Add a Pilot Manager
    pmgr = rp.PilotManager(session=session)

    # Define a ComputePilotDescription
    pdesc = rp.PilotDescription(pd_init)


    # Launch the pilot
    pilot = pmgr.submit_pilots(pdesc)

    # Register the Pilot in a UnitManager object
    tmgr = rp.TaskManager(session=session)
    tmgr.add_pilots(pilot)

    # Create a workload of Computetasks 
    n=1

    cudesc = rp.TaskDescription()
    cudesc_list=[]
    cudesc.executable = "python"  # example GPU executable
    cudesc.arguments = [SIM_SCRIPT]
    cudesc.gpu_type = "CUDA"
    cudesc.gpus_per_rank = 1
    cudesc.ranks = pd_init["nodes"] * pd_init["gpus_per_node"]
    cudesc.use_mpi = True

    print("Submitted task", cudesc)

    cudesc_list.append(cudesc)

    # Submit the ComputeUnit descriptions to the PilotManager
    tasks = tmgr.submit_tasks(cudesc_list)

    # Wait for all compute tasks to reach a final state
    tmgr.wait_tasks()

finally:
    # Close the session
    session.close()


