#!/usr/bin/env python
import logging
import os
import signal
import subprocess
import time
from optparse import OptionParser
import concurrent.futures

from pilot.plugins.pilot_agent_base import PilotAgent
from pilot.util.ssh_utils import execute_local_process, execute_ssh_command, get_localhost

STOP=False





def handle_signals():
    def handler(signum, frame):
        global STOP
        STOP=True    
        
    signal.signal(signal.SIGALRM, handler)
    signal.signal(signal.SIGABRT, handler)
    signal.signal(signal.SIGQUIT, handler)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)




class RayPilotAgent(PilotAgent):

    def __init__(self, working_directory, scheduler_file_path, worker_config_file, worker_name):
        super().__init__(working_directory, scheduler_file_path, worker_config_file, worker_name)
        self.worker_nodes = self.get_nodelist_from_resourcemanager()
                  

    def start_ray(self, node):
        start_ray_time = time.time()
        # read worker config file
        worker_config = self.get_worker_config_json()
        scheduler_address = self.get_scheduler_address()
        
        # Start Ray on the node
        # Include CUDA_VISIBLE_DEVICES if set, to ensure Ray workers see the correct GPUs
        cuda_env = ""
        cuda_visible_devices = worker_config.get("cuda_visible_devices") or os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices:
            cuda_env = f"export CUDA_VISIBLE_DEVICES={cuda_visible_devices}; "
            # Prevent Ray from overriding CUDA_VISIBLE_DEVICES when num_gpus=0
            cuda_env += "export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1; "
        # Allow explicit resource override to avoid incorrect GPU counts
        override_env = ""
        ray_override_resources = worker_config.get("ray_override_resources") or os.environ.get("RAY_OVERRIDE_RESOURCES")
        if ray_override_resources:
            override_env = f"export RAY_OVERRIDE_RESOURCES='{ray_override_resources}'; "
        
        command = f"{cuda_env}{override_env}export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1; " \
                f"ray start --address {scheduler_address} " \
                f"--num-cpus={worker_config['cores_per_node']} " \
                f"--num-gpus={worker_config['gpus_per_node']}"
        
        # Log the full command for debugging
        self.logger.info(f"[RAY_WORKER_START] Starting Ray worker on {node}")
        self.logger.info(f"[RAY_WORKER_START] CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
        self.logger.info(f"[RAY_WORKER_START] RAY_OVERRIDE_RESOURCES={ray_override_resources}")
        self.logger.info(f"[RAY_WORKER_START] num_gpus={worker_config['gpus_per_node']}")
        self.logger.info(f"[RAY_WORKER_START] Command: {command}")
            
        host_node_ip_address = get_localhost()
        if scheduler_address.startswith(host_node_ip_address):
            status = execute_local_process(command, working_directory=self.working_directory)
        else:                    
            status = execute_ssh_command(host=node, command=command, working_directory=self.working_directory)
            
        end_ray_time = time.time()
        current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_ray_time))        
        self.logger.info(f"Ray started on {node} with command {command} and status:{status};current_time:{current_time},startup_time:{end_ray_time-start_ray_time}")

    def start_workers(self):                        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for node in self.worker_nodes:
                executor.submit(self.start_ray, node)
                
    def stop_workers(self):
        for node in self.worker_nodes:
            status = execute_ssh_command(node, "ray stop")
            self.logger.info(f"Ray stopped on {node} with status {status}")
            

if __name__ == "__main__" :
    handle_signals()

    parser = OptionParser()
    parser.add_option("-s", "--start", action="store_true", dest="start",
                  help="start Ray", default=True)
    
    parser.add_option("-w", "--pilot-working-directory", type="string", action="store", dest="pilot_working_directory", default="/Users/pmantha/work/pcs-2cebd082-52f0-4cf3-af71-2a89b82dcfd3/RAY-f028ab1e-6d50-11ef-af5b-d2b91fefeab3", help="Working directory to execute agent in")

    parser.add_option("-f", "--scheduler-file", type="string", action="store", dest="scheduler_file", default="/Users/pmantha/work/pcs-2cebd082-52f0-4cf3-af71-2a89b82dcfd3/scheduler", help="Scheduler information file path")
        
    parser.add_option("-c", "--worker-config-file", type="string", action="store", dest="worker_config_file", default="/Users/pmantha/work/pcs-2cebd082-52f0-4cf3-af71-2a89b82dcfd3//worker_config.json", help="Worker config file")
    
    parser.add_option("-n", "--worker-name", type="string", action="store", dest="worker_name", default="pq-ray-worker", help="Worker name")


    
    # Parse Option from commandline arguments
    (options, args) = parser.parse_args()
  
    # Initialize object for managing Ray clusters
    ray_agent = RayPilotAgent(options.pilot_working_directory, 
                                 options.scheduler_file, 
                                 options.worker_config_file, options.worker_name)
     
    if options.start:
        ray_agent.start_workers()
    
    logging.info("Finished launching of Ray Cluster - Sleeping now")

    while not STOP:
        logging.debug("Ray Agent is running... " + str(STOP))
        time.sleep(10)
        
    logging.info("Stopping Ray Cluster")
    ray_agent.stop_workers()
            
        
    
    
    
