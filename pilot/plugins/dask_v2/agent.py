#!/usr/bin/env python
import logging
import os
import signal
import subprocess
import time
from optparse import OptionParser

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


class DaskPilotAgent(PilotAgent):

    def __init__(self, working_directory, scheduler_file_path, worker_config_file, worker_name):
        super().__init__(working_directory, scheduler_file_path, worker_config_file, worker_name)
        self.worker_nodes = self.get_nodelist_from_resourcemanager()
                  

    def start_workers(self):
        for node in self.worker_nodes:                        
            # read worker config file
            worker_config = self.get_worker_config_json()
            scheduler_address = self.get_scheduler_address()
            
            # Start Dask on the node
            command = f"dask worker {scheduler_address} " \
                      f" --resources=\'CPU={worker_config['cores_per_node']} GPU={worker_config['gpus_per_node']}\'" \
                      f" --name={self.worker_name}"                      
                    
            host_node_ip_address = get_localhost()
            if scheduler_address.startswith(host_node_ip_address):
                status = execute_local_process(command, working_directory=self.working_directory)
            else:                    
                status = execute_ssh_command(host=node, command=command, working_directory=self.working_directory)
            
            self.logger.info(f"Dask started on {node} with command {command} and status {status}")
        
    def stop_workers(self):
        for node in self.worker_nodes:
            try:
                # SSH into the node and find Dask processes
                ssh_command = f"ssh {node} 'pgrep -f \"dask (scheduler|worker|cuda)\"'"
                process = subprocess.Popen(ssh_command, shell=True, stdout=subprocess.PIPE)
                output, _ = process.communicate()

                # Kill Dask processes found on the node
                if output:
                    pids = output.decode().splitlines()
                    for pid in pids:
                        ssh_kill_command = f"ssh {node} 'kill -9 {pid}'"
                        subprocess.run(ssh_kill_command, shell=True, check=True)
                    logging.debug(f"Dask processes killed on node {node}")
                else:
                    logging.debug(f"No Dask processes found on node {node}")
            except subprocess.CalledProcessError as e:
                logging.debug(f"Error executing command on node {node}: {e}")
            except Exception as ex:
                logging.debug(f"An error occurred: {ex}")
            

if __name__ == "__main__" :
    handle_signals()

    parser = OptionParser()
    parser.add_option("-s", "--start", action="store_true", dest="start",
                  help="start Dask worker", default=True)
    
    parser.add_option("-w", "--pilot-working-directory", type="string", action="store", dest="pilot_working_directory", default="/Users/pmantha/work/pcs-2cebd082-52f0-4cf3-af71-2a89b82dcfd3/Dask-f028ab1e-6d50-11ef-af5b-d2b91fefeab3", help="Working directory to execute agent in")

    parser.add_option("-f", "--scheduler-file", type="string", action="store", dest="scheduler_file", default="/Users/pmantha/work/pcs-2cebd082-52f0-4cf3-af71-2a89b82dcfd3/scheduler", help="Scheduler information file path")
        
    parser.add_option("-c", "--worker-config-file", type="string", action="store", dest="worker_config_file", default="/Users/pmantha/work/pcs-2cebd082-52f0-4cf3-af71-2a89b82dcfd3//worker_config.json", help="Worker config file")
    
    parser.add_option("-n", "--worker-name", type="string", action="store", dest="worker_name", default="pq-dask-worker", help="Name of the worker")


    
    # Parse Option from commandline arguments
    (options, args) = parser.parse_args()
  
    # Initialize object for managing Dask clusters
    dask_agent = DaskPilotAgent(options.pilot_working_directory, 
                                 options.scheduler_file, 
                                 options.worker_config_file, options.worker_name)
     
    if options.start:
        dask_agent.start_workers()
    
    logging.info("Finished launching of Dask Cluster - Sleeping now")

    while not STOP:
        logging.debug("Dask Agent is running... " + str(STOP))
        time.sleep(10)
        
    logging.info("Stopping Dask Cluster")
    dask_agent.stop_workers()
            
        
    
    
    
