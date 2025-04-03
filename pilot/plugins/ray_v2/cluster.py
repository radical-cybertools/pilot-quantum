import json
import os
import socket
import subprocess
import time
from urllib.parse import urlparse

import ray

from pilot.pilot_enums_exceptions import ExecutionEngine
from pilot.plugins.pilot_manager_base import PilotManager
from pilot.util.ssh_utils import execute_ssh_command, execute_ssh_command_as_daemon, get_localhost
import subprocess


class RayManager(PilotManager):
    def __init__(self, working_directory):
        self.client = None
        super().__init__(working_directory=working_directory, execution_engine=ExecutionEngine.RAY)        

    def stop_ray(self):
        # Stop the Ray scheduler
        process = subprocess.Popen(['ray', 'stop'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, stderr = process.communicate()
        return_code = process.returncode
        
        if return_code != 0:
            msg = f"Failed to stop Ray scheduler. Return code: {return_code}. Error: {stderr.decode()}"
            self.logger.error(msg)
            raise RuntimeError(msg)
        else:
            self.logger.info("Ray scheduler stopped successfully.")
                    

    def start_scheduler(self):
        # Stop existing Ray processes
        self.stop_ray()
        self._stop_existing_processes("pilot.plugins.ray_v2.agent")

        # Start a new Ray scheduler in the background
        log_file = os.path.join(self.working_directory, 'ray_scheduler.log')
        
        
        host_node_ip_address = get_localhost()    
        with open(log_file, 'w') as f:
            process = subprocess.Popen(['ray', 'start', '--head', f"--node-ip-address={host_node_ip_address}", "--num-cpus=0", "--num-gpus=0"], stdout=f, stderr=subprocess.STDOUT)
                      
        # Check the status of the process
        process_status = process.poll()
        while process_status is None:
            time.sleep(5)
            process_status = process.poll()
        
        if process_status==0:
            self.logger.info("Ray scheduler started successfully.")
        else:
            self.logger.error(f"Failed to start Ray scheduler. Return code: {process_status}")
            raise RuntimeError(f"Failed to start Ray scheduler. Return code: {process_status}")
            
        
        # Wait and read the log file to get the scheduler address
        for i in range(10):
            ray_client = ray.init(_node_ip_address=host_node_ip_address)
            try:
                scheduler_address = ray_client.address_info["node_ip_address"]                 
                break
            except Exception as e:
                self.logger.info(f"Ray scheduler not ready and getting address failed with error {e}. Waiting... {i}")

        if scheduler_address is None:
            raise RuntimeError("Failed to start Ray scheduler")
        
        if not scheduler_address is host_node_ip_address:
            raise RuntimeError("Node Ip address and Ray scheduler address mismatch")
        
        print(f"Scheduler started at {scheduler_address}")
        
        # Write scheduler address to file
        scheduler_info = {
            'agent_scheduler_address': f"{scheduler_address}:6379",
            'master_url': f"ray://{scheduler_address}:10001",
            "web_ui_url": "http://%s:8265" % scheduler_address,
        }
        with open(self.scheduler_info_file, 'w') as f:
            json.dump(scheduler_info, f)

        self.logger.info(f"Scheduler details written to {self.scheduler_info_file}")
        

    def submit_pilot(self, pilot_compute_description):
        return super().submit_pilot(pilot_compute_description)

    def _get_saga_job_arguments(self):
        arguments = [ "-m", "pilot.plugins.ray_v2.agent",
                     "-s", "True",
                     "-w", self.pilot_working_directory,
                     "-f", self.scheduler_info_file, 
                     "-c", self.worker_config_file, 
                     "-n", self.pilot_compute_description.get("name", "pq-ray-worker"),                     
                     ]
                     
        
        return arguments

    def create_worker_config_file(self):
        worker_config = {
            'cores_per_node': str(self.pilot_compute_description.get("cores_per_node", "1")),
            'gpus_per_node': str(self.pilot_compute_description.get("gpus_per_node", "1")),
        }
        with open(self.worker_config_file, 'w') as f:
            json.dump(worker_config, f)
            
        self.logger.info(f"Worker config file created: {self.worker_config_file}")
    
    def get_config_data(self):
        if not self.is_scheduler_started():
            self.logger.debug("Scheduler not started")
            return None
        
        # read the master json file and return the contents
        with open(self.scheduler_info_file, 'r') as f:
            return json.load(f)
        

    def wait(self):
        super().wait()
        

    def get_client(self, configuration=None) -> object:
        """Returns Ray Client for Scheduler"""
        if self.client is None:
            details = self.get_config_data()
            if details:
                self.logger.info("Connect to Ray: %s" % details["master_url"])
                self.client = ray.init(address="%s" % details["master_url"])
        return self.client

    def cancel(self):
        ray.shutdown()
        self.stop_ray()
        super().cancel()
        

    def wait_tasks(self, tasks):
        completed = 0
        running_ids = tasks
        try:                
            while completed < len(tasks):
                ready, running_ids = ray.wait(running_ids, num_returns=1)
                completed += len(ready)
                self.logger.info(f"Tasks completed: {completed}, Pending: {len(running_ids)}")
        except Exception as e:
            self.logger.error(f"Error waiting for tasks: {e}")                

    
    def get_results(self, tasks):
        self.wait_tasks(tasks) 
        
        try:                                                   
            return ray.get(tasks)
        except Exception as e:
            self.logger.error(f"Error getting results: {e}")                
        
        return None
                


        