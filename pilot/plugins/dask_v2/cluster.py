import json
import os
import time
from urllib.parse import urlparse

import distributed
from pilot import pilot_enums_exceptions
from pilot.pilot_enums_exceptions import ExecutionEngine
from pilot.pilot_compute_service import PilotAPIException
from pilot.plugins.pilot_manager_base import PilotManager
import subprocess


class DaskManager(PilotManager):
    def __init__(self, working_directory):
        super().__init__(working_directory=working_directory, execution_engine=ExecutionEngine.DASK)

    def start_scheduler(self):
        # Stop any existing Dask workers
        self._stop_existing_processes('bootstrap_dask')
        
        # Stop any existing Dask workers
        self._stop_existing_processes('dask_v2.agent')
        
        # Stop any existing Dask workers
        self._stop_existing_processes('dask worker')

        # Stop any existing Dask schedulers
        self._stop_existing_processes('dask scheduler')

        # Start a new Dask scheduler in the background
        log_file = os.path.join(self.working_directory, 'dask_scheduler.log')
        with open(log_file, 'w') as f:
            process = subprocess.Popen(['dask', 'scheduler'], stdout=f, stderr=subprocess.STDOUT)

        
        # Wait and read the log file to get the scheduler address
        scheduler_address = None
        timeout = 10  # seconds
        start_time = time.time()
        while time.time() - start_time < timeout:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                if "Scheduler at:" in line:
                    scheduler_address = line.strip().split()[-1]
                    break
            if scheduler_address:
                break
            time.sleep(1)

        if scheduler_address is None:
            raise RuntimeError("Failed to start Dask scheduler")
        
        print(f"Scheduler started at {scheduler_address}")
        
        # Write scheduler address to file
        scheduler_info = {
            'agent_scheduler_address': scheduler_address
        }
        with open(self.scheduler_info_file, 'w') as f:
            json.dump(scheduler_info, f)
        self.logger.info(f"Scheduler details written to {self.scheduler_info_file}")
        

    def submit_pilot(self, pilot_compute_description):
        return super().submit_pilot(pilot_compute_description)

    def _get_saga_job_arguments(self):
        arguments = [ "-m", "pilot.plugins.dask_v2.agent",
                     "-s", "True",
                     "-w", self.pilot_working_directory,
                     "-f", self.scheduler_info_file, 
                     "-c", self.worker_config_file, 
                     "-n", self.pilot_compute_description.get("name", "pq-dask-worker"),                     
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
            master_host = json.load(f)["agent_scheduler_address"]

        master_host = urlparse(master_host).hostname

        if master_host is None:
            raise Exception("Scheduler not found")
        
        details = {
                "master_url": "tcp://%s:8786" % master_host,
                "web_ui_url": "http://%s:8787" % master_host,
            }
        
        return details 
    
    def wait(self):
        super().wait()
        state = self.pilot_job.get_state().lower()

        if state != "running":
            raise PilotAPIException(f"Pilot Job {self.pilot_job_id} failed to start. State: {state}")

        while True:
            if self.is_scheduler_started():
                try:
                    self.logger.info("init distributed client")
                    c = self.get_client()
                    scheduler_info = c.scheduler_info()                    
                    if len(scheduler_info.get('workers')) > 0:
                        self.logger.info(str(c.scheduler_info()))
                        c.close()
                        return
                    else:
                        self.logger.info(f"Dask cluster is still initializing. Waiting... {scheduler_info}")
                except IOError as e:
                    self.logger.warning("Dask Client Connect Attempt {} failed".format(i))

            time.sleep(5)

    def get_client(self, configuration=None) -> object:
        """Returns Dask Client for Scheduler"""
        details = self.get_config_data()
        if details:
            self.logger.info("Connect to Dask: %s" % details["master_url"])
            return distributed.Client(details["master_url"])
        return None

    def cancel(self):
        super().cancel()
        self._stop_existing_processes('dask worker')
        self._stop_existing_processes('dask scheduler')
        # Stop any existing Dask workers
        self._stop_existing_processes('bootstrap_dask')        
        # Stop any existing Dask workers
        self._stop_existing_processes('dask_v2.agent')
        

    def wait_tasks(self, tasks):
        while True:
            completed, not_completed = distributed.wait(tasks, return_when="FIRST_COMPLETED")
            self.logger.info(f"Completed: {len(completed)}, Not Completed: {len(not_completed)}")
            
            if len(completed) > 0:
                for future in completed:
                    if future.exception():
                        self.logger.error(f"Task failed: {future.exception()}")
                        
            if len(completed) == len(tasks):
                break
            
            time.sleep(10)
                    

        