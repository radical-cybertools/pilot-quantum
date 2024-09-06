import json
import os
import time
from urllib.parse import urlparse

import distributed
import ray
from pilot import pilot_enums_exceptions
from pilot.pilot_enums_exceptions import ExecutionEngine
from pilot.pilot_compute_service import PilotAPIException
from pilot.plugins.api import PilotManager
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

        # Start a new Dask scheduler in the background
        log_file = os.path.join(self.working_directory, 'ray_scheduler.log')
        
        with open(log_file, 'w') as f:
            process = subprocess.Popen(['ray', 'start', '--head'], stdout=f, stderr=subprocess.STDOUT)

        # Wait and read the log file to get the scheduler address
        for i in range(10):
            time.sleep(5)
            self.client = ray.init()
            try:
                scheduler_address = self.client.address_info["address"] 
                break
            except Exception as e:
                self.logger.info(f"Ray scheduler not ready and getting address failed with error {e}. Waiting... {i}")

        if scheduler_address is None:
            raise RuntimeError("Failed to start Ray scheduler")
        
        print(f"Scheduler started at {scheduler_address}")
        
        # Write scheduler address to file
        scheduler_info = {
            'address': f"ray://{scheduler_address}"
        }
        with open(self.scheduler_info_file, 'w') as f:
            json.dump(scheduler_info, f)

        self.logger.info(f"Scheduler details written to {self.scheduler_info_file}")
        

    def submit_pilot(self, pilot_compute_description):
        return super().submit_pilot(pilot_compute_description)

    def _get_saga_job_arguments(self):
        arguments = ["-m", "pilot.plugins.ray.bootstrap_ray"]

        arguments.extend(["-p", str(self.pilot_compute_description.get("cores_per_node", "1"))])        
        arguments.extend(["-g", str(self.pilot_compute_description.get("gpus_per_node", "1"))])        
        arguments.extend(["-w", str(self.pilot_compute_description.get("working_directory", "/tmp"))])

        return arguments
    
    def get_config_data(self):
        if not self.is_scheduler_started():
            self.logger.debug("Scheduler not started")
            return None
        
        # read the master json file and return the contents
        with open(self.scheduler_info_file, 'r') as f:
            master_host = json.load(f)["address"]

        master_host = urlparse(master_host).hostname

        if master_host is None:
            raise Exception("Scheduler not found")
        
        details = {
                "master_url": "%s:10001" % master_host,
                "web_ui_url": "http://%s:8265" % master_host,
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
        """Returns Ray Client for Scheduler"""
        if self.client is None:
            details = self.get_config_data()
            if details:
                self.logger.info("Connect to Ray: %s" % details["master_url"])
                self.client = ray.init(address="ray://%s" % details["master_url"])
        return self.client

    def cancel(self):
        super().cancel()
        self.stop_ray()

    def wait_tasks(self, tasks):
        with self.get_client():
            k = ray.get(tasks[0])
            self.logger.info(f"Tasks completed: {k}")
            return k


        