import subprocess
import time
from urllib.parse import urlparse
import uuid
import os

import pilot
from pilot import job
from pilot.job import slurm, ssh
from pilot.pilot_enums_exceptions import ExecutionEngine
from pilot.pcs_logger import PilotComputeServiceLogger


class PilotManager:
    def __init__(self, working_directory, execution_engine=ExecutionEngine.DASK):
        self.working_directory = working_directory
        self.logger = PilotComputeServiceLogger(self.working_directory)
        self.scheduler_info_file=f'{self.working_directory}/scheduler'
        self.execution_engine = execution_engine
        self.pilot_job = None

    def get_id(self):
        return self.pilot_id

    def start_scheduler(self):
        pass

    def create_pilot(self):
        pass

    def submit_pilot(self, pilot_compute_description):
        self._setup_pilot_job(pilot_compute_description)

        try:
            
            pilot_js, pilot_jd = self._setup_pilot_saga_job(self.pilot_compute_description)
            self.pilot_job = pilot_js.create_job(pilot_jd)
            self.pilot_job.run()
            self.pilot_job_id = self.pilot_job.get_id()
            self.logger.info(f"Job submitted with id: {self.pilot_job_id} and state: {self.pilot_job.get_state()}")
            return self.pilot_job
        except Exception as ex:
            self.logger.error(f"Pilot job submission failed: {str(ex)}")
            raise ex

    def _setup_pilot_job(self, pilot_compute_description):
        self.pilot_compute_description = pilot_compute_description
        self.pilot_id = self.execution_engine.name + "-" + uuid.uuid1().__str__()
        self.pilot_working_directory = os.path.join(self.working_directory, self.pilot_id)
        self.pilot_compute_description["working_directory"] = self.pilot_working_directory
        
        try:
            self.logger.info(f"Creating working directory: {self.pilot_working_directory}")
            os.makedirs(self.pilot_working_directory)
        except Exception:
            self.logger.error(f"Failed to create working directory: {self.pilot_working_directory}")
        

        
    def wait(self):
        state = self.pilot_job.get_state().lower()        
        while state != "running" and state != "done":
            self.logger.debug(f"Pilot Job {self.pilot_job_id} State {state}")            
            time.sleep(6)
            state = self.pilot_job.get_state().lower()
        

    def get_config_data(self):
        pass
                    

    def get_pilot_status(self):
        pass

    def cancel(self):
        if self.pilot_job:
            self.pilot_job.cancel()

        time.sleep(2)

    def _setup_pilot_saga_job(self, pilot_compute_description):
        resource_url = pilot_compute_description["resource"]
        url_scheme = urlparse(resource_url).scheme

        js = self._get_saga_job_service(resource_url, url_scheme)
        
        executable = self._get_pilot_saga_job_executable()
        arguments = self._get_saga_job_arguments()                
        self.logger.debug(f"Launching pilot with {executable} and arguments: {arguments}")

        jd = {"executable": executable, "arguments": arguments}
        jd.update(pilot_compute_description)

        return js, jd

    def _get_saga_job_service(self, resource_url, url_scheme):
        if url_scheme.startswith("slurm"):
            js = slurm.Service(resource_url)
        else:
            js = ssh.Service(resource_url)
        return js
    
    def _get_pilot_saga_job_executable(self):
        return "python"
        
    def _stop_existing_processes(self, process_name):
        # Find the process IDs of all running dask-scheduler processes
        try:
            result = subprocess.run(['pgrep', '-f', process_name], stdout=subprocess.PIPE, text=True)
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    print(f"Stopping existing dask process with PID: {pid}")
                    subprocess.run(['kill', '-9', pid])
        except Exception as e:
            print(f"Error stopping existing schedulers: {e}")

    def is_scheduler_started(self):
        return os.path.exists(os.path.join(self.working_directory, "scheduler"))
