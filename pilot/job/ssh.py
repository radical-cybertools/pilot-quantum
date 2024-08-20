#!/usr/bin/env python

import datetime
import os
import subprocess
import sys
import time
import traceback
import uuid
import pprint
from urllib.parse import urlparse

from pilot.pcs_logger import PilotComputeServiceLogger


class State:
    UNKNOWN = "Unknown"
    PENDING = "Pending"
    RUNNING = "Running"
    FAILED = "Failed"
    DONE = "Done"


class Service(object):
    """ Plugin for SSH

        Manages endpoint in the form of:

            ssh://<SSH Endpoint>

    """

    def __init__(self, resource_url, pilot_compute_description=None):
        """Constructor"""
        self.resource_url = resource_url
        self.pilot_compute_description = pilot_compute_description
        self.logger = PilotComputeServiceLogger()

    def create_job(self, job_description):
        if "pilot_compute_description" in job_description:
            self.pilot_compute_description = job_description["pilot_compute_description"]
        j = Job(job_description, self.resource_url, self.pilot_compute_description)
        return j

    def __del__(self):
        pass


class Job(object):
    """ Plugin for SSH (to execute defined command on remote machine)

    """

    def __init__(self, job_description, resource_url, pilot_compute_description):
        self.resource_url = resource_url
        self.logger = PilotComputeServiceLogger()

        self.job_description = job_description
        self.pilot_compute_description = job_description

        self.working_directory = os.getcwd()

        if "working_directory" in self.job_description:
            self.working_directory = self.job_description["working_directory"]
            try:
                os.makedirs(self.working_directory, exist_ok=True)
            except:
                pass

        self.logger.info(f"Working_directory {self.working_directory}, Job Description: {self.job_description}")
        self.host = urlparse(resource_url).hostname
        self.user = None
        if urlparse(resource_url).username is not None:
            self.user = urlparse(resource_url).username
        self.logger.debug("URL: " + str(self.resource_url) + " Host: " + self.host)
        self.id = "pilot-quantum-ssh-" + str(uuid.uuid1())
        self.job_id = self.id
        self.job_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.job_output = open(
                                os.path.join(self.working_directory,"pilotquantum_agent_ssh_output_" + self.job_timestamp + ".log"), "w")
        self.job_error = open(
                              os.path.join(self.working_directory, "pilotquantum_agent_ssh_error_" + self.job_timestamp + ".log"), "w")
        self.ssh_process = None

    def run(self):
        """ run command via ssh on VM"""
        # Submit job
   

        TRIAL_MAX = 3
        trials = 0
        while trials < TRIAL_MAX:
            try:
                running = self.check_vm_running()
                if not running:
                    trials = trials + 1
                    time.sleep(30)
                    continue
                else:
                    break
            except:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                self.logger.warning("Submission failed: " + str(exc_value))
                # self.__print_traceback()
                trials = trials + 1
                time.sleep(3)
                if trials == TRIAL_MAX:
                    raise Exception("Submission of agent failed.")

        self.logger.debug("Job State : %s" % (self.get_state()))
        self.run_command()


    def check_vm_running(self):
        """ check if VM is running"""
        
        args = []
        if self.user is None or self.user =="" or self.user == "None":
            args.extend(["ssh", self.host, "/bin/date"])            
        else:
            args.extend(["ssh", "-l", self.user, self.host, "/bin/date"])
            
        
        #self.logger.debug("Execute: " + str(args))
        self.logger.debug("Execute: " + pprint.pformat(args))
        subprocess_handle = subprocess.Popen(args=args,
                                             stdout=self.job_output,
                                             stderr=self.job_error,
                                             cwd=self.working_directory,
                                             shell=False)
        running = False
        if subprocess_handle.poll() is not None and subprocess_handle.poll() != 0:
            self.logger.warning("Submission failed.")
        else:
            self.logger.debug("Test Job succeeded")
            running = True
        subprocess_handle.kill()
        return running

    def wait_for_running(self, node):
        pass

    def get_id(self):
        return self.job_id

    def get_state(self):
        try:
            running = self.check_vm_running()
            if running:
                return State.RUNNING
            else:
                return State.UNKNOWN
            # result = State.UNKNOWN
            # try:
            #     if self.dask_process != None:
            #         rc = self.dask_process.poll()
            #         if rc == None:
            #             result = State.RUNNING
            #         elif rc != 0:
            #             result = State.FAILED
            #         elif rc == 0:
            #             result = State.DONE
        except:
            self.logger.warning("Instance not reachable/active yet...")

    def cancel(self):
        if self.ssh_process != None: self.ssh_process.terminate()
        self.job_output.close()
        self.job_error.close()

    
    def get_nodes_list(self):
        return [self.host]  # only single host via SSH

    
    def get_node_list(self):
        self.get_nodes_list()


    def get_nodes_list_public(self):
        self.get_nodes_list()
        

    def run_command(self):

        if self.user is not None and self.user != "" and self.user != "None":
            command = "ssh -o 'StrictHostKeyChecking=no' -l %s %s -t \"cd %s && bash -ic '%s %s'\"" % \
                      (self.user, self.host,
                      self.working_directory,
                       str(self.pilot_compute_description["executable"]),
                       " ".join(self.pilot_compute_description["arguments"]))
        else:
            command = "ssh -o 'StrictHostKeyChecking=no'  %s -t \"cd %s && bash -ic '%s %s'\"" % \
                      (self.host,
                       self.working_directory,
                       str(self.pilot_compute_description["executable"]),
                       " ".join(self.pilot_compute_description["arguments"]))

        self.logger.info("Execute SSH Command: {0}".format(command))
        # status = subprocess.call(command, shell=True)
        for i in range(3):
            self.ssh_process = subprocess.Popen(command, shell=True,
                                                cwd=self.working_directory,
                                                stdout=self.job_output,
                                                stderr=self.job_error,
                                                close_fds=True)
            time.sleep(10)
            if self.ssh_process.poll is not None:
                break

    ###########################################################################
    # private methods
    def __print_traceback(self):
        exc_type, exc_value, exc_traceback = sys.exc_info()
        self.logger.error("*** print_tb:")
        traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
        self.logger.error("*** print_exception:")
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  limit=2, file=sys.stdout)


if __name__ == "__main__":
    RESOURCE_URL_EDGE = "ssh://localhost"
    WORKING_DIRECTORY_EDGE = "/home/aluckow"
    job_description = {
        "resource": RESOURCE_URL_EDGE,
        "working_directory": WORKING_DIRECTORY_EDGE,
        "number_of_nodes": 1,
        "cores_per_node": 1,
        "dask_cores": 2,
        "project": "TG-MCB090174",
        "queue": "normal",
        "walltime": 359,
        "type": "dask"
    }
    job_service = Service("ssh://localhost")
    job = job_service.create_job(job_description)
    job.run()
    print(job.get_state())
