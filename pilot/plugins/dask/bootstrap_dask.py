#!/usr/bin/env python
""" Dask Bootstrap Script (based on Dask Distributed 2.23.x release) """
import os, sys
import pdb
import urllib.request, urllib.parse, urllib.error
import subprocess
import logging
import uuid
import shutil
import time
import signal
import socket
import hostlist
from optparse import OptionParser
import pkg_resources
import datetime

import psutil
import subprocess

logging.basicConfig(level=logging.DEBUG)

WORKING_DIRECTORY = os.path.join(os.getcwd())
try:
    os.makedirs(WORKING_DIRECTORY)
except:
    pass

STOP = False


def handler(signum, frame):
    logging.debug("Signal catched. Stop Dask")
    global STOP
    STOP = True


class DaskBootstrap():

    def __init__(self, working_directory, dask_home, config_name="default", extension_job_id=None, cores_per_node=None,
                 worker_type="dask"):
        self.working_directory = working_directory
        self.dask_home = dask_home
        self.config_name = config_name
        self.jobid = "dask-" + str(uuid.uuid1())
        self.job_working_directory = os.path.join(WORKING_DIRECTORY)
        self.job_conf_dir = os.path.join(self.job_working_directory, "config")
        self.nodes = []
        self.master = ""
        self.dask_process = None
        self.extension_job_id = extension_job_id
        self.cores_per_node = cores_per_node
        self.dask_ssh = "dask-ssh"
        self.dask_memory_limit = 92e9  # Stampede
        self.worker_type = worker_type
        # self.dask_memory_limit=110e9 #Wrangler
        try:
            os.makedirs(self.job_conf_dir)
        except:
            pass

    def get_dask_properties(self, master, hostname, broker_id):
        module = "dask.configs." + self.config_name
        print(("Access config in module: " + module + " File: dask.properties"))
        my_data = pkg_resources.resource_string(module, "dask.properties")
        # my_data = my_data%(broker_id, hostname, hostname, master)
        # my_data = os.path.expandvars(my_data)
        # return my_data

    #######################################################################################
    ## Get Node List from Resource Management System
    @staticmethod
    def get_pbs_allocated_nodes():
        print("Init PBS")
        pbs_node_file = os.environ.get("PBS_NODEFILE")
        if pbs_node_file == None:
            return ["localhost"]
        f = open(pbs_node_file)
        nodes = f.readlines()
        for i in nodes:
            i.strip()
        f.close()
        return list(set(nodes))

    @staticmethod
    def get_sge_allocated_nodes():
        logging.debug("Init SGE or Local")
        sge_node_file = os.environ.get("PE_HOSTFILE")
        if sge_node_file == None:
            return ["localhost"]
        f = open(sge_node_file)
        sgenodes = f.readlines()
        f.close()
        nodes = []
        for i in sgenodes:
            columns = i.split()
            try:
                for j in range(0, int(columns[1])):
                    print(("add host: " + columns[0].strip()))
                    nodes.append(columns[0] + "\n")
            except:
                pass
        nodes.reverse()
        return list(set(nodes))

    @staticmethod
    def get_slurm_allocated_nodes():
        print("Init nodefile from SLURM_NODELIST")
        hosts = os.environ.get("SLURM_NODELIST")
        if hosts == None:
            return ["localhost"]

        print("***** Hosts: " + str(hosts))
        hosts = hostlist.expand_hostlist(hosts)
        number_cpus_per_node = 1
        if os.environ.get("SLURM_CPUS_ON_NODE") != None:
            number_cpus_per_node = int(os.environ.get("SLURM_CPUS_ON_NODE"))
        freenodes = []
        for h in hosts:
            # for i in range(0, number_cpus_per_node):
            freenodes.append((h + "\n"))
        return list(set(freenodes))

    @staticmethod
    def get_nodelist_from_resourcemanager():
        if (os.environ.get("PBS_NODEFILE") != None and os.environ.get("PBS_NODEFILE") != ""):
            nodes = DaskBootstrap.get_pbs_allocated_nodes()
        elif (os.environ.get("SLURM_NODELIST") != None):
            nodes = DaskBootstrap.get_slurm_allocated_nodes()
        else:
            nodes = DaskBootstrap.get_sge_allocated_nodes()
        nodes = [i.strip() for i in nodes]  # remove white spaces from host names
        return nodes

    #######################################################################################
    def configure_dask(self):
        logging.debug("Dask Instance Configuration Directory: " + self.job_conf_dir)
        self.nodes = self.get_nodelist_from_resourcemanager()
        logging.debug("Dask nodes: " + str(self.nodes))
        self.master = self.nodes[0]  # socket.gethostname().split(".")[0]
        with open(os.path.join(WORKING_DIRECTORY, "dask_scheduler"), "w") as master_file:
            master_file.write(self.master + ":8786")

    def start_dask(self):
        # logging.debug("Start Dask")
        # os.system("killall -s 9 dask-scheduler")
        # os.system("pkill -9 dask-worker")
        # time.sleep(5)
        # command = "%s --remote-dask-worker distributed.cli.dask_worker %s"%(self.dask_ssh, " ".join(self.nodes))
        # if self.cores_per_node is not None and self.dask_memory_limit is not None:
        #         command = "%s --nthreads %s --memory-limit %d --remote-dask-worker distributed.cli.dask_worker %s"%(self.dask_ssh, str(self.cores_per_node), self.dask_memory_limit, " ".join(self.nodes))
        # logging.debug("Start Dask Cluster: " + command)
        # #status = subprocess.call(command, shell=True)
        # self.dask_process = subprocess.Popen(command, shell=True)
        # print("Dask started.")
        self.kill_dask_processes_on_nodes(self.nodes)
        self.launch_dask_scheduler_via_command_line()
        self.launch_dask_workers_via_command_line()

    def check_dask(self):
        try:
            import distributed
            logging.debug("Nodes: %s" % (str(self.nodes)))
            client = distributed.Client(self.master + ":8786")
            return client.scheduler_info()
        except Exception as e:
            logging.debug("failed with error: %s" % e)
        return None

    def stop_dask(self):
        logging.debug("Stop Dask")
        if self.dask_process:
            self.dask_process.kill()

    def start(self):
        self.configure_dask()
        self.start_dask()

    ##################################################################################
    # Extension

    def extend(self):
        self.configure_dask_extension()
        self.start_dask_extension()

    def kill_dask_processes_on_nodes(self, nodes):
        for node in nodes:
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

    def launch_dask_scheduler_via_command_line(self, scheduler_port=8786):
        logging.debug(f"Launching Dask scheduler started at {self.master}:{scheduler_port}")
        scheduler_command = f"ssh {self.master} dask scheduler"
        subprocess.Popen(scheduler_command, shell=True)
        client_info = self.check_dask()
        logging.debug(client_info)

        while client_info is None:
            logging.debug(client_info)
            client_info = self.check_dask()
            time.sleep(1)
        logging.debug(f"Dask scheduler started at {self.master}:{scheduler_port}")

    def launch_dask_workers_via_command_line(self, scheduler_port=8786):
        master_url = f"{self.master}:{scheduler_port}"
        logging.debug(f"Launching Workers against Dask scheduler started at {master_url}")
        worker_command = f"dask worker {master_url}"
        if self.worker_type == "dask-cuda":
            worker_command = f"dask cuda worker {master_url}"

        for node in self.nodes:
            ssh_worker_command = f"ssh {node} {worker_command} --nthreads {self.cores_per_node} --memory-limit {self.dask_memory_limit}"
            subprocess.Popen(ssh_worker_command, shell=True)
            logging.debug(f"Dask worker started on {node} using {ssh_worker_command}")

    def start_dask_extension(self):
        logging.debug("Start Dask Extension")
        os.system("killall -s 9 dask-scheduler")
        os.system("pkill -9 dask-worker")
        time.sleep(5)
        command = "%s --scheduler %s %s" % (self.dask_ssh, self.master, " ".join(self.nodes))
        logging.debug("Start Dask Cluster Extension: " + command)
        # status = subprocess.call(command, shell=True)
        self.dask_process = subprocess.Popen(command, shell=True)
        print("Dask started.")

    def configure_dask_extension(self):
        logging.debug("Dask Instance Configuration Directory: " + self.job_conf_dir)
        self.nodes = self.get_nodelist_from_resourcemanager()
        logging.debug("Dask nodes: " + str(self.nodes))
        self.master = self.find_parent_dask_scheduler()
        with open(os.path.join(WORKING_DIRECTORY, "dask_scheduler"), "w") as master_file:
            master_file.write(self.master + ":8786")

    def find_parent_dask_scheduler(self):
        path_to_parent_dask_job = os.path.join(os.getcwd(), "..", self.extension_job_id, "dask_scheduler")
        print("Master of Parent Cluster: %s" % path_to_parent_dask_job)
        dask_scheduler = None
        with open(path_to_parent_dask_job, "r") as config:
            dask_scheduler = config.read().strip().split(":")[0]  # remove port

        logging.debug("Parent Dask Scheduler: %s" % dask_scheduler)
        return dask_scheduler

    def stop(self):
        self.stop_dask()


#########################################################
#  main                                                 #
#########################################################
if __name__ == "__main__":

    signal.signal(signal.SIGALRM, handler)
    signal.signal(signal.SIGABRT, handler)
    signal.signal(signal.SIGQUIT, handler)
    signal.signal(signal.SIGINT, handler)

    parser = OptionParser()
    parser.add_option("-s", "--start", action="store_true", dest="start",
                      help="start Dask", default=True)
    parser.add_option("-q", "--quit", action="store_false", dest="start",
                      help="terminate Dask")
    parser.add_option("-j", "--job", type="string", action="store", dest="jobid",
                      help="Job ID of Dask Cluster to Extend")
    parser.add_option("-c", "--clean", action="store_true", dest="clean",
                      help="clean Dask")
    parser.add_option("-p", "--cores-per-node", type="string", action="store", dest="cores_per_node", default=None,
                      help="Core Per Node")
    parser.add_option("-t", "--worker-type", type="string", action="store", dest="worker_type", default="dask",
                      help="Dask worker type; acceptable values are dask/dask-cuda")

    parser.add_option("-n", "--config_name", action="store", type="string", dest="config_name", default="default")

    node_list = DaskBootstrap.get_nodelist_from_resourcemanager()
    number_nodes = len(node_list)
    print("nodes: %s" % str(node_list))

    run_timestamp = datetime.datetime.now()
    performance_trace_filename = "dask_performance_" + run_timestamp.strftime("%Y%m%d-%H%M%S") + ".csv"
    dask_config_filename = "dask_config_" + run_timestamp.strftime("%Y%m%d-%H%M%S")
    performance_trace_file = open(os.path.join(WORKING_DIRECTORY, performance_trace_filename), "a")
    start = time.time()
    # performance_trace_file.write("start_time, %.5f"%(time.time()))
    (options, args) = parser.parse_args()
    config_name = options.config_name
    logging.debug("Check Dask Installation on " + socket.gethostname())
    try:
        import distributed
    except:
        print("No Dask Distributed found. Please install Dask Distributed!")

    # initialize object for managing dask clusters
    dask = DaskBootstrap(WORKING_DIRECTORY, None, None, options.jobid, options.cores_per_node, options.worker_type)
    if options.jobid is not None and options.jobid != "None":
        logging.debug("Extend Dask Cluster with PS ID: %s" % options.jobid)
        dask.extend()
    elif options.start:
        dask_start_time = time.time()
        dask.start()
        dask_nodes = []
        while len(dask_nodes) >= number_nodes:
            dask_info = dask.check_dask()
            if dask_info and 'workers' in dask_info:
                dask_nodes = dask_info['workers'].keys()
            logging.debug(
                "len(dask_nodes):%s, Number_nodes:%s, Dask Info: %s" % (len(dask_nodes), number_nodes, dask_nodes))
            time.sleep(1)
        end_start = time.time()
        performance_trace_file.write("startup, %d, %.5f\n" % (number_nodes, (end_start - dask_start_time)))
        performance_trace_file.flush()
        with open("dask_started", "w") as f:
            f.write(str(node_list))

    else:
        dask.stop()
        if options.clean:
            pass
        sys.exit(0)

    logging.debug("Finished launching of Dask Cluster - Sleeping now")

    while not STOP:
        logging.debug("stop: " + str(STOP))
        dask_info = dask.check_dask()
        logging.debug(dask_info)
        time.sleep(10)

    dask.stop()
    os.remove(os.path.join(WORKING_DIRECTORY, "dask_started"))
    performance_trace_file.write("total_runtime, %d, %.5f\n" % (number_nodes, time.time() - start))
    performance_trace_file.flush()
    performance_trace_file.close()
