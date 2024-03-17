#!/usr/bin/env python
""" Ray Bootstrap Script (based on Ray 2.7 release) """
import os, sys
import pdb
import random
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
import ray
import getpass

from pilot.util.ssh_utils import execute_ssh_command, execute_ssh_command_as_daemon


logging.basicConfig(level=logging.DEBUG)

WORKING_DIRECTORY = os.path.join(os.getcwd())
try:
    os.makedirs(WORKING_DIRECTORY)
except:
    pass


STOP=False

def handler(signum, frame):
    logging.debug("Signal catched. Stop Ray")
    global STOP
    STOP=True
    
    

class RayBootstrap():

    def __init__(self, 
                 working_directory=os.getcwd(), 
                 ray_home=os.getcwd(),  
                 config_name="default", 
                 job_id=None, 
                 cores_per_node=1, 
                 gpus_per_node=0, 
                 ip_head_node=None):
        
        self.working_directory=working_directory
        self.ray_home=ray_home
        self.config_name=config_name
        self.job_id = job_id
        self.job_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.job_working_directory = os.path.join(self.working_directory) 
        self.job_conf_dir = os.path.join(self.job_working_directory, "config")
        self.job_output = open(os.path.join(self.working_directory, 
                                            "pilot-agent-" +  
                                            self.job_id[:12] + "-" + self.job_timestamp + "_output.log"), "w")
        self.job_error = open(os.path.join(self.working_directory, 
                                            "pilot-agent-" +  
                                            self.job_id[:12] + "-" + self.job_timestamp + "_output.log"), "w")
        self.ray_headnode_address = ""
        self.ray_process = None                
        self.cores_per_node=int(cores_per_node)
        self.gpu_per_nodes=int(gpus_per_node)
        self.ssh_key = "~/.ssh/mykey" #perlmutter - TODO: Should make it as input parameter from the pilot job description.
        self.ray_memory_limit=92e9    #Stampede
        self.ip_head_node=ip_head_node
        #self.ray_memory_limit=110e9 #Wrangler
        try:
            os.makedirs(self.job_conf_dir)
        except:
            pass


    
    def get_ray_properties(self, master, hostname, broker_id):
        module = "ray.configs." + self.config_name
        print(("Access config in module: " + module + " File: ray.properties"))
        my_data = pkg_resources.resource_string(module, "ray.properties")
        #my_data = my_data%(broker_id, hostname, hostname, master)
        #my_data = os.path.expandvars(my_data)
        #return my_data


    ############################################################################
    ## Get Node List from Resource Management System
    def get_pbs_allocated_nodes(self):
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

    def get_sge_allocated_nodes(self):
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
                    nodes.append(columns[0]+"\n")
            except:
                    pass
        nodes.reverse()
        return list(set(nodes))
    
    def get_slurm_allocated_nodes(self):
        print("Init nodefile from SLURM_NODELIST")
        hosts = os.environ.get("SLURM_NODELIST") 
        if hosts == None:
            return ["localhost"]

        print("***** Hosts: " + str(hosts)) 
        hosts=hostlist.expand_hostlist(hosts)
        number_cpus_per_node = 1
        if os.environ.get("SLURM_CPUS_ON_NODE")!=None:
            number_cpus_per_node=int(os.environ.get("SLURM_CPUS_ON_NODE"))
        freenodes = []
        for h in hosts:
            #for i in range(0, number_cpus_per_node):
            freenodes.append((h + "\n"))
        return list(set(freenodes))

    
    def get_nodelist_from_resourcemanager(self):
        if (os.environ.get("PBS_NODEFILE") != None and os.environ.get("PBS_NODEFILE") != ""):
            nodes = self.get_pbs_allocated_nodes()
        elif (os.environ.get("SLURM_NODELIST") != None):
            nodes = self.get_slurm_allocated_nodes()
        elif (os.environ.get("PE_HOSTFILE") != None):
            nodes = self.get_sge_allocated_nodes()        
        else:
            if self.ip_head_node==None:
                hostname = socket.gethostname()
                ip_address = socket.gethostbyname(hostname)
            else:
                ip_address=self.ip_head_node
            nodes = [ip_address]
        nodes =[i.strip() for i in nodes] # remove white spaces from host names
        return nodes


    ############################################################################
    def configure_ray(self):
        logging.debug("Ray Instance Configuration Directory: " + self.job_conf_dir)
        self.nodes = self.get_nodelist_from_resourcemanager()
        logging.debug("Ray nodes: " + str(self.nodes))
                  

    def start_ray(self):
        logging.debug("Start Ray Head Node")
       
        if ray.is_initialized():
            logging.debug("Ray is already initialized. Shutdown Ray")
            ray.shutdown()
            time.sleep(5)

        #os.environ["RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER"] = "1"        
        # ray_client = ray.init(address=self.nodes[0], 
        #                       _node_ip_address=self.nodes[0],  
        #                       dashboard_host=self.nodes[0], num_cpus=0, num_gpus=0)
        
        self.ray_headnode_address = self.nodes[0]
        # TODO clean conda env
        cmd = "conda activate pilot-quantum; ray stop; export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1; ray start --head   --dashboard-host=%s --num-cpus=0 --num-gpus=0 --ray-client-server-port=10001"%(self.ray_headnode_address)
        print("Start Ray Head Node with command: %s"%(cmd))
        result=execute_ssh_command(host=self.ray_headnode_address, 
                                   user=getpass.getuser(),              command=cmd, arguments=None,
                                   working_directory=self.working_directory,
                                   job_output=self.job_output, job_error=self.job_error)

        ray_client = ray.init()
        self.ray_headnode_address = ray_client.address_info["address"] # update if ray was bound to a different ip
        print("Ray Head Node started at {}. Start workers now".format(ray_client.address_info))
        
        with open(os.path.join(self.working_directory, "ray_scheduler"), "w") as master_file:
            master_file.write(self.ray_headnode_address)

        for i in self.nodes:
            #execute_ssh_command(i, "killall -9 ray")
            command = "export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1; ray start --address %s --num-cpus=%d --num-gpus=%d"%(self.ray_headnode_address , self.cores_per_node, self.gpu_per_nodes)
            
            result=execute_ssh_command(host=i, 
                                       user=getpass.getuser(),              command=command, arguments=None,
                                       working_directory=self.working_directory,
                                       job_output=self.job_output, job_error=self.job_error)

        #ray.shutdown()
        #time.sleep(5)
        
        
        print("Ray started.")


    def check_ray(self):
        try:
            ray_client = ray.init()
            return ray_client.address_info
        except:
            pass
        return None
        
    def stop_ray(self):
        logging.debug("Stop Ray")
        self.stop()
        

    def start(self):
        self.configure_ray()
        self.start_ray()
        
    ############################################################################
    # Extension

    def extend(self):
        self.configure_ray_extension()
        self.start_ray_extension()
    
    def start_ray_extension(self):
        logging.debug("Ray Cluster Scaling NOT SUPPORTED")
        pass
    
    def configure_ray_extension(self):
        logging.debug("Ray Cluster Scaling NOT SUPPORTED")
        
    def stop(self):
        ray.shutdown()
        command = "ray stop --address %s"%(self.ray_headnode_address)
        result=execute_ssh_command(host=self.ray_headnode_address, 
                                   user=getpass.getuser(),              command=command, arguments=None,
                                   working_directory=self.working_directory,
                                   job_output=self.job_output, job_error=self.job_error)
    


#########################################################
#  main                                                 #
#########################################################
if __name__ == "__main__" :
    

    signal.signal(signal.SIGALRM, handler)
    signal.signal(signal.SIGABRT, handler)
    signal.signal(signal.SIGQUIT, handler)
    signal.signal(signal.SIGINT, handler)

    parser = OptionParser()
    parser.add_option("-s", "--start", action="store_true", dest="start",
                  help="start Ray", default=True)
    parser.add_option("-q", "--quit", action="store_false", dest="start",
                  help="terminate Ray")
    
    parser.add_option("-j", "--job", type="string", action="store", 
                      dest="jobid", default="ray-local-" + str(uuid.uuid1()), 
                      help="Job ID of the Pilot Job")
    
    parser.add_option("-c", "--clean", action="store_true", dest="clean",
                  help="clean Ray")
    
    parser.add_option("-p", "--cores-per-node", type="string", action="store", dest="cores_per_node", default=1, help="Core Per Node")

    parser.add_option("-g", "--gpus-per-node", type="string", action="store", dest="gpus_per_node", default=1, help="GPUs Per Node")

    parser.add_option("-i", "--ip-head-node", type="string", action="store", dest="ip_head_node", default=None, help="IP Address to bind Ray daemons to on head bode")

    parser.add_option("-w", "--working-directory", type="string", action="store", dest="working_directory", default=os.getcwd(), help="Working directory to execute agent in")

    parser.add_option("-n", "--config_name", action="store", type="string", dest="config_name", default="default")
    
    # Parse Option from commandline arguments
    (options, args) = parser.parse_args()
    config_name=options.config_name

    # check whether current directory contains the string of the jobid
    if options.jobid in os.getcwd():
        working_directory = os.getcwd()
    else:       
        logging.debug("Create job directory {} within working directory {}".format(options.jobid, options.working_directory))
        working_directory = os.path.join(options.working_directory, options.jobid)
        try:
            os.makedirs(working_directory)
        except:
            pass

    logging.debug("Check Ray Installation on " + socket.gethostname())
    try:
        import ray
    except:
        print("No Ray found. Please install Ray!")
  
    # Initialize object for managing Ray clusters
    ray_cluster = RayBootstrap(working_directory, 
                               None, None, 
                               options.jobid, 
                               options.cores_per_node, 
                               options.gpus_per_node,
                               options.ip_head_node)
    run_timestamp = ray_cluster.job_timestamp

    node_list = ray_cluster.get_nodelist_from_resourcemanager()
    number_nodes = len(node_list)
    print("nodes: %s"%str(node_list))
    performance_trace_filename = "ray_performance_" + run_timestamp + ".csv"
    ray_config_filename = "ray_config_" + run_timestamp
    performance_trace_file = open(os.path.join(working_directory,   
                                               performance_trace_filename), "a")
    start = time.time()
    #performance_trace_file.write("start_time, %.5f"%(time.time()))
 
    if options.start:
        ray_cluster.start()
        ray_nodes=ray_cluster.check_ray()
        logging.debug("Ray Info: %s"%(ray_nodes))
        end_start = time.time()
        performance_trace_file.write("startup, %d, %.5f\n"%(number_nodes, (end_start-start)))
        performance_trace_file.flush()
        with open(os.path.join(working_directory, "ray_started"), "w") as f:
            f.write(str(node_list))
    else:
        ray_cluster.shutdown()
        if options.clean:
            pass
            # directory = "/tmp/zookeeper/"
            # logging.debug("delete: " + directory)
            # shutil.rmtree(directory)
        sys.exit(0)
    
    print("Finished launching of Ray Cluster - Sleeping now")

    while not STOP:
        logging.debug("stop: " + str(STOP))
        time.sleep(10)
            
    ray_cluster.stop()
    os.remove(os.path.join(working_directory, "ray_started"))
    performance_trace_file.write("total_runtime, %d, %.5f\n"%(number_nodes, time.time()-start))
    performance_trace_file.flush()
    performance_trace_file.close()
        
        
    
    
    
