#!/usr/bin/env python
import json
import os
import logging
import hostlist


class PilotAgent():
    def __init__(self, working_directory, scheduler_file_path, worker_config_file):
        self.working_directory = working_directory
        self.scheduler_file_path = scheduler_file_path
        self.worker_config_file = worker_config_file
        logging.basicConfig(filename='agent.log', level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_expanded_hostlist(self, hosts):
        return hostlist.expand_hostlist(hosts)
    
    def get_scheduler_address(self):
        with open(self.scheduler_file_path, 'r') as f:
            details = json.load(f)
            return details["agent_scheduler_address"]
        
        
    def get_worker_config_json(self):
        with open(self.worker_config_file, 'r') as f:
            return json.load(f)

    
    def get_nodelist_from_resourcemanager(self):
        nodes = ["localhost"]
        if os.environ.get("SLURM_NODELIST") is not None:            
            nodes = self.get_expanded_hostlist(os.environ.get("SLURM_NODELIST"))
        elif os.environ.get("PBS_NODEFILE") is not None:
            nodes = self.get_expanded_hostlist(os.environ.get("PBS_NODEFILE"))

        nodes =[i.strip() for i in nodes] 
        self.logger.info("Agent nodes on which compute will be scheduled: %s"%(nodes))
        return nodes

    def start_workers(self):
        pass


        

        
        
    
    
    
