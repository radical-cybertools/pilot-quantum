# Pilot-Quantum

Last Updated: 02/28/2024

# Overview:

Pilot-Quantum is presented as a Quantum-HPC middleware framework designed to address the challenges of integrating quantum and classical computing resources. It focuses on managing heterogeneous resources, including diverse Quantum Processing Unit (QPU) modalities and various integration types with classical resources, such as accelerators.
 
Requirements:

	* SLURM/PBS/Torque cluster
	* Setup password-less documentation
	

Anaconda is the preferred distribution


## Installation
Requirement (in case a manual installation is required):

The best way to utilize Pilot-Quantum is Anaconda, which provides an easy way to install
important dependencies (such as PySpark and Dask). Make sure the PySpark version is compabitible 
with the Pilot-Quantum version (currently 2.4.4).


    conda install paramiko  
    conda install -c conda-forge boto3 python-openstackclient pykafka pyspark dask distributed python-confluent-kafka pexpect redis-py  
    pip install hostlist

To install Pilot-Quantum type:

    pip install --upgrade .
    
or (if pip issues, e.g. on Stampede2)
   
    python setup.py install
    
    

## Running


Try to run a Hadoop cluster inside a PBS/Torque job:

    psm --resource pbs+ssh://india.futuregrid.org --number_cores 8

Some Blog Posts about SAGA-Hadoop:

    * <http://randomlydistributed.blogspot.com/2011/01/running-hadoop-10-on-distributed.html>


# Packages:

see `hadoop1` for setting up a Hadoop 1.x.x cluster

see `hadoop2` for setting up a Hadoop 2.7.x cluster
 
see `spark` for setting up a Spark 2.2.x cluster

see `kafka` for setting up a Kafka 1.0.x cluster

see `flink` for setting up a Flink 1.1.4 cluster

see `dask` for setting up a Dask Distributed 1.20.2 cluster

see `ray` for setting up a Ray cluster


# Examples:


***Stampede:***

    psm --resource=slurm://localhost --queue=normal --walltime=239 --number_cores=256 --project=xxx


***Gordon:***

    psm --resource=pbs://localhost --walltime=59 --number_cores=16 --project=TG-CCR140028 --framework=spark
    

***Wrangler***

    export JAVA_HOME=/usr/java/jdk1.8.0_45/
    psm --resource=slurm://localhost --queue=normal --walltime=59 --number_cores=24 --project=xxx


# Useful Commands:

* Testing Kafka Config: `bin/kafka-configs.sh --bootstrap-server localhost:9092 --describe --all  --entity-type brokers`
