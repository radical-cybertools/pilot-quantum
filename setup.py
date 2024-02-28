#!/usr/bin/env python

import os
import sys

from setuptools import setup, find_packages
import subprocess

#try:
#    import saga
#except:
#    print "#."

#if sys.version_info < (2, 6):
#    sys.stderr.write("BigJob requires Python 2.6 and above. Installation unsuccessful!")
#    sys.exit(1)

VERSION_FILE="VERSION"    
    

def update_version():
    if not os.path.isdir(".git"):
        print("This does not appear to be a Git repository.")
        return
    try:
        p = subprocess.Popen(["git", "describe",
                              "--tags", "--always"],
                             stdout=subprocess.PIPE)
    except EnvironmentError:
        print("Unable to run git, not modifying VERSION")
        return
    stdout = p.communicate()[0]
    if p.returncode != 0:
        print("Unable to run git, not modifying VERSION")
        return
    
    ver = stdout.strip()
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VERSION')
    f = open(fn, "w")
    f.write(ver)
    f.close()
    print("Pilot-Quantum VERSION: '%s'" % ver)


def get_version():
    try:
        fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VERSION')
        f = open(fn)
        version = f.read().strip()
        f.close()
    except EnvironmentError:
        return "-1"
    return version

def create_manifest():
    pass

    
#update_version()
    
setup(name='Pilot-Quantum',
      version=get_version(),
      description='Framework for Management of Quantum-HPC applications on HPC clusters (Torque/PBS/SLURM)',
      author='Andre Luckow',
      author_email='aluckow@clemson.edu',
      url='https://github.com/radical-cybertools/pilot-quantum',
      classifiers = ['Development Status :: 5 - Production/Stable',                  
                    'Programming Language :: Python',
                    'Environment :: Console',                    
                    'Topic :: Utilities',
                    ],
      platforms = ('Unix', 'Linux', 'Mac OS'),
      license = "License :: OSI Approved :: Apache Software License",
      include_package_data = True,
      package_dir = {'':'.'},
      packages=find_packages(),
            # ['hadoop1', 'hadoop2', 'hadoop2.configs.default',
            #     'hadoop2.configs.default',
            #     'hadoop2.configs.stampede',
            #     'hadoop2.configs.gordon',
            #     "spark", "commandline"],

      # data files for easy_install
      package_data= {'': ['*.xml', '*.yaml', '*.properties']},
      install_requires=['uuid', 'pykafka', 'argparse', 'python-hostlist', 'dask', 'distributed', 'pyspark', 'paramiko', 'boto3'],
      entry_points = {
        'console_scripts': ['pq=commandline.main:main',
                            'pilot-quantum=commandline.main:main']
      }
)
