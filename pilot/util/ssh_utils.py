import os, sys
import time
import subprocess


def execute_ssh_command(host="localhost", user=None, command="/bin/date", arguments=None, working_directory=os.getcwd(),
                        job_output=None, job_error=None, keyfile=None) -> object:
    """
    Execute SSH Command
    :param host:
    :param user:
    :param command:
    :param arguments:
    :param working_directory:
    :param job_output:
    :param job_error:
    :return: True/False - Success or Failure
    """

    user_parameter = ""
    if user is not None:
        user_parameter = "-l {}".format(user)

    key_parameter = ""
    if keyfile is not None:
        key_parameter = "-i {}".format(keyfile)

    arguments_parameter = ""
    if arguments is not None:
        arguments_parameter = " ".join(arguments)
    #     ssh_command = "ssh -o 'StrictHostKeyChecking=no' -l %s %s -t \"bash -ic '%s'\"" % (user, host, command)
    # else:
    #     ssh_command = "ssh -o 'StrictHostKeyChecking=no' %s -t \"bash -ic '%s'\"" % (host, command)

    ssh_command = f"ssh -o 'StrictHostKeyChecking=no' {key_parameter} {user_parameter} {host} -t \"bash -ic '{command} {arguments_parameter}'\""
    print("Execute SSH : {0}".format(ssh_command))
    # status = subprocess.call(command, shell=True)
    for i in range(3):
        ssh_process = subprocess.Popen(ssh_command, shell=True,
                                       cwd=working_directory,
                                       stdout=job_output,
                                       stderr=job_error,
                                       close_fds=True)
        if ssh_process.poll is not None:
            ssh_process.wait(240)
            return True
        time.sleep(10)
    return False


def execute_ssh_command_shell_as_daemon(host, user=None, command="/bin/date", arguments=None, working_directory=os.getcwd(),
                        job_output=None, job_error=None, keyfile=None) -> object:
    """
    Execute SSH Command
    :param host:
    :param user:
    :param command:
    :param arguments:
    :param working_directory:
    :param job_output:
    :param job_error:
    :return: True/False - Success or Failure
    """

    user_parameter = ""
    if user is not None:
        user_parameter = "-l {}".format(user)

    key_parameter = ""
    if keyfile is not None:
        key_parameter = "-i {}".format(keyfile)

    arguments_parameter = ""
    if arguments is not None:
        arguments_parameter = " ".join(arguments)
    #     ssh_command = "ssh -o 'StrictHostKeyChecking=no' -l %s %s -t \"bash -ic '%s'\"" % (user, host, command)
    # else:
    #     ssh_command = "ssh -o 'StrictHostKeyChecking=no' %s -t \"bash -ic '%s'\"" % (host, command)

    ssh_command = f"ssh -o 'StrictHostKeyChecking=no' {key_parameter} {user_parameter} {host} -t \"bash -ic 'nohup {command} {arguments_parameter}&'\""
    print("Execute SSH : {0}".format(ssh_command))
    # status = subprocess.call(command, shell=True)
    for i in range(3):
        ssh_process = subprocess.Popen(ssh_command, shell=True,
                                       cwd=working_directory,
                                       stdout=job_output,
                                       stderr=job_error,
                                       close_fds=True)
        if ssh_process.poll is None:
            time.sleep(10)
            continue
        return True
    return False


def execute_ssh_command_as_daemon(host, user=None, command="/bin/date", arguments=None, working_directory=os.getcwd(),
                        job_output=None, job_error=None, keyfile=None) -> object:
    """
    Execute SSH Command FOR KAFKA
    :param host:
    :param user:
    :param command:
    :param arguments:
    :param working_directory:
    :param job_output:
    :param job_error:
    :return: True/False - Success or Failure
    """

    user_parameter = ""
    if user is not None:
        user_parameter = "-l {}".format(user)

    key_parameter = ""
    if keyfile is not None:
        key_parameter = "-i {}".format(keyfile)

    arguments_parameter = ""
    if arguments is not None:
        arguments_parameter = " ".join(arguments)
    #     ssh_command = "ssh -o 'StrictHostKeyChecking=no' -l %s %s -t \"bash -ic '%s'\"" % (user, host, command)
    # else:
    #     ssh_command = "ssh -o 'StrictHostKeyChecking=no' %s -t \"bash -ic '%s'\"" % (host, command)

    ssh_command = f"ssh -o 'StrictHostKeyChecking=no' {key_parameter} {user_parameter} {host} '{command} {arguments_parameter}'"
    print("Execute SSH : {0}".format(ssh_command))
    # status = subprocess.call(command, shell=True)
    for i in range(3):
        ssh_process = subprocess.Popen(ssh_command, shell=True,
                                       cwd=working_directory,
                                       stdout=job_output,
                                       stderr=job_error,
                                       close_fds=True)
        if ssh_process.poll is not None:
            return True
        time.sleep(10)
    return False



def install_pilot_quantum(hostname, pilot_compute_description):
    """
    Installs and bootstraps latest pilot-quantum and mini apps from github
    :param hostname:
    :return:
    """

    # Pilot-quantum
    start = time.time()
    command = "pip install --upgrade git+ssh://git@github.com/radical-cybertools/pilot-quantum.git"
    result = execute_ssh_command(hostname,
                                 user=pilot_compute_description["os_ssh_username"],
                                 command=command,
                                 keyfile=pilot_compute_description["os_ssh_keyfile"])
    print("Host: {} Command: {} Result: {} Time: {}".format(hostname, command, result, time.time()-start))

    # Dask Distributed
    start = time.time()
    command = "pip install --upgrade dask distributed"
    result = execute_ssh_command(hostname,
                                 user=pilot_compute_description["os_ssh_username"],
                                 command=command,
                                 keyfile=pilot_compute_description["os_ssh_keyfile"])
    print("Host: {} Command: {} Result: {} Time: {}".format(hostname, command, result, time.time() - start))

    # MINI Apps
    start = time.time()
    command = "pip install --upgrade git+ssh://git@github.com/radical-cybertools/quantum-miniapps.git"
    result = execute_ssh_command(hostname,
                                 user=pilot_compute_description["os_ssh_username"],
                                 command=command,
                                 keyfile=pilot_compute_description["os_ssh_keyfile"])
    print("Host: {} Command: {} Result: {} Time: {}".format(hostname, command, result, time.time()-start))
