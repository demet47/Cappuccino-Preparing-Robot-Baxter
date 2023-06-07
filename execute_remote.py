import os
import paramiko
import sys
import time
from dotenv import load_dotenv


load_dotenv()
ssh = paramiko.SSHClient()
ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))


server = os.getenv("BAXTER_IP_ADDRESS") # server ip address

username = os.getenv("SSH_USERNAME")
password = os.getenv("SSH_PASSWORD") 


ssh.connect(server, username=username, password=password)
sftp = ssh.open_sftp()
serr = ""
command = "/home/ruser/miniconda3/condabin/conda activate ros_env; source ~/alper_workspace/ros_ws/devel/setup.sh; cd alper_workspace; pwd;"
remote_command = "bash -l -c '{}'".format(command + sys.argv[1])
try:
    sin, sout, serr = ssh.exec_command(remote_command)
    print("STD OUT: ", sout.read().decode())
except:
    print("STD ERR: ", serr.read().decode())


sftp.close()
ssh.close()

#command_to_run_on_remote_0 += "; echo -e \"\\x03\"" to append ctrl + C to the end of cmd

#exp_path = "export PATH=/home/ruser/miniconda3/envs/ros_env/bin:/home/ruser/miniconda3/envs/ros_env/x86_64-conda-linux-gnu/sysroot/usr/bin:/home/ruser/miniconda3/condabin:/usr/local/bin:/usr/bin:/bin:/opt/bin:/usr/x86_64-pc-linux-gnu/gcc-bin/4.7.3"

