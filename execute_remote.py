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


