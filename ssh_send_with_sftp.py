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
password = os.getenv("SSH_PASSWORD") #TODO: carry these to a environment variable file
localpath = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/trajectories/" +  sys.argv[1]
remotepath = "/home/ruser/trajectories/" + sys.argv[1]

command_to_run_on_remote = "bash -l -c '{}'".format("/home/ruser/miniconda3/condabin/conda activate ros_env; source ~/alper_workspace/ros_ws/devel/setup.sh; cd alper_workspace; rosrun baxter_examples joint_trajectory_file_playback.py -f ~/trajectories/" + sys.argv[1])


ssh.connect(server, username=username, password=password)
sftp = ssh.open_sftp()
sftp.put(localpath, remotepath)
print("SENT THE TRAJECTORY FILE")
time.sleep(4)
try:
    sin, sout, serr = ssh.exec_command(command_to_run_on_remote)
    print("STD OUT: ", sout.read().decode())
except:
    print("STD ERR: ", serr.read().decode())

sftp.close()
ssh.close()