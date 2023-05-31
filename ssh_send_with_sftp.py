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
localpath = "C:/Users/dmtya/Cappuccino-Preparing-Robot-Baxter/trajectories/output_1.csv"
remotepath = "/home/ruser/trajectories/" + sys.argv[1]
command_to_run_on_remote = "cd alper_workspace; source activate_env.sh; nohup rosrun baxter_examples joint_trajectory_file_playback.py -f ../trajectories/output_1.csv"


ssh.connect(server, username=username, password=password)
sftp = ssh.open_sftp()
sftp.put(localpath, remotepath)
time.sleep(4)
sin, sout, serr = ssh.exec_command(command_to_run_on_remote)
print("SENT THE TRAJECTORY FILE")
print("STD OUT: ", sout.read().decode())
print("STD ERR: ", serr.read().decode())

sftp.close()
ssh.close()