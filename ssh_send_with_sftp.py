import os
import paramiko
import sys
import time

ssh = paramiko.SSHClient()
ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))


server = "79.123.176.144" # server ip address
username = "ruser"
password = "rethink" #TODO: carry these to a environment variable file
localpath = "C:/Users/dmtya/Cappuccino-Preparing-Robot-Baxter/output.csv"
remotepath = "/home/ruser/output.csv"
command_to_run_on_remote_0 = "cd alper_workspace; source activate_env.sh; nohup rosrun baxter_examples joint_trajectory_file_playback.py -f ../output.csv"


ssh.connect(server, username=username, password=password)
sftp = ssh.open_sftp()
sftp.put(localpath, remotepath)
time.sleep(4)
sin, sout, serr = ssh.exec_command(command_to_run_on_remote_0)
print("STD OUT: ", sout.decode())
print("STD ERR: ", serr.decode())

sftp.close()
ssh.close()