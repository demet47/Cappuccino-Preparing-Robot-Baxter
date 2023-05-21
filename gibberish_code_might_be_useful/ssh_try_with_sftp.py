import os
import paramiko
import sys

ssh = paramiko.SSHClient()
ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))


server = "79.123.176.144" # server ip address
username = "ruser"
password = "rethink" #TODO: carry these to a environment variable file
localpath = "C:\\Users\\dmtya\\OneDrive\\Masaüstü\\Graduation Project\\Cappuccino-Preparing-Robot-Baxter\\output.csv"
remotepath = "/home/output.csv"


ssh.connect(server, username=username, password=password)
sftp = ssh.open_sftp()
sftp.put(localpath, remotepath)
sftp.close()
ssh.close()