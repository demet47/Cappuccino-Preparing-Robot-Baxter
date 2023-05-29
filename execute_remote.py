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
sin, sout, serr = ssh.exec_command(sys.argv[1])

print("STD OUT: ", sout.read().decode())
print("STD ERR: ", serr.read().decode())

sftp.close()
ssh.close()

#command_to_run_on_remote_0 += "; echo -e \"\\x03\""