import subprocess

localpath = "initial_data/trajectoriessalimdemet.pt"
remotepath = "/home/ruser/trajectoriessalimdemet.pt"

subprocess.run(["scp", localpath, remotepath])