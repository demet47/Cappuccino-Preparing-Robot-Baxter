import subprocess

localpath = "C:\\Users\\dmtya\\OneDrive\\Masaüstü\\Graduation Project\\Cappuccino-Preparing-Robot-Baxter\\output.csv"
remotepath = "/home/output.csv"

subprocess.run(["scp", localpath, remotepath])