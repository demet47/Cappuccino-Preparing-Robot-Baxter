import subprocess

localpath = "C:\\Users\\dmtya\\OneDrive\\Masaüstü\\Graduation Project\\Cappuccino-Preparing-Robot-Baxter\\output.csv"
remotepath = "ruser@79.123.176.144:/home/ruser/output.csv"
command = ['scp', localpath, remotepath]

#subprocess.run(["scp", localpath, remotepath])
foo_proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
time.sleep()
foo_proc.stdin.write(b"rethink\n")
outputlog, errorlog = foo_proc.communicate()
print(outputlog)
print(errorlog)
foo_proc.stdin.close()