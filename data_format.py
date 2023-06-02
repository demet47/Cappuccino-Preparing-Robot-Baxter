import csv
import numpy as np
import os

def convert(path, normalize):
    with open(path, "r") as f:
        # first line is the header, and the second line does not follow 100hz, so skip.
        lines = f.readlines()[2:]
    lines = [[float(item) for item in line.strip().split(",")] for line in lines]
    max = lines[-1][0]
    if normalize == True: 
        for i in range(0,len(lines)):
            lines[i][0] = lines[i][0] / max
    return lines


#folder = "C:/Users/dmtya/Cappuccino-Preparing-Robot-Baxter/carry_data/train" #sys.argv[1]

def path_to_list_of_trajectories(folder, normalize):
    data = []
    max_size = 0
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            path = os.path.join(folder, file)
            l = convert(path, normalize)
            if max_size < len(l):
                max_size = len(l)
            data.append(np.array(l, np.float32))
    return np.array(data), max_size


#normalized times. Extended length elements, extension done with the termination values
def data_per_trajectory(folder_name, normalize):
    list_of_trajectories, max_size = path_to_list_of_trajectories(folder_name, normalize)
    train_joints = []
    train_n = []
    train_t = []
    train_p = []

    for traj in list_of_trajectories:
        j = traj[:,3:] # joint angles
        p = traj[0,1:3] # x, y coordinates
        ext = traj[-1, 3:] # extension last joint angles
        length = max_size - traj.shape[0] # length of extension
        for _ in range(0, length):
            j = np.append(j, [ext], axis=0)
        # j = j.reshape(16,j.size//16)
        train_joints.append(j.T)
        train_n.append(max_size)
        time = np.linspace(0,1, max_size)
        train_t.append(time)
        train_p.append(p)

    return train_joints, train_n, train_t, train_p

# /home/colors/Desktop/Cappuccino-Preparing-Robot-Baxter/carry_data/train1
# a,b,c,d = data_per_trajectory('./carry_data/train1', True)

# print(a[0].shape)
# print(b[0])
# print(c[0])
# print(d[0])