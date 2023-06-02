import csv
import numpy as np
import os
import pandas as pd


observation_right = np.load("./train_joints_right.npy")
observation_left = np.load("./train_joints_left.npy")

default_position_left = np.load("left_arm_default_data.npy")
default_position_right = np.load("right_arm_default_data.npy")

index = 0

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

#takes a (time_length, number_of_joints) sized array of trajectory information
def trajectory_list_to_csv(traj_list, left):
    traj_time = traj_list.shape[1]
    headers = np.array(["time","left_s0","left_s1","left_e0","left_e1","left_w0","left_w1","left_w2","left_gripper"])

    values = traj_list[0].squeeze(0).detach().numpy()
    headers = headers.reshape(1,17)

    time_len = values.shape[0]
    times = np.arange(0.45, 0.45 + time_len * 0.01, 0.01).reshape(-1, 1)
    values = np.hstack((times,values))
    record = np.vstack((headers,values))

    file_name = "./trajectories/output_" + str(index) + ".csv"
    index = index + 1
    np.savetxt(file_name, record, delimiter=',', fmt='%s')