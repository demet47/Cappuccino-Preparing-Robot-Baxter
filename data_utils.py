import csv
import numpy as np
import os
import pandas as pd


observation_right = np.load(os.getcwd().replace("\\", "/") + "/meta_trajectories/train_joints_right.npy") #shape(1,8,40)
observation_left = np.load(os.getcwd().replace("\\", "/") + "/meta_trajectories/train_joints_left.npy") #shape(1,8,40)

default_position_left = np.load(os.getcwd().replace("\\", "/") + "/meta_trajectories/left_arm_default_data.npy") #shape(4108, 8)
default_position_right = np.load(os.getcwd().replace("\\", "/") + "/meta_trajectories/right_arm_default_data.npy") #shape(4046, 8)


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
    global index
    global default_position_left
    global default_position_right

    headers = np.array(["time","left_s0","left_s1","left_e0","left_e1","left_w0","left_w1","left_w2","left_gripper", "right_s0","right_s1","right_e0","right_e1","right_w0","right_w1","right_w2","right_gripper"])
    headers = headers.reshape(1,17)


    time_len = traj_list.shape[0]
    times = np.arange(0.45, 0.45 + time_len * 0.01, 0.01).reshape(-1, 1)
    traj_list = np.hstack((times,traj_list))
    if left:
        if default_position_right.shape[0] < traj_list.shape[0]:
            default_position_right = np.vstack((default_position_right, default_position_right))
        traj_list = np.hstack((traj_list, default_position_right[:traj_list.shape[0], :]))
    else:
        if default_position_left.shape[0] < traj_list.shape[0]:
            default_position_left = np.vstack((default_position_left, default_position_left))
        traj_list = np.hstack((default_position_left[:traj_list.shape[0], :], traj_list))

    record = np.vstack((headers,traj_list))

    file_name = "./trajectories/output_" + str(index) + ".csv"
    index = index + 1
    np.savetxt(file_name, record, delimiter=',', fmt='%s')
    return "output_" + str(index-1) + ".csv"