import data_utils
import numpy as np
import subprocess


pred = np.load("C:/Users/dmtya/Cappuccino-Preparing-Robot-Baxter/prediction_left_tuned.npy")


data_utils.trajectory_list_to_csv(pred.T, True)
