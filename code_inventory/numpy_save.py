import numpy as np
import pandas as pd
import os

observation_right = pd.read_csv(os.getcwd().replace("\\", "/") + "/carry_data/train_left/carry_1_1.csv").to_numpy()[:,11:]
observation_left = pd.read_csv(os.getcwd().replace("\\", "/") + "/carry_data/train_right/carry_8_1.csv").to_numpy()[:,3:11]
np.save("left_arm_default_data.npy",observation_left)
np.save("right_arm_default_data.npy", observation_right)