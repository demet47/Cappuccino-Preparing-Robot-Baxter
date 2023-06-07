import argparse

import numpy as np
import matplotlib.pyplot as plt
import wandb

import pytorch_models
import utils
import data

def interpolateJointData(trajectory, num_of_steps_expected):
    return np.append(trajectory, np.tile(trajectory[-1], ((num_of_steps_expected - len(trajectory)), 1)))


def plot_trajs(dataset, pretext, model, config, log_wandb=True, save_path=None):
    if save_path:
        results = {}

    m_post, cov_post = model.mean.copy(), model.cov.copy()
    for traj_i, traj_name in zip(dataset.data, dataset.names):
        context_times = [0, 1, 2, -1, -2, -3]
        x_cond = traj_i[context_times, 0].numpy()
        y_cond = traj_i[context_times, :].numpy()[:, config["target_dims"]]
        model.condition(x_cond, y_cond)
        mean, std = model.query(traj_i[:, 0].numpy())
        if save_path:
            results[traj_name] = {"mean": mean, "std": std}
        fig, ax = plt.subplots(2, 4, figsize=(16, 8))
        for i in range(2):
            for j in range(4):
                ax[i][j].plot(traj_i[:, 0], traj_i[:, i*4+j+9], c="k")
                ax[i][j].plot(traj_i[:, 0], mean[:, i*4+j], c="b")
                ax[i][j].fill_between(traj_i[:, 0], mean[:, i*4+j] - std[:, i*4+j],
                                      mean[:, i*4+j] + std[:, i*4+j], color="b", alpha=0.2)
                ax[i][j].scatter(traj_i[context_times, 0], traj_i[context_times, i*4+j+9], c="r", marker="x")
                ax[i][j].set_title(f"Joint {i*4+j+9}")
        wandb.log({f"{pretext}-{traj_name}": wandb.Image(fig)})
        plt.close(fig)
        model.mean, model.cov = m_post, cov_post  # reset model to unconditioned state
    if save_path:
        np.save(save_path, results)


parser = argparse.ArgumentParser("Run the ProMP model on a given dataset")
parser.add_argument("--config", help="path to config file", type=str, default="config.yaml")
args = parser.parse_args()

config = utils.parse_and_init(args)

trainset = data.BaxterDemonstrationDataset(config["trainset_path"])
testset = data.BaxterDemonstrationDataset(config["testset_path"])


X0, X1, X2, X3, X4 = (np.genfromtxt('./initial_data/salimdemet1_0.csv', delimiter=','), np.genfromtxt('./initial_data/salimdemet1_1.csv', delimiter=','), 
                      np.genfromtxt('./initial_data/salimdemet1_2.csv', delimiter=','), np.genfromtxt('./initial_data/salimdemet1_3.csv', delimiter=','),
                      np.genfromtxt('./initial_data/salimdemet1_4.csv', delimiter=','))

X = np.array([X0, X1, X2, X3, X4], dtype=object) #list of np arrays
length = len(max(X, key=len))
X = np.array([interpolateJointData(test, length) for test in X])
X = X.reshape(5,X.shape[1]//17, 17)


Y0, Y1, Y2, Y3, Y4 = (np.genfromtxt('./initial_data/salimdemet2_0.csv', delimiter=','), np.genfromtxt('./initial_data/salimdemet2_1.csv', delimiter=','), 
                      np.genfromtxt('./initial_data/salimdemet2_2.csv', delimiter=','), np.genfromtxt('./initial_data/salimdemet2_3.csv', delimiter=','),
                      np.genfromtxt('./initial_data/salimdemet2_4.csv', delimiter=','))
Y = np.array([Y0, Y1, Y2, Y3, Y4], dtype=object)
length = len(max(Y, key=len))
Y = np.array([interpolateJointData(element, length) for element in Y])
Y = Y.reshape(5,Y.shape[1]//17, 17)



VX_0, VX_1, VX_2, VX_3, VX_4 = (np.genfromtxt('./initial_data/salimdemet1_5.csv', delimiter=','), np.genfromtxt('./initial_data/salimdemet1_6.csv', delimiter=','),
                          np.genfromtxt('./initial_data/salimdemet1_7.csv', delimiter=','), X3, X0)
v_X = np.array([VX_0, VX_1, VX_2, VX_3, VX_4], dtype=object)
length = len(max(v_X, key=len))
v_X = np.array([interpolateJointData(element, length) for element in v_X])
v_X = v_X.reshape(5,v_X.shape[1]//17, 17)



v_Y = np.array([Y3, Y1, Y4, Y2, Y0], dtype=object)
length = len(max(v_Y, key=len))
v_Y = np.array([interpolateJointData(element, length) for element in v_Y])
v_Y = v_Y.reshape(5,v_Y.shape[1]//17, 17)



xmin, xmax = np.inf, -np.inf
for traj in X:
    xmin = min(traj[:][0].min(), xmin)
    xmax = max(traj[:][0].max(), xmax)

slack = (xmax - xmin) * 0.1
model = pytorch_models.ProMP(n_dims=len(config["target_dims"]), n_basis=config["n_basis"],
                     kernel_range=(xmin-slack, xmax+slack))
model.learn_from_demonstrations(X.tolist(), Y.tolist())

plot_trajs(trainset, "train", model, config)
plot_trajs(testset, "test", model, config)
