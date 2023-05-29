import os

import wandb
import torch
import yaml
import matplotlib.pyplot as plt
import models
import numpy as np
import torch


def parse_and_init(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # create a save folder if not exists
    save_folder = config["save_folder"]
    os.makedirs(save_folder, exist_ok=True)
    # create an output folder inside the text folder
    out_folder = os.path.join(save_folder, "out")
    os.makedirs(out_folder, exist_ok=True)
    # initialize wandb run
    if ("run_id" in config) and (config["run_id"] is not None):
        wandb.init(project="lwcnmp", entity="colorslab", dir=save_folder,
                   id=config["run_id"], resume="must")
    else:
        wandb.init(project="lwcnmp", entity="colorslab", config=config, dir=save_folder)
    # force device to be cpu if cuda is not available
    if not torch.cuda.is_available():
        wandb.config.update({"device": "cpu"}, allow_val_change=True)
    # also save the config file in the save folder
    with open(os.path.join(save_folder, "config.yaml"), "w") as f:
        yaml.dump(wandb.config, f)
    return wandb.config


def create_model_from_config(config, resume=False, use_wandb=True):
    if config["model"] == "cnmp":
        model = models.CNP(config["in_shape"], config["hidden_size"], config["num_hidden_layers"], config["min_std"])
    elif config["model"] == "lwcnmp":
        model = models.LocallyWeightedCNP(config["in_shape"], config["hidden_size"],
                                          config["num_hidden_layers"], config["min_std"], config["weight_std"])
    else:
        raise ValueError("Invalid model name")

    if resume and use_wandb:
        artifact = wandb.use_artifact(f"colorslab/lwcnmp/model-{wandb.run.id}:latest", type="model")
        artifact_dir = artifact.download()
        state_dict = torch.load(os.path.join(artifact_dir, "model.pt"), map_location="cpu").get("model_state_dict")
        model.load_state_dict(state_dict)
    elif resume:
        state_dict = torch.load(os.path.join(config["save_folder"], "model.pt"),
                                map_location="cpu").get("model_state_dict")
        model.load_state_dict(state_dict)
    return model


def plot_trajs(dataset, pretext, model, config, log_wandb=True, save_path=None):
    if save_path:
        results = {}
    # test the model
    model.eval()
    with torch.no_grad():
        for traj_i, traj_name in zip(dataset.data, dataset.names):
            context_times = [0, 1, 2, -1, -2, -3]
            # predict
            mean, std = model(observation=traj_i[[0, 1, 2, -1, -2, -3]].unsqueeze(0).to(config["device"]),
                              target=traj_i[..., config["context_dims"]].unsqueeze(0).to(config["device"]))
            mean = mean.cpu()
            std = std.cpu()
            if save_path:
                results[traj_name] = {"mean": mean, "std": std}
            fig, ax = plt.subplots(2, 4, figsize=(16, 8))
            for i in range(2):
                for j in range(4):
                    ax[i][j].plot(traj_i[:, 0], traj_i[:, i*4+j+9], c="k")
                    ax[i][j].plot(traj_i[:, 0], mean[0][:, i*4+j+8], c="b")
                    ax[i][j].fill_between(traj_i[:, 0], mean[0][:, i*4+j+8] - std[0][:, i*4+j+8],
                                          mean[0][:, i*4+j+8] + std[0][:, i*4+j+8], color="b", alpha=0.2)
                    ax[i][j].scatter(traj_i[context_times, 0], traj_i[context_times, i*4+j+9], c="r", marker="x")
                    ax[i][j].set_title(f"Joint {i*4+j+9}")
            wandb.log({f"{pretext}-{traj_name}": wandb.Image(plt)})
            plt.close(fig)
    model.train()
    if save_path:
        torch.save(results, save_path)


def save_checkpoint(model, optimizer, iteration, loss, save_path, aliases=["latest"]):
    model.cpu().eval()
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iter": iteration,
        "loss": loss,
    }, save_path)
    artifact = wandb.Artifact(f"model-{wandb.run.id}", type="model")
    artifact.add_file(save_path)
    wandb.log_artifact(artifact, aliases=aliases)
    model.to(wandb.config["device"]).train()

def load_checkpoint_from_wandb():
    # you should initialize the wandb run with the respective
    # run_id before calling this function
    full_run_id = f"colorslab/lwcnmp/model-{wandb.run.id}:latest"
    artifact = wandb.use_artifact(full_run_id, type='model')
    artifact_dir = artifact.download()
    checkpoint = torch.load(os.path.join(artifact_dir, "model.pt"))
    return checkpoint

'''

def interpolateJointData(trajectory, num_of_steps_expected):
    return np.append(trajectory, np.tile(trajectory[-1], ((num_of_steps_expected - len(trajectory)), 1)))


def form_pt_dictionary_from_csv_dataset():
    Y= np.array([np.genfromtxt('./initial_data/salimdemet1_0.csv', delimiter=',')[1:,1:], np.genfromtxt('./initial_data/salimdemet1_1.csv', delimiter=',')[1:,1:], 
                np.genfromtxt('./initial_data/salimdemet1_2.csv', delimiter=',')[1:,1:], np.genfromtxt('./initial_data/salimdemet1_3.csv', delimiter=',')[1:,1:],
                np.genfromtxt('./initial_data/salimdemet1_4.csv', delimiter=',')[1:,1:], np.genfromtxt('./initial_data/salimdemet2_0.csv', delimiter=',')[1:,1:],
                np.genfromtxt('./initial_data/salimdemet2_1.csv', delimiter=',')[1:,1:], np.genfromtxt('./initial_data/salimdemet1_2.csv', delimiter=',')[1:,1:],
                np.genfromtxt('./initial_data/salimdemet2_3.csv', delimiter=',')[1:,1:], np.genfromtxt('./initial_data/salimdemet2_4.csv', delimiter=',')[1:,1:],
                np.genfromtxt('./initial_data/salimdemet3_0.csv', delimiter=',')[1:,1:], np.genfromtxt('./initial_data/salimdemet3_1.csv', delimiter=',')[1:,1:], 
                np.genfromtxt('./initial_data/salimdemet3_2.csv', delimiter=',')[1:,1:], np.genfromtxt('./initial_data/salimdemet3_3.csv', delimiter=',')[1:,1:],
                np.genfromtxt('./initial_data/salimdemet3_4.csv', delimiter=',')[1:,1:], np.genfromtxt('./initial_data/salimdemet4_0.csv', delimiter=',')[1:,1:],
                np.genfromtxt('./initial_data/salimdemet4_1.csv', delimiter=',')[1:,1:], np.genfromtxt('./initial_data/salimdemet4_2.csv', delimiter=',')[1:,1:],
                np.genfromtxt('./initial_data/salimdemet4_3.csv', delimiter=',')[1:,1:], np.genfromtxt('./initial_data/salimdemet4_4.csv', delimiter=',')[1:,1:]], dtype=object)


    v_Y = np.array([np.genfromtxt('./initial_data/salimdemet5_0.csv', delimiter=',')[1:,1:], np.genfromtxt('./initial_data/salimdemet5_1.csv', delimiter=',')[1:,1:], 
                    np.genfromtxt('./initial_data/salimdemet5_2.csv', delimiter=',')[1:,1:], np.genfromtxt('./initial_data/salimdemet5_3.csv', delimiter=',')[1:,1:],
                    np.genfromtxt('./initial_data/salimdemet5_4.csv', delimiter=',')[1:,1:], np.genfromtxt('./initial_data/salimdemet6_0.csv', delimiter=',')[1:,1:],
                    np.genfromtxt('./initial_data/salimdemet6_1.csv', delimiter=',')[1:,1:], np.genfromtxt('./initial_data/salimdemet6_2.csv', delimiter=',')[1:,1:],
                    np.genfromtxt('./initial_data/salimdemet6_3.csv', delimiter=',')[1:,1:], np.genfromtxt('./initial_data/salimdemet6_4.csv', delimiter=',')[1:,1:]], dtype=object)

    length1 = len(max(Y, key=len))
    length2 = len(max(v_Y, key=len))
    length = max([length1,length2])

    time_array = np.linspace(0.5, 0.5 + (length-1)*0.01, length)

    gamma = [[43.1,20.2, 0, 88.5, 21.1,0], [33.6,9.6, 0,88.5, 21.1,0], [33.3,32.0,0,88.5, 21.1,0], [54.9,32.5,0,88.5, 21.1,0], [55.3,9.2,0,88.5, 21.1,0], [66.2,22.4,0,88.5, 21.1,0]]

    X = np.array([], dtype=object)
    for i in range(0,20): #gamma is a list type, not numpy array
        e = gamma[i//5]
        x_axis = np.array([np.array([t]+e, dtype=np.float32) for t in time_array], dtype=np.float32)
        X = np.append(X, x_axis)
        
        
    v_X = np.array([])
    for i in range(0,10): #gamma is a list type, not numpy array
        e = gamma[i//5]
        x_axis = np.array([np.array([t]+ e, dtype=np.float32) for t in time_array], dtype=np.float32)
        v_X = np.append(v_X, x_axis)


    Y = np.array([interpolateJointData(element, length) for element in Y])
    Y = Y.reshape(20,length,16).astype(np.float32)

    v_X = v_X.reshape(10,length,7)
    X = X.reshape(20, length, 7)

    v_Y = np.array([interpolateJointData(element, length) for element in v_Y])
    v_Y = v_Y.reshape(10,length, 16).astype(np.float32)

    merged_array_training = np.concatenate((X, Y), axis=2)
    merged_array_validation = np.concatenate((v_X, v_Y), axis=2)
    merged_array_training = merged_array_training.tolist()
    merged_array_validation = merged_array_validation.tolist()

    dataset_train = {   
        "data_0":torch.tensor(merged_array_training[0]), "data_1": torch.tensor(merged_array_training[1]), "data_2": torch.tensor(merged_array_training[2]),
        "data_3":torch.tensor(merged_array_training[3]), "data_4": torch.tensor(merged_array_training[4]), "data_5": torch.tensor(merged_array_training[5]),
        "data_6":torch.tensor(merged_array_training[6]), "data_7": torch.tensor(merged_array_training[7]), "data_8": torch.tensor(merged_array_training[8]),
        "data_9":torch.tensor(merged_array_training[9]), "data_10": torch.tensor(merged_array_training[10]), "data_11": torch.tensor(merged_array_training[11]),
        "data_12":torch.tensor(merged_array_training[12]), "data_13": torch.tensor(merged_array_training[13]), "data_14": torch.tensor(merged_array_training[14]),
        "data_15":torch.tensor(merged_array_training[15]), "data_16": torch.tensor(merged_array_training[16]), "data_17": torch.tensor(merged_array_training[17]),
        "data_18":torch.tensor(merged_array_training[18]), "data_19": torch.tensor(merged_array_training[19])
                    }
    
    dataset_validation = {
        "data_0":torch.tensor(merged_array_validation[0]), "data_1": torch.tensor(merged_array_validation[1]), "data_2": torch.tensor(merged_array_validation[2]),
        "data_3":torch.tensor(merged_array_validation[3]), "data_4": torch.tensor(merged_array_validation[4]), "data_5": torch.tensor(merged_array_validation[5]),
        "data_6":torch.tensor(merged_array_validation[6]), "data_7": torch.tensor(merged_array_validation[7]), "data_8": torch.tensor(merged_array_validation[8]),
        "data_9":torch.tensor(merged_array_validation[9])
    }

    
    torch.save(dataset_train, "train_dataset.pt")
    torch.save(dataset_validation, "validate_dataset.pt")


form_pt_dictionary_from_csv_dataset()


'''