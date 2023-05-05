"""Test and plot the predictions"""
import argparse
import os

import data
import utils

parser = argparse.ArgumentParser("Test script. See README.md for more details.")
parser.add_argument("--config", help="path to config file", type=str, default="config.yaml")
parser.add_argument("--use_wandb", help="whether to use wandb", type=bool, default=False)
args = parser.parse_args()

config = utils.parse_and_init(args)
model = utils.create_model_from_config(config, resume=True, use_wandb=args.use_wandb)
model.to(config["device"])
model.eval()
for param in model.parameters():
    param.requires_grad = False

# load test set
trainset = data.BaxterDemonstrationDataset(config["trainset_path"])
testset = data.BaxterDemonstrationDataset(config["testset_path"])


utils.plot_trajs(trainset, "train", model, config,
                 save_path=os.path.join(config["save_folder"], "train_predictions.pt"))
utils.plot_trajs(testset, "test", model, config,
                 save_path=os.path.join(config["save_folder"], "test_predictions.pt"))
