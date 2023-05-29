"""Training script"""
import argparse
import os
import time

import torch
import wandb

import data
import utils


parser = argparse.ArgumentParser("Training script. See README.md for more details.")
parser.add_argument("--config", help="path to config file", type=str, default="config.yaml")
args = parser.parse_args()

config = utils.parse_and_init(args)
model = utils.create_model_from_config(config)
model.to(config["device"])

# create a dataset
trainset = data.BaxterDemonstrationDataset(config["trainset_path"],
                                           max_context=config["max_context"],
                                           max_target=config["max_target"])
testset = data.BaxterDemonstrationDataset(config["testset_path"])

loader = torch.utils.data.DataLoader(trainset, batch_size=config["batch_size"],
                                     shuffle=True, collate_fn=data.unequal_collate)


# create an optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

# training loop
avg_loss = 0.0
best_loss = 1e10
it = 0
save_freq = config["save_freq"]
test_freq = config["test_freq"]
start = time.time()
while it < config["max_iter"]:
    for batch_idx, (context, target, context_mask, target_mask) in enumerate(loader):
        # compute the loss
        loss = model.nll_loss(observation=context.to(config["device"]),
                              target=target[..., config["context_dims"]].to(config["device"]),
                              target_truth=target[..., config["target_dims"]].to(config["device"]),
                              observation_mask=context_mask.to(config["device"]),
                              target_mask=target_mask.to(config["device"]))
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update the average loss
        avg_loss += loss.item()
        it += 1

        if it % save_freq == 0:
            avg_loss /= save_freq
            wandb.log({"loss": avg_loss, "iteration": it})

            # save the model
            aliases = ["latest"]
            if avg_loss < best_loss:
                best_loss = avg_loss
                aliases.append("best")
            save_path = os.path.join(config["save_folder"], "model.pt")
            utils.save_checkpoint(model, optimizer, it, avg_loss, save_path, aliases=aliases)

            print(f"Iteration={it}, loss={avg_loss:.4f}")
            avg_loss = 0.0

        if it % test_freq == 0:
            utils.plot_trajs(trainset, "train", model, config)
            utils.plot_trajs(testset, "test", model, config)
