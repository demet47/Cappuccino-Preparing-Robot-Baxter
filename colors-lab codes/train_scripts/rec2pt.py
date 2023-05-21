import sys
import os

import torch


def convert(path, normalizeTime=False):
    with open(path, "r") as f:
        # first line is the header, and the second line does not follow 100hz, so skip.
        lines = f.readlines()[2:]
    lines = [[float(item) for item in line.strip().split(",")] for line in lines]
    lines = torch.tensor(lines, dtype=torch.float32)
    lines[:, 0] = lines[:, 0]
    if normalizeTime:
        lines[:, 0] = torch.linspace(0, 1, lines.shape[0])
    return lines


folder = sys.argv[1]
output = sys.argv[2]
normalizeTime = bool(int(sys.argv[3]))
print(normalizeTime)
data = {}
for file in os.listdir(folder):
    if file.endswith(".csv"):
        path = os.path.join(folder, file)
        data[file] = convert(path, normalizeTime)
print(data)
torch.save(data, os.path.join(folder, output))
