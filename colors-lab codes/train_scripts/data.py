import torch
from torch.nn.utils.rnn import pad_sequence

PI = 3.141592653589793


class TwoSkillsDataset:
    def __init__(self, N, freq_shift_noise=0.2, amp_shift_noise=0.1, val_split=0.2):
        self.N = int(N*(1-val_split))
        b = 1 - freq_shift_noise/2

        x = torch.linspace(0, PI*2, 200).repeat(N//2, 1) * (torch.rand(N//2, 1) * freq_shift_noise + b) + \
            torch.randn(N//2, 1)*0.03
        y = torch.sin(x)*0.2 + torch.randn(N//2, 1) * amp_shift_noise
        data = torch.stack([x, y], dim=-1)
        x = torch.linspace(0, PI*2, 200).repeat(N//2, 1) * (torch.rand(N//2, 1) * freq_shift_noise + b) + \
            torch.randn(N//2, 1)*0.03
        y = torch.sin(x)*0.2 + torch.randn(N//2, 1) * amp_shift_noise
        data = torch.cat([data, torch.stack([y.flip(1), x.flip(1)], dim=-1)], dim=0)
        data = torch.cat([torch.linspace(0, 1, 200).repeat(data.shape[0], 1).unsqueeze(2), data], dim=-1)
        self.data = data

    def get_sample(self, batch_size=1, max_context=10, max_target=10):
        context_all, target_all, context_mask, target_mask = [], [], [], []
        for _ in range(batch_size):
            n_context = torch.randint(1, max_context, ())
            n_target = torch.randint(1, max_target, ())
            idx = torch.randint(0, self.N, ())
            traj = self.data[idx]
            R = torch.randperm(traj.shape[0])
            context = traj[R[:n_context]]
            target = traj[R[:(n_context+n_target)]]
            context_all.append(context)
            target_all.append(target)
            context_mask.append(torch.ones(context.shape[0]))
            target_mask.append(torch.ones(target.shape[0]))
        context_all = pad_sequence(context_all, batch_first=True)
        target_all = pad_sequence(target_all, batch_first=True)
        context_mask = pad_sequence(context_mask, batch_first=True)
        target_mask = pad_sequence(target_mask, batch_first=True)
        return context_all, target_all, context_mask, target_mask


class BaxterDemonstrationDataset(torch.utils.data.Dataset):
    def __init__(self, path, max_context=10, max_target=10):
        data_dict = torch.load(path)
        self.data = []
        self.names = []
        for key, value in data_dict.items():
            self.data.append(value)
            self.names.append(key)
        self.max_context = max_context
        self.max_target = max_target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traj = self.data[idx]
        n_context = torch.randint(3, self.max_context, ())
        n_target = torch.randint(3, self.max_target, ())
        R = torch.randperm(traj.shape[0])
        context = traj[R[:n_context]]
        target = traj[R[:(n_context+n_target)]]
        return context, target


def unequal_collate(batch):
    context_all, target_all, context_mask, target_mask = [], [], [], []
    for context, target in batch:
        context_all.append(context)
        target_all.append(target)
        context_mask.append(torch.ones(context.shape[0]))
        target_mask.append(torch.ones(target.shape[0]))
    context_all = pad_sequence(context_all, batch_first=True)
    target_all = pad_sequence(target_all, batch_first=True)
    context_mask = pad_sequence(context_mask, batch_first=True)
    target_mask = pad_sequence(target_mask, batch_first=True)
    return context_all, target_all, context_mask, target_mask
