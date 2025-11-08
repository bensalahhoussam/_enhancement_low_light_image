import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import math


class WarmupStableDecayLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, stable_steps, decay_steps, warmup_start_lr=1e-5, base_lr=1e-3, final_lr=1e-5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = base_lr
        self.final_lr = final_lr  # Final learning rate after decay
        super(WarmupStableDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warm-up phase: from warmup_start_lr to base_lr
            return [self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * (self.last_epoch / self.warmup_steps) for _ in self.optimizer.param_groups]
        elif self.last_epoch < self.warmup_steps + self.stable_steps:
            # Stable phase: keep learning rate at base_lr
            return [self.base_lr for _ in self.optimizer.param_groups]
        else:
            # Decay phase: cosine annealing from base_lr to final_lr
            progress = (self.last_epoch - self.warmup_steps - self.stable_steps) / self.decay_steps
            # Cosine decay formula adjusted to interpolate between base_lr and final_lr
            return [self.final_lr + 0.5 * (self.base_lr - self.final_lr) * (1 + math.cos(math.pi * progress)) for _ in self.optimizer.param_groups]


def smart_optimizer(model, name="Adam", lr=0.001, momentum=0.9, decay=1e-5):

    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if p_name == "bias":  # bias (no decay)
                g[2].append(p)
            elif p_name == "weight" and isinstance(v, bn):  # weight (no decay)
                g[1].append(p)
            else:
                g[0].append(p)  # weight (with decay)

    if name == "Adam":
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum

    elif name == "RMSProp":
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == "AdamW":
        optimizer = torch.optim.AdamW(g[2], lr=lr, weight_decay=decay,betas=(momentum,0.99))

    elif name == "SGD":
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f"Optimizer {name} not implemented.")

    optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)

    return optimizer


'''import matplotlib.pyplot as plt
model = torch.nn.Linear(10, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)'''
"""epochs = 1000
steps_per_epoch = 400*1000

warmup_steps = int(steps_per_epoch * epochs * 0.2)
stable_steps = int(steps_per_epoch * epochs * 0.1)
decay_steps = int(steps_per_epoch * epochs * 0.7)

# Initialize optimizer and scheduler
scheduler = WarmupStableDecayLR(
    optimizer,
    warmup_steps=warmup_steps,
    stable_steps=stable_steps,
    decay_steps=decay_steps,
    warmup_start_lr=1e-4,
    base_lr=1e-3,
    final_lr=1e-6
)
lrs = []
for epoch in range(epochs):
    for j in range(steps_per_epoch):
        optimizer.step()  # Dummy step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        lrs.append(current_lr)
        print(f"Step {j + 1:02d}: LR = {current_lr:.6f}")




plt.plot(lrs)
plt.title("Warmup → Stable → Cosine Decay LR")
plt.xlabel("Step (Batch)")
plt.ylabel("Learning Rate")
plt.grid(True)
plt.show()"""