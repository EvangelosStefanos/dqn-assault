import dqn
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import settings
import shared

# experiment 5
# modify cnn architectures
episodes = 1001
exploration_rate_decay = 0.99996
loggers = {}
name = "05"


# default architecture


net = None
start = time.time()
print("///////////////////////////////////////////////////////////////////////")
print("//// STARTING EXPERIMENT: RUNNING WITH DEFAULT NETWORK ARCHITECTURE")
print("///////////////////////////////////////////////////////////////////////")
loggers["net=1"] = dqn.run(
    episodes=episodes,
    gamma=0.5,
    prefix=f"{name}--default-nnet",
    exploration_rate_decay=exploration_rate_decay,
    net=net,
)
print("///////////////////////////////////////////////////////////////////////")
print("//// ENDING EXPERIMENT: RUNNING WITH DEFAULT NETWORK ARCHITECTURE")
print(f"//// RUN TIME: {shared.runtime(start=start)}")
print("///////////////////////////////////////////////////////////////////////")


# less layers


net = nn.Sequential(
    nn.Conv2d(in_channels=4, out_channels=8, kernel_size=8, stride=4),  # 8x20
    nn.ReLU(),
    nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=4),  # 16x5
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 512),
    nn.ReLU(),
    nn.Linear(512, 7),
)
start = time.time()
print("///////////////////////////////////////////////////////////////////////")
print("//// STARTING EXPERIMENT: RUNNING WITH REDUCED NUMBER OF LAYERS")
print("///////////////////////////////////////////////////////////////////////")
loggers["net=less_layers"] = dqn.run(
    episodes=episodes,
    gamma=0.5,
    prefix=f"{name}--reduced-nnet-layers",
    exploration_rate_decay=exploration_rate_decay,
    net=net,
)
print("///////////////////////////////////////////////////////////////////////")
print("//// ENDING EXPERIMENT: RUNNING WITH REDUCED NUMBER OF LAYERS")
print(f"//// RUN TIME: {shared.runtime(start=start)}")
print("///////////////////////////////////////////////////////////////////////")


# more layers


net = nn.Sequential(
    nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
    nn.ReLU(),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
    nn.ReLU(),
    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(128 * 3 * 3, 512),
    nn.ReLU(),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, 7),
)
start = time.time()
print("///////////////////////////////////////////////////////////////////////")
print("//// STARTING EXPERIMENT: RUNNING WITH INCREASED NUMBER OF LAYERS")
print("///////////////////////////////////////////////////////////////////////")
loggers["net=more_layers"] = dqn.run(
    episodes=episodes,
    gamma=0.5,
    prefix=f"{name}--increased-nnet-layers",
    exploration_rate_decay=exploration_rate_decay,
    net=net,
)
print("///////////////////////////////////////////////////////////////////////")
print("//// ENDING EXPERIMENT: RUNNING WITH INCREASED NUMBER OF LAYERS")
print(f"//// RUN TIME: {shared.runtime(start=start)}")
print("///////////////////////////////////////////////////////////////////////")


# low channels


net = nn.Sequential(
    nn.Conv2d(in_channels=4, out_channels=8, kernel_size=8, stride=4),  # 8x20
    nn.ReLU(),
    nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, stride=2),  # 16x10
    nn.ReLU(),
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=2),  # 32x5
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(32 * 5 * 5, 512),
    nn.ReLU(),
    nn.Linear(512, 7),
)
start = time.time()
print("///////////////////////////////////////////////////////////////////////")
print("//// STARTING EXPERIMENT: RUNNING WITH REDUCED NUMBER OF CHANNELS")
print("///////////////////////////////////////////////////////////////////////")
loggers["net=low_channels"] = dqn.run(
    episodes=episodes,
    gamma=0.5,
    prefix=f"{name}--reduced-nnet-channels",
    exploration_rate_decay=exploration_rate_decay,
    net=net,
)
print("///////////////////////////////////////////////////////////////////////")
print("//// ENDING EXPERIMENT: RUNNING WITH REDUCED NUMBER OF CHANNELS")
print(f"//// RUN TIME: {shared.runtime(start=start)}")
print("///////////////////////////////////////////////////////////////////////")


# high channels


net = nn.Sequential(
    nn.Conv2d(in_channels=4, out_channels=64, kernel_size=8, stride=4),
    nn.ReLU(),
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2),
    nn.ReLU(),
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(256 * 5 * 5, 512),
    nn.ReLU(),
    nn.Linear(512, 7),
)
start = time.time()
print("///////////////////////////////////////////////////////////////////////")
print("//// STARTING EXPERIMENT: RUNNING WITH INCREASED NUMBER OF CHANNELS")
print("///////////////////////////////////////////////////////////////////////")
loggers["net=high_channels"] = dqn.run(
    episodes=episodes,
    gamma=0.5,
    prefix=f"{name}--increased-nnet-channels",
    exploration_rate_decay=exploration_rate_decay,
    net=net,
)
print("///////////////////////////////////////////////////////////////////////")
print("//// ENDING EXPERIMENT: RUNNING WITH INCREASED NUMBER OF CHANNELS")
print(f"//// RUN TIME: {shared.runtime(start=start)}")
print("///////////////////////////////////////////////////////////////////////")
print("\a")

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

for key, logger in loggers.items():
    axs[0].plot(logger.eps, logger.moving_avg_ep_rewards, label=key)
axs[0].legend()
axs[0].set(xlabel="Episode", ylabel="Reward")

for key, logger in loggers.items():
    axs[1].plot(logger.eps, logger.moving_avg_ep_lengths, label=key)
axs[1].legend()
axs[1].set(xlabel="Episode", ylabel="Length")

plt.savefig(f"output/{name}.png")
