import dqn
import torch.nn as nn
import torch.optim
import matplotlib.pyplot as plt
import time, datetime

import settings

# experiment 4
# modify optimizer
episodes = 101
exploration_rate_decay = 0.99996
loggers = {}
name = "04"


start = time.time()
print("///////////////////////////////////////////////////////////////////////")
print("//// STARTING EXPERIMENT: RUNNING WITH ADAM OPTIMIZER")
print("///////////////////////////////////////////////////////////////////////")
loggers["optim=adam"] = dqn.run(
    episodes=episodes,
    gamma=0.5,
    prefix=f"{name}--optim-adam",
    exploration_rate_decay=exploration_rate_decay,
)
print("///////////////////////////////////////////////////////////////////////")
print("//// ENDING EXPERIMENT: RUNNING WITH ADAM OPTIMIZER")
print("//// RUN TIME: ", (time.time() - start) / 60, " minutes")
print("///////////////////////////////////////////////////////////////////////")

start = time.time()
print("///////////////////////////////////////////////////////////////////////")
print("//// STARTING EXPERIMENT: RUNNING WITH SGD OPTIMIZER")
print("///////////////////////////////////////////////////////////////////////")
loggers["optim=sgd"] = dqn.run(
    episodes=episodes,
    gamma=0.5,
    prefix=f"{name}--optim-sgd",
    exploration_rate_decay=exploration_rate_decay,
    optimizer=torch.optim.SGD,
)
print("///////////////////////////////////////////////////////////////////////")
print("//// ENDING EXPERIMENT: RUNNING WITH SGD OPTIMIZER")
print("//// RUN TIME: ", (time.time() - start) / 60, " minutes")
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
