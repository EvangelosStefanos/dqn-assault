import dqn
import matplotlib.pyplot as plt
import time
import settings
import shared

# experiment 1
# default parameters
episodes = 201
exploration_rate_decay = 0.99996
loggers = {}
name = "01"


start = time.time()
print("///////////////////////////////////////////////////////////////////////")
print("//// STARTING EXPERIMENT: RUNNING WITH DEFAULT SETTINGS")
print("///////////////////////////////////////////////////////////////////////")
loggers["default"] = dqn.run(
    episodes=episodes,
    exploration_rate_decay=exploration_rate_decay,
    prefix=f"{name}-"
)
print("///////////////////////////////////////////////////////////////////////")
print("//// ENDING EXPERIMENT: RUNNING WITH DEFAULT SETTINGS")
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
