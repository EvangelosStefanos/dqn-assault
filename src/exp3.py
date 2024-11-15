import dqn
import matplotlib.pyplot as plt
import time
import settings
import shared

# experiment 3
# exploration (low, middle, high)
episodes = 201
loggers = {}
name = "03"


# high exploration
start = time.time()
print("///////////////////////////////////////////////////////////////////////")
print("//// STARTING EXPERIMENT: RUNNING WITH HIGH EXPLORATION RATE")
print("///////////////////////////////////////////////////////////////////////")
loggers["exp_rate=high"] = dqn.run(
    episodes=episodes,
    exploration_rate=1,
    exploration_rate_decay=0.99995,
    prefix=f"{name}--exp-rate-high",
)
print("///////////////////////////////////////////////////////////////////////")
print("//// ENDING EXPERIMENT: RUNNING WITH HIGH EXPLORATION RATE")
print(f"//// RUN TIME: {shared.runtime(start=start)}")
print("///////////////////////////////////////////////////////////////////////")

# middle exploration
start = time.time()
print("///////////////////////////////////////////////////////////////////////")
print("//// STARTING EXPERIMENT: RUNNING WITH MID EXPLORATION RATE")
print("///////////////////////////////////////////////////////////////////////")
loggers["exp_rate=middle"] = dqn.run(
    episodes=episodes,
    exploration_rate=0.5,
    exploration_rate_decay=0.99998,
    prefix=f"{name}--exp-rate-middle",
)
print("///////////////////////////////////////////////////////////////////////")
print("//// ENDING EXPERIMENT: RUNNING WITH MID EXPLORATION RATE")
print(f"//// RUN TIME: {shared.runtime(start=start)}")
print("///////////////////////////////////////////////////////////////////////")

# low exploration
start = time.time()
print("///////////////////////////////////////////////////////////////////////")
print("//// STARTING EXPERIMENT: RUNNING WITH LOW EXPLORATION RATE")
print("///////////////////////////////////////////////////////////////////////")
loggers["exp_rate=low"] = dqn.run(
    episodes=episodes,
    exploration_rate=0.2,
    exploration_rate_decay=0.99999,
    prefix=f"{name}--exp-rate-low",
)
print("///////////////////////////////////////////////////////////////////////")
print("//// ENDING EXPERIMENT: RUNNING WITH LOW EXPLORATION RATE")
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
