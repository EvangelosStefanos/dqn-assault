import dqn
import matplotlib.pyplot as plt
import time
import settings
import shared

# experiment 2
# modify gamma, (low, middle, high)
episodes = 201
exploration_rate_decay = 0.99996
loggers = {}
name = "02"


# low gamma


start = time.time()
print("///////////////////////////////////////////////////////////////////////")
print("//// STARTING EXPERIMENT: RUNNING WITH VERY LOW GAMMA")
print("///////////////////////////////////////////////////////////////////////")
loggers["gamma=.001"] = dqn.run(
    episodes=episodes,
    gamma=0.001,
    prefix=f"{name}--gamma-.001",
    exploration_rate_decay=exploration_rate_decay,
)
print("///////////////////////////////////////////////////////////////////////")
print("//// ENDING EXPERIMENT: RUNNING WITH VERY LOW GAMMA")
print(f"//// RUN TIME: {shared.runtime(start=start)}")
print("///////////////////////////////////////////////////////////////////////")

start = time.time()
print("///////////////////////////////////////////////////////////////////////")
print("//// STARTING EXPERIMENT: RUNNING WITH LOW GAMMA")
print("///////////////////////////////////////////////////////////////////////")
loggers["gamma=.1"] = dqn.run(
    episodes=episodes,
    gamma=0.1,
    prefix=f"{name}--gamma-.1",
    exploration_rate_decay=exploration_rate_decay,
)
print("///////////////////////////////////////////////////////////////////////")
print("//// ENDING EXPERIMENT: RUNNING WITH LOW GAMMA")
print(f"//// RUN TIME: {shared.runtime(start=start)}")
print("///////////////////////////////////////////////////////////////////////")


# middle gamma


start = time.time()
print("///////////////////////////////////////////////////////////////////////")
print("//// STARTING EXPERIMENT: RUNNING WITH MID GAMMA")
print("///////////////////////////////////////////////////////////////////////")
loggers["gamma=.5"] = dqn.run(
    episodes=episodes,
    gamma=0.5,
    prefix=f"{name}--gamma-.5",
    exploration_rate_decay=exploration_rate_decay,
)
print("///////////////////////////////////////////////////////////////////////")
print("//// ENDING EXPERIMENT: RUNNING WITH MID GAMMA")
print(f"//// RUN TIME: {shared.runtime(start=start)}")
print("///////////////////////////////////////////////////////////////////////")


# high gamma # default gamma is 0.9


start = time.time()
print("///////////////////////////////////////////////////////////////////////")
print("//// STARTING EXPERIMENT: RUNNING WITH HIGH GAMMA")
print("///////////////////////////////////////////////////////////////////////")
loggers["gamma=.9"] = dqn.run(
    episodes=episodes,
    prefix=f"{name}--gamma-.9",
    gamma=0.9,
    exploration_rate_decay=exploration_rate_decay,
)
print("///////////////////////////////////////////////////////////////////////")
print("//// ENDING EXPERIMENT: RUNNING WITH HIGH GAMMA")
print(f"//// RUN TIME: {shared.runtime(start=start)}")
print("///////////////////////////////////////////////////////////////////////")

start = time.time()
print("///////////////////////////////////////////////////////////////////////")
print("//// STARTING EXPERIMENT: RUNNING WITH VERY HIGH GAMMA")
print("///////////////////////////////////////////////////////////////////////")
loggers["gamma=.999"] = dqn.run(
    episodes=episodes,
    prefix=f"{name}--gamma-.999",
    gamma=0.999,
    exploration_rate_decay=exploration_rate_decay,
)
print("///////////////////////////////////////////////////////////////////////")
print("//// ENDING EXPERIMENT: RUNNING WITH VERY HIGH GAMMA")
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
