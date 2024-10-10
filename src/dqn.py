# based on https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
# upgraded from gym to gymnasium v0.29.1
# rendering: https://stackoverflow.com/questions/77042526/how-to-record-and-save-video-of-gym-environment/77783510#77783510


def run(
    episodes=1000,
    exploration_rate=None,
    exploration_rate_decay=None,
    exploration_rate_min=None,
    gamma=None,
    optimizer=None,
    net=None,
    memory=None,
    prefix="",
    render=False,
):
    import torch
    from torch import nn
    from torchvision import transforms as T
    from PIL import Image
    import numpy as np
    from pathlib import Path
    from collections import deque
    import random, datetime, os, copy

    # Gym is an OpenAI toolkit for RL
    import gymnasium as gym
    from gymnasium.spaces import Box
    from gymnasium.wrappers import FrameStack

    from torchinfo import summary

    OUTPUT_DIR = "output"
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    OUTPUT_VIDEO_DIR = "output/videos"
    Path(OUTPUT_VIDEO_DIR).mkdir(exist_ok=True)

    # Initialize assault environment
    if render:
        env = gym.make("Assault-v4", render_mode="rgb_array")
        env.metadata["render_fps"] = 30
        env = gym.wrappers.RecordVideo(
            env=env,
            video_folder=OUTPUT_VIDEO_DIR,
            name_prefix=prefix,
            episode_trigger=lambda x: x % episodes == 0,
        )
    else:
        env = gym.make("Assault-v4")

    env.reset()

    if render:
        env.start_video_recorder()

    next_state, reward, terminated, truncated, info = env.step(action=0)
    done = terminated or truncated
    print(f"{next_state.shape},\n {reward},\n {done},\n {info}")
    # height = 210, width = 160, color = 3
    # print(env.action_space, env.action_space.n, env.get_action_meanings())

    # In[2]:

    class SkipFrame(gym.Wrapper):
        def __init__(self, env, skip):
            """Return only every `skip`-th frame"""
            super().__init__(env)
            self._skip = skip

        def step(self, action):
            """Repeat action, and sum reward"""
            total_reward = 0.0
            done = False
            for i in range(self._skip):
                # Accumulate reward and repeat the same action
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                if done:
                    break
            return obs, total_reward, terminated, truncated, info

    class GrayScaleObservation(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            obs_shape = self.observation_space.shape[:2]
            self.observation_space = Box(
                low=0, high=255, shape=obs_shape, dtype=np.uint8
            )

        def permute_orientation(self, observation):
            # permute [H, W, C] array to [C, H, W] tensor
            observation = np.transpose(observation, (2, 0, 1))
            observation = torch.tensor(observation.copy(), dtype=torch.float)
            return observation

        def observation(self, observation):
            observation = self.permute_orientation(observation)
            transform = T.Grayscale()
            observation = transform(observation)
            return observation

    class ResizeObservation(gym.ObservationWrapper):
        def __init__(self, env, shape):
            super().__init__(env)
            if isinstance(shape, int):
                self.shape = (shape, shape)
            else:
                self.shape = tuple(shape)

            obs_shape = self.shape + self.observation_space.shape[2:]
            self.observation_space = Box(
                low=0, high=255, shape=obs_shape, dtype=np.uint8
            )

        def observation(self, observation):
            transforms = T.Compose([T.Resize(self.shape), T.Normalize(0, 255)])
            observation = transforms(observation).squeeze(0)
            return observation

    # Apply Wrappers to environment
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)

    # In[3]:

    class Mario:
        def __init__(self, state_dim, action_dim, save_dir):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.save_dir = save_dir

            self.use_cuda = torch.cuda.is_available()

            # Mario's DNN to predict the most optimal action - we implement this in the Learn section
            self.net = MarioNet(self.state_dim, self.action_dim).float()
            if self.use_cuda:
                self.net = self.net.to(device="cuda")

            self.exploration_rate = 1
            if exploration_rate is not None:
                self.exploration_rate = exploration_rate

            self.exploration_rate_decay = 0.99999975
            if exploration_rate_decay is not None:
                self.exploration_rate_decay = exploration_rate_decay

            self.exploration_rate_min = 0.1
            if exploration_rate_min is not None:
                self.exploration_rate_min = exploration_rate_min
            self.curr_step = 0

            self.save_every = 5e5  # no. of experiences between saving Mario Net

        def act(self, state):
            """
            Given a state, choose an epsilon-greedy action and update value of step.

            Inputs:
            state(LazyFrame): A single observation of the current state, dimension is (state_dim)
            Outputs:
            action_idx (int): An integer representing which action Mario will perform
            """
            # EXPLORE
            if np.random.rand() < self.exploration_rate:
                action_idx = np.random.randint(self.action_dim)

            # EXPLOIT
            else:
                state = state.__array__()
                if self.use_cuda:
                    state = torch.tensor(state).cuda()
                else:
                    state = torch.tensor(state)
                state = state.unsqueeze(0)
                action_values = self.net(state, model="online")
                action_idx = torch.argmax(action_values, axis=1).item()

            # decrease exploration_rate
            self.exploration_rate *= self.exploration_rate_decay
            self.exploration_rate = max(
                self.exploration_rate_min, self.exploration_rate
            )

            # increment step
            self.curr_step += 1
            return action_idx

    # In[4]:

    class Mario(Mario):  # subclassing for continuity
        def __init__(self, state_dim, action_dim, save_dir):
            super().__init__(state_dim, action_dim, save_dir)
            self.memory = deque(maxlen=1000)
            if memory is not None:
                self.memory = deque(maxlen=memory)
            self.batch_size = 32

        def cache(self, state, next_state, action, reward, done):
            """
            Store the experience to self.memory (replay buffer)

            Inputs:
            state (LazyFrame),
            next_state (LazyFrame),
            action (int),
            reward (float),
            done(bool))
            """
            state = state.__array__()
            next_state = next_state.__array__()

            if self.use_cuda:
                state = torch.tensor(state).cuda()
                next_state = torch.tensor(next_state).cuda()
                action = torch.tensor([action]).cuda()
                reward = torch.tensor([reward]).cuda()
                done = torch.tensor([done]).cuda()
            else:
                state = torch.tensor(state)
                next_state = torch.tensor(next_state)
                action = torch.tensor([action])
                reward = torch.tensor([reward])
                done = torch.tensor([done])

            self.memory.append(
                (
                    state,
                    next_state,
                    action,
                    reward,
                    done,
                )
            )

        def recall(self):
            """
            Retrieve a batch of experiences from memory
            """
            batch = random.sample(self.memory, self.batch_size)
            state, next_state, action, reward, done = map(torch.stack, zip(*batch))
            return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    # In[5]:

    class MarioNet(nn.Module):
        """mini cnn structure
        input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
        """

        def __init__(self, input_dim, output_dim):
            super().__init__()
            c, h, w = input_dim

            if h != 84:
                raise ValueError(f"Expecting input height: 84, got: {h}")
            if w != 84:
                raise ValueError(f"Expecting input width: 84, got: {w}")

            self.online = nn.Sequential(
                # in 4 x 84 x 84 -> 32 x 20 x 20 -> 64 x 9 x 9 -> 64 x 7 x 7
                nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(3136, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim),
            )
            if net is not None:
                self.online = net

            self.target = copy.deepcopy(self.online)

            # Q_target parameters are frozen.
            for p in self.target.parameters():
                p.requires_grad = False

        def forward(self, input, model):
            if model == "online":
                return self.online(input)
            elif model == "target":
                return self.target(input)

    # In[6]:

    class Mario(Mario):
        def __init__(self, state_dim, action_dim, save_dir):
            super().__init__(state_dim, action_dim, save_dir)
            self.gamma = 0.9
            if gamma is not None:
                self.gamma = gamma

        def td_estimate(self, state, action):
            current_Q = self.net(state, model="online")[
                np.arange(0, self.batch_size), action
            ]  # Q_online(s,a)
            return current_Q

        @torch.no_grad()
        def td_target(self, reward, next_state, done):
            next_state_Q = self.net(next_state, model="online")
            best_action = torch.argmax(next_state_Q, axis=1)
            next_Q = self.net(next_state, model="target")[
                np.arange(0, self.batch_size), best_action
            ]
            return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    # In[7]:

    class Mario(Mario):
        def __init__(self, state_dim, action_dim, save_dir):
            super().__init__(state_dim, action_dim, save_dir)
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
            if optimizer is not None:
                self.optimizer = optimizer(self.net.parameters(), lr=0.00025)
            self.loss_fn = torch.nn.SmoothL1Loss()

        def update_Q_online(self, td_estimate, td_target):
            loss = self.loss_fn(td_estimate, td_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.item()

        def sync_Q_target(self):
            self.net.target.load_state_dict(self.net.online.state_dict())

    # In[8]:

    class Mario(Mario):
        def save(self):
            save_path = (
                self.save_dir
                / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
            )
            torch.save(
                dict(
                    model=self.net.state_dict(), exploration_rate=self.exploration_rate
                ),
                save_path,
            )
            print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    # In[9]:

    class Mario(Mario):
        def __init__(self, state_dim, action_dim, save_dir):
            super().__init__(state_dim, action_dim, save_dir)
            self.burnin = 1e4  # min. experiences before training
            self.learn_every = 3  # no. of experiences between updates to Q_online
            self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

        def learn(self):
            if self.curr_step % self.sync_every == 0:
                self.sync_Q_target()

            if self.curr_step % self.save_every == 0:
                self.save()

            if self.curr_step < self.burnin:
                return None, None

            if self.curr_step % self.learn_every != 0:
                return None, None

            # Sample from memory
            state, next_state, action, reward, done = self.recall()

            # Get TD Estimate
            td_est = self.td_estimate(state, action)

            # Get TD Target
            td_tgt = self.td_target(reward, next_state, done)

            # Backpropagate loss through Q_online
            loss = self.update_Q_online(td_est, td_tgt)

            return (td_est.mean().item(), loss)

    # In[10]:

    import numpy as np
    import time, datetime
    import matplotlib.pyplot as plt

    class MetricLogger:
        def __init__(self, save_dir):
            self.save_log = save_dir / "log"
            with open(self.save_log, "w") as f:
                f.write(
                    f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                    f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                    f"{'TimeDelta':>15}{'Time':>20}\n"
                )
            self.ep_rewards_plot = save_dir / "reward_plot.jpg"
            self.ep_lengths_plot = save_dir / "length_plot.jpg"
            self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
            self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

            # History metrics
            self.ep_rewards = []
            self.ep_lengths = []
            self.ep_avg_losses = []
            self.ep_avg_qs = []
            self.eps = []

            # Moving averages, added for every call to record()
            self.moving_avg_ep_rewards = []
            self.moving_avg_ep_lengths = []
            self.moving_avg_ep_avg_losses = []
            self.moving_avg_ep_avg_qs = []

            # Current episode metric
            self.init_episode()

            # Timing
            self.record_time = time.time()

        def log_step(self, reward, loss, q):
            self.curr_ep_reward += reward
            self.curr_ep_length += 1
            if loss:
                self.curr_ep_loss += loss
                self.curr_ep_q += q
                self.curr_ep_loss_length += 1

        def log_episode(self):
            "Mark end of episode"
            self.ep_rewards.append(self.curr_ep_reward)
            self.ep_lengths.append(self.curr_ep_length)
            if self.curr_ep_loss_length == 0:
                ep_avg_loss = 0
                ep_avg_q = 0
            else:
                ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
                ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
            self.ep_avg_losses.append(ep_avg_loss)
            self.ep_avg_qs.append(ep_avg_q)

            self.init_episode()

        def init_episode(self):
            self.curr_ep_reward = 0.0
            self.curr_ep_length = 0
            self.curr_ep_loss = 0.0
            self.curr_ep_q = 0.0
            self.curr_ep_loss_length = 0

        def record(self, episode, epsilon, step):
            mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
            mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
            mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
            mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
            self.moving_avg_ep_rewards.append(mean_ep_reward)
            self.moving_avg_ep_lengths.append(mean_ep_length)
            self.moving_avg_ep_avg_losses.append(mean_ep_loss)
            self.moving_avg_ep_avg_qs.append(mean_ep_q)
            self.eps.append(episode)

            last_record_time = self.record_time
            self.record_time = time.time()
            time_since_last_record = np.round(self.record_time - last_record_time, 3)

            print(
                f"Episode {episode} - "
                f"Step {step} - "
                f"Epsilon {epsilon} - "
                f"Mean Reward {mean_ep_reward} - "
                f"Mean Length {mean_ep_length} - "
                f"Mean Loss {mean_ep_loss} - "
                f"Mean Q Value {mean_ep_q} - "
                f"Time Delta {time_since_last_record} - "
                f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
            )

            with open(self.save_log, "a") as f:
                f.write(
                    f"{episode:8d}{step:8d}{epsilon:10.3f}"
                    f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                    f"{time_since_last_record:15.3f}"
                    f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
                )

            for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs"]:
                plt.plot(getattr(self, f"moving_avg_{metric}"))
                plt.savefig(getattr(self, f"{metric}_plot"))
                plt.clf()

    # In[14]:

    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print()

    save_dir = Path("output/checkpoints") / (
        str(prefix) + "_" + datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    )
    save_dir.mkdir(parents=True)

    mario = Mario(
        state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir
    )
    summary(mario.net.online, input_size=(1, 4, 84, 84))
    logger = MetricLogger(save_dir)

    starting_episode = 0
    for e in range(starting_episode, starting_episode + episodes):

        state, info = env.reset()

        # Play the game!
        while True:
            # import time
            # time.sleep(.1)

            # Run agent on the state
            action = mario.act(state)

            # Agent performs action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if render:
                env.render()

            # Remember
            mario.cache(state, next_state, action, reward, done)

            # Learn
            q, loss = mario.learn()

            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            state = next_state

            # Check if end of game
            if done:  # or info["flag_get"]:
                break

        logger.log_episode()
        if e % 5 == 0:
            logger.record(
                episode=e, epsilon=mario.exploration_rate, step=mario.curr_step
            )

    mario.save()
    if render:
        env.close_video_recorder()
    # In[ ]:
    # Run for 1 more episode with render support
    if episodes > 1:
        run(
            episodes=1,
            exploration_rate=exploration_rate,
            exploration_rate_decay=exploration_rate_decay,
            exploration_rate_min=exploration_rate_min,
            gamma=gamma,
            optimizer=optimizer,
            net=net,
            memory=memory,
            prefix=prefix,
            render=True,
        )
    return logger
