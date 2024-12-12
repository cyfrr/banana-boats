import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import gymnasium as gym
from gymnasium import spaces
import pygame
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("lake_trash_collection.log"),
                        logging.StreamHandler()
                    ])


class LakeTrashEnvironment(gym.Env):
    """
    Custom gym environment simulating a lake with trash collection challenges
    """

    def __init__(self,
                 lake_size=(800, 600),  # Lake dimensions
                 num_trash_items=50,  # Initial number of trash items
                 max_steps=1000):  # Maximum steps per episode
        super().__init__()

        # Environment parameters
        self.lake_size = lake_size
        self.num_trash_items = num_trash_items
        self.max_steps = max_steps

        # Define action and observation spaces
        # Actions: [move_x, move_y, pickup_trash]
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # Observations include:
        # - Robot position (x, y)
        # - Trash positions
        # - Wind and wave factors
        # - Remaining capacity
        obs_size = 2 + (num_trash_items * 3) + 2 + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        # Simulation state variables
        self.robot_position = None
        self.trash_items = None
        self.wind_factor = None
        self.wave_factor = None
        self.current_step = 0
        self.cargo_capacity = 10  # Max number of trash items robot can carry
        self.drop_off_point = (lake_size[0] // 2, lake_size[1] + 50)  # Outside lake bottom

        # Rendering
        self.screen = None

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)

        # Reset robot position to center of lake
        self.robot_position = np.array([
            self.lake_size[0] // 2,
            self.lake_size[1] // 2
        ], dtype=np.float32)

        # Generate random trash items with varying sizes and positions
        self.trash_items = []
        for _ in range(self.num_trash_items):
            trash_size = np.random.uniform(0.1, 5.0)  # Size variation
            trash_pos = np.random.uniform(0, self.lake_size[0]), np.random.uniform(0, self.lake_size[1])
            self.trash_items.append({
                'position': np.array(trash_pos, dtype=np.float32),
                'size': trash_size,
                'collected': False
            })

        # Generate environmental factors
        self.wind_factor = np.random.uniform(-1, 1)
        self.wave_factor = np.random.uniform(0, 1)

        # Reset step counter
        self.current_step = 0

        return self._get_observation(), {}

    def step(self, action):
        """
        Execute one step in the environment

        Action space:
        - action[0]: x movement (-1 to 1)
        - action[1]: y movement (-1 to 1)
        - action[2]: pickup intention (-1 to 1)
        """
        # Update robot position with wind and wave influences
        movement = action[:2] * 10  # Scale movement
        movement[0] += self.wind_factor * 2  # Wind influence
        movement[1] += self.wave_factor * 1.5  # Wave influence

        self.robot_position += movement

        # Constrain robot within lake boundaries
        self.robot_position = np.clip(
            self.robot_position,
            [0, 0],
            [self.lake_size[0], self.lake_size[1]]
        )

        # Pickup trash logic
        reward = 0
        current_cargo = sum(1 for item in self.trash_items if item['collected'])

        if action[2] > 0 and current_cargo < self.cargo_capacity:
            # Find closest uncollected trash
            closest_trash = min(
                (item for item in self.trash_items if not item['collected']),
                key=lambda x: np.linalg.norm(x['position'] - self.robot_position),
                default=None
            )

            if closest_trash:
                distance = np.linalg.norm(closest_trash['position'] - self.robot_position)
                if distance < 20:  # Close enough to pickup
                    closest_trash['collected'] = True
                    reward += 10 / (1 + distance)  # Reward depends on efficiency

        # Check if at drop-off point and unload trash
        if np.linalg.norm(self.robot_position - self.drop_off_point) < 50:
            collected_trash = [item for item in self.trash_items if item['collected']]
            for item in collected_trash:
                item['collected'] = False
            reward += len(collected_trash) * 5  # Bonus for successful dropoff

        # Penalize steps and energy consumption
        reward -= 0.1

        # Determine if episode is done
        self.current_step += 1
        done = (
                self.current_step >= self.max_steps or
                all(item['collected'] for item in self.trash_items)
        )

        return self._get_observation(), reward, done, False, {}

    def _get_observation(self):
        """
        Construct observation vector:
        - Robot position (2)
        - Trash positions and status (num_trash_items * 3)
        - Wind and wave factors (2)
        - Remaining cargo capacity (1)
        """
        obs = [
            *self.robot_position,
            *[val
              for item in self.trash_items
              for val in (
                  item['position'][0] if not item['collected'] else -1,
                  item['position'][1] if not item['collected'] else -1,
                  item['size'] if not item['collected'] else 0
              )
              ],
            self.wind_factor,
            self.wave_factor,
            self.cargo_capacity - sum(1 for item in self.trash_items if item['collected'])
        ]
        return np.array(obs, dtype=np.float32)

    def render(self, mode='human'):
        """Render the environment using Pygame"""
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(self.lake_size)
            pygame.display.set_caption("Lake Trash Collection Simulation")

        self.screen.fill((135, 206, 235))  # Sky blue background

        # Render trash
        for trash in self.trash_items:
            if not trash['collected']:
                pygame.draw.circle(
                    self.screen,
                    (139, 69, 19),  # Brown color
                    trash['position'].astype(int),
                    int(trash['size'])
                )

        # Render robot
        pygame.draw.circle(
            self.screen,
            (255, 0, 0),  # Red robot
            self.robot_position.astype(int),
            10
        )

        pygame.display.flip()


class TrashCollectionAgent(nn.Module):
    """Neural Network for trash collection strategy"""

    def __init__(self, observation_size, action_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.network(x)


class DeepQLearningTrainer:
    def __init__(self, env, learning_rate=1e-3):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        # Network and target network
        self.policy_net = TrashCollectionAgent(
            env.observation_space.shape[0],
            env.action_space.shape[0]
        ).to(self.device)
        self.target_net = TrashCollectionAgent(
            env.observation_space.shape[0],
            env.action_space.shape[0]
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Training parameters
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995

    def train(self, num_episodes=200):
        episode_rewards = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state = torch.FloatTensor(state).to(self.device)
            total_reward = 0
            done = False

            while not done:
                # Epsilon-greedy action selection
                epsilon = max(
                    self.epsilon_end,
                    self.epsilon_start * (self.epsilon_decay ** episode)
                )

                if np.random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = self.policy_net(state).cpu().numpy()

                next_state, reward, done, _, _ = self.env.step(action)
                total_reward += reward

                # Convert to tensors
                next_state = torch.FloatTensor(next_state).to(self.device)
                reward = torch.FloatTensor([reward]).to(self.device)

                # TODO: Implement experience replay and Q-learning update

                state = next_state

            episode_rewards.append(total_reward)
            logging.info(f"Episode {episode}: Total Reward = {total_reward}")

            # Periodically update target network
            if episode % 10 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        # Plotting results
        plt.figure(figsize=(10, 5))
        plt.plot(episode_rewards)
        plt.title('Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.savefig('training_rewards.png')
        plt.close()

        logging.info("Training completed. Results saved to training_rewards.png")


def main():
    # Create environment and trainer
    env = LakeTrashEnvironment()
    trainer = DeepQLearningTrainer(env)

    # Train the agent
    trainer.train()


if __name__ == "__main__":
    main()
    import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import pygame


class LakeTrashEnvironment(gym.Env):
    """
    Custom gym environment simulating a lake with trash collection challenges
    """

    def __init__(self,
                 lake_size=(800, 600),  # Lake dimensions
                 num_trash_items=50,  # Initial number of trash items
                 max_steps=1000):  # Maximum steps per episode
        super().__init__()

        # Environment parameters
        self.lake_size = lake_size
        self.num_trash_items = num_trash_items
        self.max_steps = max_steps

        # Define action and observation spaces
        # Actions: [move_x, move_y, pickup_trash]
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # Observations include:
        # - Robot position (x, y)
        # - Trash positions
        # - Wind and wave factors
        # - Remaining capacity
        obs_size = 2 + (num_trash_items * 3) + 2 + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        # Simulation state variables
        self.robot_position = None
        self.trash_items = None
        self.wind_factor = None
        self.wave_factor = None
        self.current_step = 0
        self.cargo_capacity = 10  # Max number of trash items robot can carry
        self.drop_off_point = (lake_size[0] // 2, lake_size[1] + 50)  # Outside lake bottom

        # Rendering
        self.screen = None

    def reset(self):
        """Reset the environment to initial state"""
        # Reset robot position to center of lake
        self.robot_position = np.array([
            self.lake_size[0] // 2,
            self.lake_size[1] // 2
        ], dtype=np.float32)

        # Generate random trash items with varying sizes and positions
        self.trash_items = []
        for _ in range(self.num_trash_items):
            trash_size = np.random.uniform(0.1, 5.0)  # Size variation
            trash_pos = np.random.uniform(0, self.lake_size[0]), np.random.uniform(0, self.lake_size[1])
            self.trash_items.append({
                'position': np.array(trash_pos, dtype=np.float32),
                'size': trash_size,
                'collected': False
            })

        # Generate environmental factors
        self.wind_factor = np.random.uniform(-1, 1)
        self.wave_factor = np.random.uniform(0, 1)

        # Reset step counter
        self.current_step = 0

        return self._get_observation()

    def step(self, action):
        """
        Execute one step in the environment

        Action space:
        - action[0]: x movement (-1 to 1)
        - action[1]: y movement (-1 to 1)
        - action[2]: pickup intention (-1 to 1)
        """
        # Update robot position with wind and wave influences
        movement = action[:2] * 10  # Scale movement
        movement[0] += self.wind_factor * 2  # Wind influence
        movement[1] += self.wave_factor * 1.5  # Wave influence

        self.robot_position += movement

        # Constrain robot within lake boundaries
        self.robot_position = np.clip(
            self.robot_position,
            [0, 0],
            [self.lake_size[0], self.lake_size[1]]
        )

        # Pickup trash logic
        reward = 0
        current_cargo = sum(1 for item in self.trash_items if item['collected'])

        if action[2] > 0 and current_cargo < self.cargo_capacity:
            # Find closest uncollected trash
            closest_trash = min(
                (item for item in self.trash_items if not item['collected']),
                key=lambda x: np.linalg.norm(x['position'] - self.robot_position),
                default=None
            )

            if closest_trash:
                distance = np.linalg.norm(closest_trash['position'] - self.robot_position)
                if distance < 20:  # Close enough to pickup
                    closest_trash['collected'] = True
                    reward += 10 / (1 + distance)  # Reward depends on efficiency

        # Check if at drop-off point and unload trash
        if np.linalg.norm(self.robot_position - self.drop_off_point) < 50:
            collected_trash = [item for item in self.trash_items if item['collected']]
            for item in collected_trash:
                item['collected'] = False
            reward += len(collected_trash) * 5  # Bonus for successful dropoff

        # Penalize steps and energy consumption
        reward -= 0.1

        # Determine if episode is done
        self.current_step += 1
        done = (
                self.current_step >= self.max_steps or
                all(item['collected'] for item in self.trash_items)
        )

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        """
        Construct observation vector:
        - Robot position (2)
        - Trash positions and status (num_trash_items * 3)
        - Wind and wave factors (2)
        - Remaining cargo capacity (1)
        """
        obs = [
            *self.robot_position,
            *[val
              for item in self.trash_items
              for val in (
                  item['position'][0] if not item['collected'] else -1,
                  item['position'][1] if not item['collected'] else -1,
                  item['size'] if not item['collected'] else 0
              )
              ],
            self.wind_factor,
            self.wave_factor,
            self.cargo_capacity - sum(1 for item in self.trash_items if item['collected'])
        ]
        return np.array(obs, dtype=np.float32)

    def render(self, mode='human'):
        """Render the environment using Pygame"""
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(self.lake_size)
            pygame.display.set_caption("Lake Trash Collection Simulation")

        self.screen.fill((135, 206, 235))  # Sky blue background

        # Render trash
        for trash in self.trash_items:
            if not trash['collected']:
                pygame.draw.circle(
                    self.screen,
                    (139, 69, 19),  # Brown color
                    trash['position'].astype(int),
                    int(trash['size'])
                )

        # Render robot
        pygame.draw.circle(
            self.screen,
            (255, 0, 0),  # Red robot
            self.robot_position.astype(int),
            10
        )

        pygame.display.flip()


class TrashCollectionAgent(nn.Module):
    """Neural Network for trash collection strategy"""

    def __init__(self, observation_size, action_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.network(x)


class DeepQLearningTrainer:
    def __init__(self, env, learning_rate=1e-3):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Network and target network
        self.policy_net = TrashCollectionAgent(
            env.observation_space.shape[0],
            env.action_space.shape[0]
        ).to(self.device)
        self.target_net = TrashCollectionAgent(
            env.observation_space.shape[0],
            env.action_space.shape[0]
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Training parameters
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995

    def train(self, num_episodes=200):
        episode_rewards = []

        for episode in range(num_episodes):
            state = torch.FloatTensor(self.env.reset()).to(self.device)
            total_reward = 0
            done = False

            while not done:
                # Epsilon-greedy action selection
                epsilon = max(
                    self.epsilon_end,
                    self.epsilon_start * (self.epsilon_decay ** episode)
                )

                if np.random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = self.policy_net(state).cpu().numpy()

                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward

                # Convert to tensors
                next_state = torch.FloatTensor(next_state).to(self.device)
                reward = torch.FloatTensor([reward]).to(self.device)

                # TODO: Implement experience replay and Q-learning update

                state = next_state

            episode_rewards.append(total_reward)

            # Periodically update target network
            if episode % 10 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        # Plotting results
        plt.figure(figsize=(10, 5))
        plt.plot(episode_rewards)
        plt.title('Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.savefig('training_rewards.png')
        plt.close()


def main():
    # Create environment and trainer
    env = LakeTrashEnvironment()
    trainer = DeepQLearningTrainer(env)

    # Train the agent
    trainer.train()


if __name__ == "__main__":
    main()