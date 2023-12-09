# Import necessary libraries
import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.cliffwalking import CliffWalkingEnv
from gymnasium.error import DependencyNotInstalled
from os import path
from time import time

# Do not change this class
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
image_path = path.join(path.dirname(gym.__file__), "envs", "toy_text")


class CliffWalking(CliffWalkingEnv):
    def __init__(self, is_hardMode=True, num_cliffs=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_hardMode = is_hardMode

        # Generate random cliff positions
        if self.is_hardMode:
            self.num_cliffs = num_cliffs
            self._cliff = np.zeros(self.shape, dtype=bool)
            self.start_state = (3, 0)
            self.terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
            self.cliff_positions = []
            while len(self.cliff_positions) < self.num_cliffs:
                new_row = np.random.randint(0, 4)
                new_col = np.random.randint(0, 11)
                state = (new_row, new_col)
                if (
                        (state not in self.cliff_positions)
                        and (state != self.start_state)
                        and (state != self.terminal_state)
                ):
                    self._cliff[new_row, new_col] = True
                    if not self.is_valid():
                        self._cliff[new_row, new_col] = False
                        continue
                    self.cliff_positions.append(state)

        # Calculate transition probabilities and rewards
        self.P = {}
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            self.P[s] = {a: [] for a in range(self.nA)}
            self.P[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            self.P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            self.P[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            self.P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

    def _calculate_transition_prob(self, current, delta):
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        if self._cliff[tuple(new_position)]:
            return [(1.0, self.start_state_index, -100, False)]

        terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
        is_terminated = tuple(new_position) == terminal_state
        return [(1 / 3, new_state, -1, is_terminated)]

    # DFS to check that it's a valid path.
    def is_valid(self):
        frontier, discovered = [], set()
        frontier.append((3, 0))
        while frontier:
            r, c = frontier.pop()
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= self.shape[0] or c_new < 0 or c_new >= self.shape[1]:
                        continue
                    if (r_new, c_new) == self.terminal_state:
                        return True
                    if not self._cliff[r_new][c_new]:
                        frontier.append((r_new, c_new))
        return False

    def step(self, Action):
        if Action not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid action {Action}   must be in [0, 1, 2, 3]")

        if self.is_hardMode:
            match Action:
                case 0:
                    Action = np.random.choice([0, 1, 3], p=[1 / 3, 1 / 3, 1 / 3])
                case 1:
                    Action = np.random.choice([0, 1, 2], p=[1 / 3, 1 / 3, 1 / 3])
                case 2:
                    Action = np.random.choice([1, 2, 3], p=[1 / 3, 1 / 3, 1 / 3])
                case 3:
                    Action = np.random.choice([0, 2, 3], p=[1 / 3, 1 / 3, 1 / 3])

        return super().step(Action)

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy-text]`"
            ) from e
        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("CliffWalking - Edited by Audrina & Kian")
                self.window_surface = pygame.display.set_mode(self.window_size)
            else:  # rgb_array
                self.window_surface = pygame.Surface(self.window_size)
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.elf_images is None:
            hikers = [
                path.join(image_path, "img/elf_up.png"),
                path.join(image_path, "img/elf_right.png"),
                path.join(image_path, "img/elf_down.png"),
                path.join(image_path, "img/elf_left.png"),
            ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in hikers
            ]
        if self.start_img is None:
            file_name = path.join(image_path, "img/stool.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.goal_img is None:
            file_name = path.join(image_path, "img/cookie.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.mountain_bg_img is None:
            bg_image = [
                path.join(image_path, "img/mountain_bg1.png"),
                path.join(image_path, "img/mountain_bg2.png"),
            ]
            self.mountain_bg_img = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in bg_image
            ]
        if self.near_cliff_img is None:
            near_cliff_image = [
                path.join(image_path, "img/mountain_near-cliff1.png"),
                path.join(image_path, "img/mountain_near-cliff2.png"),
            ]
            self.near_cliff_img = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in near_cliff_image
            ]
        if self.cliff_img is None:
            file_name = path.join(image_path, "img/mountain_cliff.png")
            self.cliff_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        for s in range(self.nS):
            row, col = np.unravel_index(s, self.shape)
            pos = (col * self.cell_size[0], row * self.cell_size[1])
            check_board_mask = row % 2 ^ col % 2
            self.window_surface.blit(self.mountain_bg_img[check_board_mask], pos)

            if self._cliff[row, col]:
                self.window_surface.blit(self.cliff_img, pos)
            if s == self.start_state_index:
                self.window_surface.blit(self.start_img, pos)
            if s == self.nS - 1:
                self.window_surface.blit(self.goal_img, pos)
            if s == self.s:
                elf_pos = (pos[0], pos[1] - 0.1 * self.cell_size[1])
                last_action = self.lastaction if self.lastaction is not None else 2
                self.window_surface.blit(self.elf_images[last_action], elf_pos)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )


# Create an environment
env = CliffWalking(render_mode="")  # human
observation, info = env.reset(seed=30)

start_time = time()

traps = [float(i[0] * 12 + i[1]) for i in env.cliff_positions]

states = dict()
for i in range(0, 4):
    for j in range(0, 12):
        if i * 12 + j in traps:
            states[i * 12 + j] = -100
        elif i == 3 and j == 11:
            states[i * 12 + j] = 1000
        else:
            states[i * 12 + j] = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}


def calculateScore(inp):
    global states
    return max(states[inp].values()) if isinstance(states[inp], type(dict())) else states[inp]


def changeScore(kp, ks, v1, v2, v3):
    global states
    score = (v1 + v2 + v3) / 3
    change = score - states[kp][ks]
    states[kp][ks] = score
    return abs(change)


def updateState():
    global states
    totalChanges = 0
    for Key, Val in states.items():
        if isinstance(Val, type(dict())):
            row = Key // 12
            col = Key % 12
            up = row * 12 + col if row == 0 else (row - 1) * 12 + col
            right = row * 12 + col if col == 11 else row * 12 + col + 1
            down = row * 12 + col if row == 3 else (row + 1) * 12 + col
            left = row * 12 + col if col == 0 else row * 12 + col - 1
            for Ks, Vs in Val.items():
                upScore = calculateScore(up)
                rightScore = calculateScore(right)
                leftScore = calculateScore(left)
                downScore = calculateScore(down)

                if Ks == 0:
                    totalChanges += changeScore(Key, Ks, upScore, rightScore, leftScore)
                elif Ks == 1:
                    totalChanges += changeScore(Key, Ks, upScore, rightScore, downScore)
                elif Ks == 2:
                    totalChanges += changeScore(Key, Ks, leftScore, rightScore, downScore)
                elif Ks == 3:
                    totalChanges += changeScore(Key, Ks, upScore, rightScore, downScore)
    return totalChanges


n = 1000
changes = updateState()

while n > 0 and changes > 0.005:
    changes = updateState()
    n -= 1

policies = dict()
for Kp, Vp in states.items():
    if isinstance(Vp, type(dict())):
        Kmax = 1
        VMax = Vp[1]
        for K, V in Vp.items():
            if V > VMax:
                Kmax = K
                VMax = V
        policies[Kp] = Kmax

policies[47] = 1

execution_time = round(time() - start_time, 2)

# Define the maximum number of iterations
max_iter_number = 1000
next_state = 36
winRate = 0
sumRewards = 0
for __ in range(max_iter_number):
    # TODO: Implement the agent policy here
    # Note: .sample() is used to sample random Action from the environment's Action space

    # Choose an Action (Replace this random Action with your agent's policy)
    # Action = env.action_space.sample()
    action = policies[next_state]

    # Perform the Action and receive feedback from the environment
    next_state, reward, done, truncated, info = env.step(action)

    sumRewards += reward

    if done:
        winRate += 1
        sumRewards += 1000

    if done or truncated:
        observation, info = env.reset()

# Close the environment
env.close()

output = '-------------------------------\n'
output += 'Markov Decision Process\n'
output += "Execution Time: " + str(execution_time) + "s\n\n"
output += 'MDP RESULTS\n'
output += '-iteration: ' + str(1000 - n) + '\n'
output += '-last iteration changes: ' + str(changes) + '\n'
output += '-States: ' + str(states) + '\n'
output += '-Policies: ' + str(policies) + '\n\n'
output += 'GAME RESULTS\n'
output += '-iteration number:' + str(max_iter_number) + '\n'
output += '-win rate:' + str(winRate) + '\n'
output += '-rewards:' + str(sumRewards) + '\n'

f = open("../Outputs/The_Phoenix-UIAI4021-PR2.txt")
f.write(output)
f.close()

print('Done')
