import random
import time
from collections import defaultdict
import pygame

# Directions: up, right, down, left
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

class SnakeGame:
    def __init__(self, width=30, height=30, display=False, cell_size=20):
        self.width = width
        self.height = height
        self.display = display
        self.cell_size = cell_size
        if self.display:
            pygame.init()
            self.window = pygame.display.set_mode(
                (self.width * self.cell_size, self.height * self.cell_size)
            )
            pygame.display.set_caption("Snake Q-Learning - Back to the Future")
            self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # Start with a snake of length 3 in the center moving right
        cx, cy = self.width // 2, self.height // 2
        self.snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.direction_index = 1  # moving right
        self.spawn_food()
        self.score = 0
        return self.get_state()

    def spawn_food(self):
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if (x, y) not in self.snake:
                self.food = (x, y)
                break

    def step(self, action):
        # action: 0 = turn left, 1 = straight, 2 = turn right
        if action == 0:
            self.direction_index = (self.direction_index - 1) % 4
        elif action == 2:
            self.direction_index = (self.direction_index + 1) % 4
        dx, dy = DIRECTIONS[self.direction_index]
        head_x, head_y = self.snake[0]
        old_dist = abs(head_x - self.food[0]) + abs(head_y - self.food[1])
        new_head = (head_x + dx, head_y + dy)

        reward = 0.0
        done = False

        # Check collisions
        if (
            new_head[0] < 0 or new_head[0] >= self.width or
            new_head[1] < 0 or new_head[1] >= self.height or
            new_head in self.snake
        ):
            done = True
            reward = -1.0
            return self.get_state(), reward, done

        # Move snake
        self.snake.insert(0, new_head)
        if new_head == self.food:
            reward = 1.0
            self.score += 1
            self.spawn_food()
        else:
            self.snake.pop()
            new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            if new_dist < old_dist:
                reward += 0.1
            else:
                reward -= 0.05

        reward -= 0.01
        return self.get_state(), reward, done

    def danger_at(self, position):
        x, y = position
        return (
            x < 0 or x >= self.width or y < 0 or y >= self.height or
            position in self.snake
        )

    def get_state(self):
        head_x, head_y = self.snake[0]
        dir_idx = self.direction_index
        dir_vec = DIRECTIONS[dir_idx]
        left_vec = DIRECTIONS[(dir_idx - 1) % 4]
        right_vec = DIRECTIONS[(dir_idx + 1) % 4]

        state = (
            1 if self.danger_at((head_x + left_vec[0], head_y + left_vec[1])) else 0,
            1 if self.danger_at((head_x + dir_vec[0], head_y + dir_vec[1])) else 0,
            1 if self.danger_at((head_x + right_vec[0], head_y + right_vec[1])) else 0,
            1 if self.food[0] < head_x else 0,
            1 if self.food[0] > head_x else 0,
            1 if self.food[1] < head_y else 0,
            1 if self.food[1] > head_y else 0,
            dir_idx,
        )
        return state

    def render(self):
        if not self.display:
            board = [['.' for _ in range(self.width)] for _ in range(self.height)]
            for x, y in self.snake:
                board[y][x] = 'S'
            fx, fy = self.food
            board[fy][fx] = 'F'
            print("\n".join("".join(row) for row in board))
            print("Score:", self.score)
            return

        self.window.fill((0, 0, 0))
        for x, y in self.snake:
            pygame.draw.rect(
                self.window,
                (0, 229, 255),
                (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size),
            )
        fx, fy = self.food
        pygame.draw.rect(
            self.window,
            (255, 107, 0),
            (fx * self.cell_size, fy * self.cell_size, self.cell_size, self.cell_size),
        )
        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        if self.display:
            pygame.quit()


class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount=0.9):
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0])
        self.lr = learning_rate
        self.gamma = discount

    def choose_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 2)
        q_values = self.q_table[state]
        max_q = max(q_values)
        return q_values.index(max_q)

    def learn(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]
        next_max = max(self.q_table[next_state])
        target = reward + (self.gamma * next_max if not done else 0.0)
        self.q_table[state][action] += self.lr * (target - current_q)


if __name__ == "__main__":
    episodes = 1200
    env = SnakeGame(width=30, height=30, display=False)
    agent = QLearningAgent(learning_rate=0.1, discount=0.9)

    epsilon = 1.0
    epsilon_min = 0.02
    epsilon_decay = 0.995

    for ep in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state, epsilon)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep + 1}, score: {env.score}, epsilon: {epsilon:.3f}")

    env.close()

    # Play one game with learned policy and display
    env = SnakeGame(width=30, height=30, display=True)
    state = env.reset()
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        action = agent.choose_action(state, epsilon=0.0)
        next_state, reward, done = env.step(action)
        state = next_state
        env.render()
    print("Final score:", env.score)
    env.close()
