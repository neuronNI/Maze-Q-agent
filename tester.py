import numpy as np
import random
import time
import os

# --- КОНФИГУРАЦИЯ ---
# Измените эти параметры, чтобы настроить скорость и обучение

# Количество "игр", которые агент сыграет для обучения на КАЖДОМ лабиринте
TRAINING_EPISODES = 5000 

# Задержка между шагами в секундах (чем меньше, тем быстрее анимация)
ANIMATION_SPEED = 0.1 

# Файл с лабиринтами, которые нужно решить
MAZE_FILE = 'level.txt' # Поменяйте на 'mazes_5x5_mid.txt', 'hard' и т.д.

# --- Часть 1: Утилиты и загрузка данных ---

def clear_screen():
    """Очищает экран терминала (работает на Windows, macOS, Linux)."""
    os.system('cls' if os.name == 'nt' else 'clear')

def load_mazes_from_file(filename):
    """Загружает все лабиринты из одного файла."""
    mazes = {}
    try:
        with open(filename, 'r') as f:
            content = f.read().strip().split('ID: ')[1:]
        for maze_data in content:
            lines = maze_data.strip().split('\n')
            maze_id = int(lines[0])
            maze_grid = [list(row) for row in lines[1:]]
            mazes[maze_id] = maze_grid
        return mazes
    except FileNotFoundError:
        print(f"Ошибка: Файл '{filename}' не найден. Убедитесь, что он находится в той же папке.")
        return None

# --- Часть 2: Среда лабиринта (Maze Environment) ---
# (Без изменений по сравнению с предыдущей версией)

class MazeEnvironment:
    def __init__(self, maze_grid):
        self.maze = np.array(maze_grid)
        self.start_pos = tuple(np.argwhere(self.maze == 'S')[0])
        self.goal_pos = tuple(np.argwhere(self.maze == 'G')[0])
        self.current_pos = self.start_pos
        self.rows, self.cols = self.maze.shape
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 0:Вверх, 1:Вниз, 2:Влево, 3:Вправо

    def reset(self):
        self.current_pos = self.start_pos
        return self.current_pos

    def step(self, action_index):
        move = self.actions[action_index]
        next_pos = (self.current_pos[0] + move[0], self.current_pos[1] + move[1])

        if not (0 <= next_pos[0] < self.rows and 0 <= next_pos[1] < self.cols):
            return self.current_pos, -10, False
        
        tile = self.maze[next_pos]
        if tile == '#':
            return self.current_pos, -10, False
        
        self.current_pos = next_pos
        if tile == 'G':
            return next_pos, 100, True
        
        return next_pos, -1, False

# --- Часть 3: Агент Q-learning (Модель) ---
# (Без изменений по сравнению с предыдущей версией)

class QLearningAgent:
    def __init__(self, state_space_shape, action_space_size=4):
        self.q_table = np.zeros(state_space_shape + (action_space_size,))
        self.action_space_size = action_space_size
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.01

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state][action] = new_value

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# --- Часть 4: Обучение и Анимация ---

def train(agent, env, episodes):
    """Цикл для тренировки агента (без вывода на экран)."""
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
        agent.update_epsilon()

def solve_and_animate(agent, env, maze_id, filename):
    """
    Использует обученного агента для решения лабиринта с пошаговой анимацией.
    """
    state = env.reset()
    done = False
    steps = 0
    max_steps = env.rows * env.cols * 2 # Защита от зацикливания

    while not done and steps < max_steps:
        clear_screen()
        print(f"--- Анимация решения: Лабиринт ID {maze_id} из файла {filename} ---")
        
        # Отрисовка текущего состояния
        grid_copy = [list(row) for row in env.maze]
        grid_copy[state[0]][state[1]] = 'A' # 'A' - Агент
        for row in grid_copy:
            print(" ".join(row))
            
        print(f"\nШаг: {steps}")
        print(f"Текущая позиция: {state}")
        
        time.sleep(ANIMATION_SPEED)

        # Агент выбирает лучшее действие (без случайности)
        action = np.argmax(agent.q_table[state])
        next_state, _, done = env.step(action)
        state = next_state
        steps += 1
    
    # Показать финальный кадр
    clear_screen()
    print(f"--- РЕЗУЛЬТАТ: Лабиринт ID {maze_id} из файла {filename} ---")
    grid_copy = [list(row) for row in env.maze]
    grid_copy[state[0]][state[1]] = 'A'
    for row in grid_copy:
        print(" ".join(row))
    if done:
        print(f"\nЦель достигнута за {steps} шагов!")
    else:
        print(f"\nАгент не смог найти выход за {max_steps} шагов.")


# --- Основной блок для запуска ---
if __name__ == '__main__':
    mazes = load_mazes_from_file(MAZE_FILE)

    if mazes:
        # Сортируем ID, чтобы проходить лабиринты по порядку
        sorted_ids = sorted(mazes.keys())
        
        for maze_id in sorted_ids:
            clear_screen()
            print(f"--- Подготовка к лабиринту ID: {maze_id} из файла {MAZE_FILE} ---")
            
            # 1. Создаем среду для текущего лабиринта
            maze_grid = mazes[maze_id]
            environment = MazeEnvironment(maze_grid)
            
            # 2. Создаем НОВОГО агента для каждого лабиринта
            # Это важно, т.к. Q-таблица специфична для каждой среды
            agent = QLearningAgent(state_space_shape=environment.maze.shape)

            # 3. Обучаем агента
            print(f"Идет обучение агента на {TRAINING_EPISODES} эпизодах... (это может занять несколько секунд)")
            train(agent, environment, episodes=TRAINING_EPISODES)
            print("Обучение завершено!")
            time.sleep(2) # Небольшая пауза перед анимацией

            # 4. Запускаем анимацию решения
            solve_and_animate(agent, environment, maze_id, MAZE_FILE)
            
            # 5. Пауза перед следующим лабиринтом
            try:
                input("\nНажмите Enter, чтобы перейти к следующему лабиринту...")
            except KeyboardInterrupt:
                print("\nВыход из программы.")
                break
        
        print("\nВсе лабиринты из файла пройдены!")