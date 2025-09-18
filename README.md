# Maze-Q-agent
🧠 Q-Learning Maze Solver
General description

The project is an implementation of a learning agent based on the Q-learning algorithm, designed to find the optimal path in a maze.
The solution is focused on the research of Reinforcement Learning (RL) methods and the demonstration of their practical application in navigation tasks in discrete environments.

The agent learns by interacting with the environment, receiving positive or negative rewards depending on their actions. As the number of training episodes increases, the agent forms a strategy that minimizes the number of steps to the goal and avoids incorrect moves.

Key features

Support for multiple mazes from a single file (different sizes and difficulty levels).

A Q-learning agent with configurable parameters (alpha, gamma, epsilon, decay).

Step-by-step visualization of the solution process in the terminal.

Flexible configuration of animation speed and number of training episodes.

Easy modification of rewards for experimenting with agent behavior.

Solution architecture

MazeEnvironment

Defines the rules of the environment: acceptable actions, boundaries, walls, start and goal.

Returns a new state, reward, and episode end signal.

QLearningAgent

Uses a Q-table (state × action).

Implements the epsilon-greedy strategy to balance research and operation.

Updates the Q values based on the Bellman function.

Training Loop

Launches a series of training episodes.

Gradually reduces the risk, reducing the likelihood of accidental actions.

Forms a stable exit strategy from the maze.

Solve & Animate

Uses a trained agent to traverse the maze.

Visualizes the agent's movement and steps before reaching the goal.

Application areas

Education: A practical example of Reinforcement Learning in discrete environments.

Research: a basic framework for testing and comparing RL algorithms (Q-learning, SARSA, DQN).

Prototyping: the basis for the development of navigation systems in confined spaces.

Visualization: demonstration of the principles of operation of trained agents in a visual form.

Development prospects

Adding learning metrics (episode success, average number of steps).

Saving/loading the Q-table for reuse of the model.

Integration with the graphical interface (PyGame, Tkinter, Web UI).

Generate random mazes or connect external data sources.

Experiments with neural network approaches (Deep Q-Network, Policy Gradient).

Conclusion

This project is an educational and research platform for working with reinforcement learning methods.
It can be used both for educational purposes (demonstrating the principles of RL) and as a starting point for more complex navigation and optimization solutions.
