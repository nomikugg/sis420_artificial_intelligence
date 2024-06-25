import numpy as np
import pygame
import pickle
import matplotlib.pyplot as plt

class PuzzleEnv:
    def __init__(self, rows, cols):
        """
        Constructor de la clase PuzzleEnv.

        Parameters:
        - rows (int): Número de filas del rompecabezas.
        - cols (int): Número de columnas del rompecabezas.
        """
        self.rows = rows  # Número de filas del rompecabezas
        self.cols = cols  # Número de columnas del rompecabezas
        self.goal_state = self.generate_goal_state()  # Genera el estado objetivo del rompecabezas
        self.current_state = self.generate_start_state()  # Genera el estado inicial aleatorio del rompecabezas
        self.empty_pos = (rows - 1, cols - 1)  # Posición de la casilla vacía

        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']  # Acciones posibles: mover hacia arriba, abajo, izquierda, derecha
        self.num_actions = len(self.actions)  # Número total de acciones

        self.tile_size = 50  # Tamaño de cada casilla del rompecabezas en píxeles
        self.window_width = cols * self.tile_size  # Ancho de la ventana del juego
        self.window_height = rows * self.tile_size  # Alto de la ventana del juego
        self.window_title = 'Puzzle Environment'  # Título de la ventana
        self.colors = {
            'black': (0, 0, 0),    # Definición de colores en formato RGB
            'white': (255, 255, 255),
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255)
        }

        pygame.init()  # Inicializa Pygame
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))  # Crea la ventana de Pygame
        pygame.display.set_caption(self.window_title)  # Asigna el título a la ventana

    def generate_goal_state(self):
        """
        Genera el estado objetivo del rompecabezas.

        Returns:
        - np.ndarray: Matriz del estado objetivo del rompecabezas.
        """
        goal_state = np.arange(1, self.rows * self.cols)  # Genera una matriz con números del 1 al 19
        goal_state = np.append(goal_state, 0)  # Añade el número 0 como la casilla vacía
        np.random.shuffle(goal_state)  # Mezcla los números
        return goal_state.reshape((self.rows, self.cols))  # Retorna la matriz con la forma adecuada

    def generate_start_state(self):
        """
        Genera un estado inicial aleatorio del rompecabezas.

        Returns:
        - np.ndarray: Matriz del estado inicial aleatorio del rompecabezas.
        """
        start_state = self.goal_state.flatten()  # Aplana el estado objetivo
        np.random.shuffle(start_state)  # Mezcla los números
        return start_state.reshape((self.rows, self.cols))  # Retorna la matriz con la forma adecuada

    def reset(self):
        """
        Reinicia el estado actual del rompecabezas y retorna el estado inicial.

        Returns:
        - np.ndarray: Matriz del estado inicial aleatorio del rompecabezas.
        """
        self.current_state = self.generate_start_state()  # Genera un nuevo estado inicial aleatorio
        self.empty_pos = tuple(np.argwhere(self.current_state == 0)[0])  # Encuentra la posición de la casilla vacía
        return self.current_state

    def step(self, action):
        """
        Realiza una acción en el rompecabezas y retorna el siguiente estado, la recompensa, si el episodio ha terminado y
        cualquier información adicional.

        Parameters:
        - action (int): Índice de la acción a realizar (0: UP, 1: DOWN, 2: LEFT, 3: RIGHT).

        Returns:
        - tuple: (next_state, reward, done, {}).
          - next_state (np.ndarray): Siguiente estado después de realizar la acción.
          - reward (float): Recompensa obtenida por la acción.
          - done (bool): Indica si el episodio ha terminado.
          - {} (dict): Información adicional opcional.
        """
        row, col = self.empty_pos  # Obtiene la posición de la casilla vacía

        if action == 0 and row > 0:  # Mover hacia arriba
            self.swap((row, col), (row - 1, col))
        elif action == 1 and row < self.rows - 1:  # Mover hacia abajo
            self.swap((row, col), (row + 1, col))
        elif action == 2 and col > 0:  # Mover hacia la izquierda
            self.swap((row, col), (row, col - 1))
        elif action == 3 and col < self.cols - 1:  # Mover hacia la derecha
            self.swap((row, col), (row, col + 1))

        reward = self.calculate_reward()  # Calcula la recompensa basada en el estado actual
        done = (self.current_state == self.goal_state).all()  # Verifica si se ha alcanzado el estado objetivo

        self.render()  # Renderiza el estado actual del rompecabezas

        return self.current_state, reward, done, {}

    def swap(self, pos1, pos2):
        """
        Intercambia dos posiciones en la matriz del rompecabezas.

        Parameters:
        - pos1 (tuple): Primera posición a intercambiar.
        - pos2 (tuple): Segunda posición a intercambiar.
        """
        self.current_state[pos1], self.current_state[pos2] = self.current_state[pos2], self.current_state[pos1]
        self.empty_pos = pos2  # Actualiza la posición de la casilla vacía

    def calculate_reward(self):
        """
        Calcula la recompensa basada en la posición actual del rompecabezas.

        Returns:
        - float: Recompensa obtenida por la acción realizada.
        """
        if (self.current_state == self.goal_state).all():  # Si se alcanza el estado objetivo
            return 100.0  # Recompensa alta
        else:
            return -1.0  # Penalización por cada movimiento

    def render(self):
        """
        Renderiza el estado actual del rompecabezas utilizando Pygame.
        """
        self.screen.fill(self.colors['white'])  # Limpia la pantalla con el color de fondo

        # Dibuja las casillas del rompecabezas y los números correspondientes
        for row in range(self.rows):
            for col in range(self.cols):
                value = self.current_state[row, col]
                rect = pygame.Rect(col * self.tile_size, row * self.tile_size, self.tile_size, self.tile_size)
                pygame.draw.rect(self.screen, self.colors['black'], rect, 1)
                if value != 0:  # No dibujar nada para la casilla vacía (0)
                    text_surface = pygame.font.SysFont(None, 24).render(str(value), True, self.colors['blue'])
                    self.screen.blit(text_surface, rect.center)

        pygame.display.flip()  # Actualiza la pantalla

    def close(self):
        """
        Cierra la ventana de Pygame y finaliza la aplicación.
        """
        pygame.quit()  # Cierra Pygame


def state_to_index(state):
    """
    Convierte el estado del rompecabezas a un índice único para el diccionario de la tabla Q.

    Parameters:
    - state (np.ndarray): Estado actual del rompecabezas.

    Returns:
    - str: Clave única para representar el estado del rompecabezas en el diccionario de la tabla Q.
    """
    return ''.join(map(str, state.flatten()))


def run(episodes, rows, cols, is_training=True):
    """
    Función principal para entrenar o probar el agente que resuelve el rompecabezas.

    Parameters:
    - episodes (int): Número de episodios de entrenamiento.
    - rows (int): Número de filas del rompecabezas.
    - cols (int): Número de columnas del rompecabezas.
    - is_training (bool): Indica si se está entrenando al agente (True) o probando (False).
    """
    env = PuzzleEnv(rows, cols)  # Inicializa el entorno del rompecabezas

    if is_training:
        q_table = {}  # Inicializa la tabla Q como un diccionario vacío
    else:
        with open('q_table.pkl', 'rb') as f:
            q_table = pickle.load(f)  # Carga la tabla Q desde un archivo pickle

    epsilon = 1.0  # Factor de exploración inicial
    epsilon_min = 0.01  # Valor mínimo de epsilon
    epsilon_decay = (epsilon - epsilon_min) / episodes  # Decaimiento lineal de epsilon
    alpha = 0.5  # Tasa de aprendizaje
    gamma = 0.9  # Factor de descuento para recompensas futuras

    rewards_per_episode = np.zeros(episodes)  # Arreglo para almacenar las recompensas por episodio

    for episode in range(episodes):
        state = env.reset()  # Reinicia el rompecabezas y obtiene el estado inicial
        state_index = state_to_index(state)  # Convierte el estado actual a un índice para la tabla Q
        total_reward = 0
        done = False

        while not done:
            if is_training and np.random.rand() < epsilon:
                action = np.random.randint(env.num_actions)  # Acción aleatoria (exploración)
            else:
                action = np.argmax(q_table.get(state_index, [0] * env.num_actions))  # Acción basada en la tabla Q

            next_state, reward, done, _ = env.step(action)  # Ejecuta la acción en el rompecabezas

            if is_training:
                next_state_index = state_to_index(next_state)  # Convierte el siguiente estado a un índice
                q_value = q_table.get(state_index, [0] * env.num_actions)[action]  # Obtiene el valor Q actual
                max_next_q = np.max(q_table.get(next_state_index, [0] * env.num_actions)) if not done else 0
                new_q = q_value + alpha * (reward + gamma * max_next_q - q_value)  # Actualiza el valor Q
                q_table[state_index] = q_table.get(state_index, [0] * env.num_actions)
                q_table[state_index][action] = new_q  # Actualiza la tabla Q para el estado actual

                total_reward += reward  # Acumula la recompensa total del episodio

            state = next_state
            state_index = next_state_index

        rewards_per_episode[episode] = total_reward  # Almacena la recompensa total del episodio

        if is_training and epsilon > epsilon_min:
            epsilon -= epsilon_decay  # Decaimiento de epsilon durante el entrenamiento

        if (episode + 1) % 100 == 0:
            print(f'Episode {episode + 1}/{episodes}, Total Reward: {total_reward}')

    plt.plot(rewards_per_episode)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Reward per Episode')
    plt.show()

    if is_training:
        with open('q_table.pkl', 'wb') as f:
            pickle.dump(q_table, f)  # Guarda la tabla Q entrenada en un archivo pickle

    env.close()  # Cierra el entorno de Pygame al finalizar


if __name__ == '__main__':
    episodes = 100 # Número total de episodios para entrenar al agente
    rows = 4  # Número de filas del rompecabezas
    cols = 5  # Número de columnas del rompecabezas
    run(episodes, rows, cols, is_training=True)  # Ejecuta el entrenamiento del agente
