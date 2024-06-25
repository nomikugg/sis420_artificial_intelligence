import pygame
import pickle
import numpy as np

# Dimensiones de la pantalla
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400

# Tamaño de la cuadrícula del rompecabezas
ROWS = 4
COLS = 5
CELL_SIZE = 80

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Cargar el q_table desde el archivo pickle
with open('q_tablebest.pkl', 'rb') as f:
    q_table = pickle.load(f)

# Inicialización de Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

# Función para dibujar el rompecabezas
def draw_puzzle(agent_position):
    screen.fill(WHITE)
    for row in range(ROWS):
        for col in range(COLS):
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)
            if (row, col) == agent_position:
                pygame.draw.circle(screen, BLUE, rect.center, CELL_SIZE // 3)

    pygame.display.flip()

# Función principal del juego
def play_game():
    agent_position = (0, 0)
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        current_state = (agent_position[0], agent_position[1])
        action = np.argmax(q_table[current_state[0], current_state[1]])

        if action == 0:  # UP
            agent_position = (max(agent_position[0] - 1, 0), agent_position[1])
        elif action == 1:  # DOWN
            agent_position = (min(agent_position[0] + 1, ROWS - 1), agent_position[1])
        elif action == 2:  # LEFT
            agent_position = (agent_position[0], max(agent_position[1] - 1, 0))
        elif action == 3:  # RIGHT
            agent_position = (agent_position[0], min(agent_position[1] + 1, COLS - 1))

        draw_puzzle(agent_position)
        clock.tick(2)  # Velocidad del juego (cuadros por segundo)

    pygame.quit()

if __name__ == '__main__':
    play_game()
