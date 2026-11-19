import gymnasium as gym
from gymnasium import spaces
import numpy as np


class WildGuardEnv(gym.Env):
  """Entorno de Gymnasium con el mapa oficial del proyecto WildGuard (10x10)

  Zonas: F (Bosque), R (Río), P (Pradera), H (Base de carga en B1 e I10)
  """

  metadata = {"render_modes": ["human"]}

  def __init__(self):
    super(WildGuardEnv, self).__init__()

    self.grid_size = 10
    # Acciones: 0: Arriba, 1: Abajo, 2: Izquierda, 3: Derecha, 4: Tomar Foto, 5: Cargar Batería
    self.action_space = spaces.Discrete(6)

    # Matriz exacta del mapa de la imagen (10 filas x 10 columnas)
    self.mapa = [
        ["F", "H", "F", "F", "F", "R", "R", "P", "P", "P"],  # Fila 1 (A1 - J1)
        ["F", "F", "F", "F", "F", "R", "R", "P", "P", "P"],  # Fila 2
        ["F", "F", "F", "F", "F", "R", "P", "P", "P", "P"],  # Fila 3
        ["F", "F", "F", "F", "F", "R", "P", "P", "P", "P"],  # Fila 4
        ["F", "F", "F", "F", "F", "R", "P", "P", "P", "P"],  # Fila 5
        ["F", "F", "F", "F", "R", "R", "R", "P", "P", "P"],  # Fila 6
        ["F", "F", "F", "F", "F", "R", "P", "P", "P", "P"],  # Fila 7
        ["F", "F", "F", "F", "F", "R", "P", "P", "P", "P"],  # Fila 8
        ["F", "F", "F", "F", "F", "R", "P", "P", "P", "P"],  # Fila 9
        ["F", "F", "F", "F", "R", "R", "R", "P", "H", "P"],  # Fila 10 (A10 - J10)
    ]

    # Espacio de observación: [pos_x, pos_y, bateria, tipo_terreno, ultima_especie]
    # Tipo de terreno: 0: Bosque(F), 1: Río(R), 2: Pradera(P), 3: Base(H)
    self.observation_space = spaces.Box(
        low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
        high=np.array([9, 9, 100, 3, 10], dtype=np.float32),
        dtype=np.float32,
    )

    self.reset()

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    # Iniciamos al dron en la base de carga principal (B1 -> Fila 0, Columna 1)
    self.agent_pos = np.array([0, 1], dtype=int)
    self.battery = 100
    self.last_detected_class = 0
    return self._get_obs(), {}

  def step(self, action):
    self.battery -= 2  # Consumo por paso

    # Movimiento del dron con límites del mapa
    if action == 0 and self.agent_pos[0] > 0:
      self.agent_pos[0] -= 1  # Arriba
    elif action == 1 and self.agent_pos[0] < self.grid_size - 1:
      self.agent_pos[0] += 1  # Abajo
    elif action == 2 and self.agent_pos[1] > 0:
      self.agent_pos[1] -= 1  # Izquierda
    elif action == 3 and self.agent_pos[1] < self.grid_size - 1:
      self.agent_pos[1] += 1  # Derecha

    # Identificar en qué tipo de terreno cayó el dron
    terreno_actual = self.mapa[self.agent_pos[0]][self.agent_pos[1]]
    reward = -1  # Costo base por paso

    # Penalización extra si cruza por el río (por ejemplo, humedad)
    if terreno_actual == "R":
      reward -= 2

    # Acción 4: Tomar Foto (Simula la CNN de Brandon)
    if action == 4:
      predicted_class, cnn_probabilities = self._simular_salida_cnn_brandon()
      self.last_detected_class = predicted_class
      reward += 15  # Recompensa por detección de fauna

    # Acción 5: Recargar batería (Solo funciona si está en una casilla 'H')
    elif action == 5:
      if terreno_actual == "H":
        self.battery = 100
        reward += 20
      else:
        reward -= 5  # Penalización por intentar cargar fuera de la base

    terminated = self.battery <= 0
    truncated = False

    return self._get_obs(), reward, terminated, truncated, {}

  def _simular_salida_cnn_brandon(self):
    predicted_class = np.random.randint(0, 10)
    probabilities = np.random.dirichlet(np.ones(10))
    return predicted_class, probabilities

  def _get_obs(self):
    terreno = self.mapa[self.agent_pos[0]][self.agent_pos[1]]
    mapping = {"F": 0, "R": 1, "P": 2, "H": 3}
    terreno_val = mapping.get(terreno, 0)

    return np.array(
        [
            self.agent_pos[0],
            self.agent_pos[1],
            self.battery,
            terreno_val,
            self.last_detected_class,
        ],
        dtype=np.float32,
    )


# Prueba rápida del entorno con el mapa oficial
env = WildGuardEnv()
obs, info = env.reset()
print(
    "¡Mapa oficial 10x10 cargado con éxito! Dron ubicado en Base (B1). Observación"
    " inicial:",
    obs,
)