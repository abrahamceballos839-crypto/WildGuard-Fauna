import gymnasium as gym
from gymnasium import spaces
import numpy as np
from clasificador_cnn import WildGuardCNN


class WildGuardEnv(gym.Env):
    """Entorno de Gymnasium con el mapa oficial del proyecto WildGuard (10x10)
    integrado con la CNN real (EfficientNetB0).

    Zonas: F (Bosque), R (Río), P (Pradera), H (Base de carga en B1 e I10)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, usar_cnn=True, max_steps=200):
        super(WildGuardEnv, self).__init__()

        self.grid_size = 10
        self.max_steps = max_steps
        self.usar_cnn = usar_cnn

        # Acciones: 0: Arriba, 1: Abajo, 2: Izquierda, 3: Derecha, 4: Tomar Foto, 5: Cargar Batería
        self.action_space = spaces.Discrete(6)

        # Matriz exacta del mapa oficial (10 filas x 10 columnas)
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

        # Especies de Animals-10 asociadas a cada hábitat del mapa
        self.fauna_por_zona = {
            "F": ["scoiattolo", "ragno", "farfalla", "gatto", "cane"],
            "P": ["cavallo", "mucca", "pecora", "gallina", "elefante"],
            "R": ["elefante", "cavallo", "farfalla", "cane"],
        }

        # Espacio de observación: [pos_x, pos_y, bateria, tipo_terreno, ultima_especie]
        # Tipo de terreno: 0: Bosque(F), 1: Río(R), 2: Pradera(P), 3: Base(H)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([9, 9, 100, 3, 10], dtype=np.float32),
            dtype=np.float32,
        )

        # Cargar la CNN real de Brandon una sola vez
        self.clasificador = WildGuardCNN() if self.usar_cnn else None

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Dron inicia en la base de carga principal (B1 -> Fila 0, Columna 1)
        self.agent_pos = np.array([0, 1], dtype=int)
        self.battery = 100
        self.last_detected_class = 0
        self.steps = 0
        self.celdas_fotografiadas = set()
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        self.battery = max(0, self.battery - 2)  # Consumo por paso

        # Movimiento del dron con límites del mapa
        if action == 0 and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1  # Arriba
        elif action == 1 and self.agent_pos[0] < self.grid_size - 1:
            self.agent_pos[0] += 1  # Abajo
        elif action == 2 and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1  # Izquierda
        elif action == 3 and self.agent_pos[1] < self.grid_size - 1:
            self.agent_pos[1] += 1  # Derecha

        terreno_actual = self.mapa[self.agent_pos[0]][self.agent_pos[1]]
        reward = -1  # Costo base por paso
        info = {"terreno": terreno_actual}

        # Penalización extra si cruza por el río (humedad/riesgo)
        if terreno_actual == "R":
            reward -= 2

        # Acción 4: Tomar Foto con la CNN real de Brandon
        if action == 4:
            pos_actual = (int(self.agent_pos[0]), int(self.agent_pos[1]))

            if terreno_actual == "H":
                reward -= 5  # No hay fauna silvestre dentro de la base de carga
                info["evento"] = "sin_fauna_en_base"
            elif pos_actual in self.celdas_fotografiadas:
                reward -= 2  # Evita que el dron spamee fotos en la misma casilla
                info["evento"] = "celda_ya_fotografiada"
            else:
                self.celdas_fotografiadas.add(pos_actual)
                especie_real = str(
                    self.np_random.choice(self.fauna_por_zona[terreno_actual])
                )

                if self.usar_cnn:
                    foto = self.clasificador.tomar_foto_aleatoria(
                        clase_real=especie_real, usar_cache=True
                    )
                    predicted_idx = self.clasificador.class_names.index(
                        foto["especie_predicha"]
                    )
                    self.last_detected_class = predicted_idx

                    if foto["acierto"] and foto["confianza"] >= 0.70:
                        reward += 15  # Detección correcta y confiable
                    elif foto["acierto"]:
                        reward += 8   # Acierto con baja confianza
                    else:
                        reward -= 3   # Clasificación incorrecta de la CNN

                    info.update(foto)
                else:
                    predicted_class, _ = self._simular_salida_cnn_brandon()
                    self.last_detected_class = predicted_class
                    reward += 15

        # Acción 5: Recargar batería (solo funciona en casilla 'H')
        elif action == 5:
            if terreno_actual == "H":
                if self.battery <= 50:
                    reward += 35  # Premia recargar cuando realmente lo necesita
                else:
                    reward -= 2   # Evita recargas innecesarias con batería alta
                self.battery = 100
                info["evento"] = "recarga_exitosa"
            else:
                reward -= 5  # Penalización por intentar cargar fuera de la base
                info["evento"] = "recarga_fallida"

        # Fin de episodio por batería agotada o límite de pasos
        terminated = self.battery <= 0
        if terminated:
            reward -= 100
            info["evento"] = "bateria_agotada"

        truncated = self.steps >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, info

    def _simular_salida_cnn_brandon(self):
        predicted_class = int(np.random.randint(0, 10))
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


if __name__ == "__main__":
    print("Iniciando entorno WildGuard 10x10 con CNN real...")
    env = WildGuardEnv(usar_cnn=True)
    obs, info = env.reset()
    print("Dron en Base (B1). Observación inicial:", obs)

    # Secuencia de prueba: bajar al bosque, tomar foto, moverse a la derecha y tomar otra foto
    acciones_prueba = [
        (1, "Bajar al Bosque (B2)"),
        (4, "Tomar foto en Bosque"),
        (3, "Mover a la derecha (C2)"),
        (4, "Tomar foto en Bosque"),
        (0, "Subir (C1)"),
        (2, "Regresar a Base (B1)"),
        (5, "Intentar recargar en Base"),
    ]

    for accion, desc in acciones_prueba:
        obs, reward, terminated, truncated, info = env.step(accion)
        print(f"\nAcción: {desc}")
        print(f"  Obs: {obs} | Recompensa: {reward}")
        if "especie_predicha" in info:
            print(
                f"  CNN -> Real: {info['clase_real']} | "
                f"Predicha: {info['especie_predicha']} | "
                f"Confianza: {info['confianza']:.2%} | Acierto: {info['acierto']}"
            )
        elif "evento" in info:
            print(f"  Evento: {info['evento']}")