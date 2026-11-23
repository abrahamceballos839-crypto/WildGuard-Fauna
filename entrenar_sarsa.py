import os
import numpy as np
import matplotlib.pyplot as plt
from entorno_gymnasium import WildGuardEnv


def obtener_estado(obs, env):
    """Discretiza la observación en (fila, col, nivel_bateria, celda_fotografiada)."""
    x = int(obs[0])
    y = int(obs[1])
    bat = int(obs[2])

    # Cortes alineados con el umbral de recarga (<= 50) del entorno
    if bat <= 20:
        nivel_bat = 0
    elif bat <= 40:
        nivel_bat = 1
    elif bat <= 50:
        nivel_bat = 2
    elif bat <= 75:
        nivel_bat = 3
    else:
        nivel_bat = 4

    ya_foto = 1 if (x, y) in env.celdas_fotografiadas else 0
    return (x, y, nivel_bat, ya_foto)


def elegir_accion(Q, estado, epsilon, rng):
    """Política epsilon-greedy con desempate aleatorio."""
    if rng.random() < epsilon:
        return int(rng.integers(0, 6))
    valores = Q[estado]
    max_val = np.max(valores)
    mejores = np.flatnonzero(np.isclose(valores, max_val))
    return int(rng.choice(mejores))


def entrenar_sarsa(
    episodios=8000,
    alpha=0.1,
    gamma=0.95,
    epsilon_inicio=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.9994,
):
    print(f"--- Fase 1: Entrenando agente SARSA ({episodios} episodios) ---")
    env = WildGuardEnv(usar_cnn=False, max_steps=200)
    rng = np.random.default_rng(42)

    # Tabla Q: [10 filas, 10 cols, 5 niveles bateria, 2 estados foto, 6 acciones]
    Q = np.zeros((10, 10, 5, 2, 6), dtype=np.float32)

    recompensas_episodio = []
    fotos_episodio = []
    epsilon = epsilon_inicio

    for ep in range(episodios):
        obs, _ = env.reset()
        estado = obtener_estado(obs, env)
        accion = elegir_accion(Q, estado, epsilon, rng)

        recompensa_total = 0.0
        done = False

        while not done:
            obs_sig, reward, terminated, truncated, _ = env.step(accion)
            done = terminated or truncated
            estado_sig = obtener_estado(obs_sig, env)
            accion_sig = elegir_accion(Q, estado_sig, epsilon, rng)

            # Regla de actualización SARSA (on-policy)
            objetivo = reward if terminated else reward + gamma * Q[estado_sig][accion_sig]
            Q[estado][accion] += alpha * (objetivo - Q[estado][accion])

            estado = estado_sig
            accion = accion_sig
            recompensa_total += reward

        recompensas_episodio.append(recompensa_total)
        fotos_episodio.append(len(env.celdas_fotografiadas))
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if (ep + 1) % 500 == 0:
            prom_r = np.mean(recompensas_episodio[-100:])
            prom_f = np.mean(fotos_episodio[-100:])
            print(
                f"Episodio {ep + 1:4d}/{episodios} | "
                f"Recompensa prom (últ. 100): {prom_r:6.1f} | "
                f"Celdas foto prom: {prom_f:4.1f} | Epsilon: {epsilon:.3f}"
            )

    os.makedirs("models", exist_ok=True)
    np.save("models/q_table_sarsa.npy", Q)
    print("Tabla Q guardada en models/q_table_sarsa.npy")
    return Q, recompensas_episodio, fotos_episodio


def evaluar_con_cnn_real(Q, max_steps=100):
    print("\n--- Fase 2: Evaluación integrada con la CNN real de Brandon ---")
    env = WildGuardEnv(usar_cnn=True, max_steps=max_steps)
    rng = np.random.default_rng(7)

    obs, _ = env.reset()
    estado = obtener_estado(obs, env)

    trayectoria = [(int(obs[0]), int(obs[1]))]
    puntos_foto = []
    recompensa_total = 0.0
    detecciones = 0
    aciertos_cnn = 0
    recargas = 0
    done = False

    while not done:
        # Política greedy pura (epsilon = 0)
        accion = elegir_accion(Q, estado, epsilon=0.0, rng=rng)
        obs, reward, terminated, truncated, info = env.step(accion)
        done = terminated or truncated
        estado = obtener_estado(obs, env)

        recompensa_total += reward
        pos = (int(obs[0]), int(obs[1]))
        trayectoria.append(pos)

        if "especie_predicha" in info:
            detecciones += 1
            if info["acierto"]:
                aciertos_cnn += 1
            puntos_foto.append(pos)
            print(
                f"  Paso {env.steps:3d} en {pos} (Bat {int(obs[2]):3d}%) -> "
                f"Real: {info['clase_real']:10s} | CNN: {info['especie_predicha']:10s} "
                f"({info['confianza']:.1%}) | Acierto: {info['acierto']}"
            )
        elif info.get("evento") == "recarga_exitosa":
            recargas += 1
            print(f"  Paso {env.steps:3d} en {pos} -> Recarga exitosa en Base H (Bat -> 100%)")
        elif info.get("evento") == "bateria_agotada":
            print(f"  Paso {env.steps:3d} en {pos} -> BATERÍA AGOTADA")

    precision_demo = (aciertos_cnn / detecciones * 100) if detecciones > 0 else 0.0
    print("\n=== Resumen de Misión Integrada (SARSA + EfficientNetB0) ===")
    print(f"Pasos completados:  {env.steps}/{max_steps}")
    print(f"Recompensa total:   {recompensa_total:.1f}")
    print(f"Fotos tomadas:      {detecciones} (Aciertos CNN: {aciertos_cnn}/{detecciones} = {precision_demo:.1f}%)")
    print(f"Recargas en base:   {recargas}")
    print(f"Batería final:      {int(obs[2])}%")

    return env, trayectoria, puntos_foto


def guardar_graficas(recompensas, fotos, env, trayectoria, puntos_foto):
    os.makedirs("resultados_rl", exist_ok=True)

    # 1. Curva de aprendizaje (Recompensa por episodio)
    ventana = 100
    media_movil = np.convolve(recompensas, np.ones(ventana) / ventana, mode="valid")

    plt.figure(figsize=(10, 5))
    plt.plot(recompensas, alpha=0.25, color="steelblue", label="Recompensa por episodio")
    plt.plot(
        range(ventana - 1, len(recompensas)),
        media_movil,
        color="darkblue",
        linewidth=2,
        label=f"Media móvil ({ventana} episodios)",
    )
    plt.title("Curva de Aprendizaje del Dron WildGuard (SARSA)")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa Acumulada")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("resultados_rl/grafica_entrenamiento_sarsa.png", dpi=150)
    plt.close()

    # 2. Mapa 10x10 con la ruta ejecutada por el dron y las fotos tomadas
    colores = {
        "F": [0.22, 0.56, 0.34],  # Bosque
        "R": [0.26, 0.56, 0.84],  # Río
        "P": [0.85, 0.80, 0.45],  # Pradera
        "H": [0.88, 0.34, 0.25],  # Base de carga
    }
    grid_rgb = np.zeros((10, 10, 3))
    for r in range(10):
        for c in range(10):
            grid_rgb[r, c] = colores[env.mapa[r][c]]

    plt.figure(figsize=(7, 7))
    plt.imshow(grid_rgb)

    filas = [p[0] for p in trayectoria]
    cols = [p[1] for p in trayectoria]
    plt.plot(cols, filas, color="white", linewidth=2, linestyle="-", label="Ruta del dron")
    plt.scatter(cols[0], filas[0], c="black", s=120, marker="s", zorder=5, label="Inicio (B1)")

    if puntos_foto:
        f_rows = [p[0] for p in puntos_foto]
        f_cols = [p[1] for p in puntos_foto]
        plt.scatter(f_cols, f_rows, c="yellow", edgecolors="black", s=110, marker="*", zorder=6, label="Foto CNN")

    for r in range(10):
        for c in range(10):
            plt.text(c, r, env.mapa[r][c], ha="center", va="center", color="black", fontsize=8, alpha=0.6)

    plt.xticks(range(10), list("ABCDEFGHIJ"))
    plt.yticks(range(10), range(1, 11))
    plt.title("Ruta de Patrullaje WildGuard (SARSA + CNN Real)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("resultados_rl/ruta_dron_sarsa.png", dpi=150)
    plt.close()

    print("\nGráficas guardadas en:")
    print(" - resultados_rl/grafica_entrenamiento_sarsa.png")
    print(" - resultados_rl/ruta_dron_sarsa.png")


if __name__ == "__main__":
    Q, recompensas, fotos = entrenar_sarsa()
    env_eval, trayectoria, puntos_foto = evaluar_con_cnn_real(Q, max_steps=100)
    guardar_graficas(recompensas, fotos, env_eval, trayectoria, puntos_foto)