# WildGuard-Fauna 🐾
Dron de conservación para identificar fauna silvestre mediante Visión Artificial y Aprendizaje por Refuerzo.
Proyecto final de Aprendizaje Automático (Universidad Tecnológica de León, 2026).

## Equipo
- **Jasciel Abraham Ceballos Martínez:** Líder de proyecto y datos.
- **Mario Osvaldo Cortez Cruz:** ML clásico (supervisado).
- **Betzie Yuliana López Reyes:** ML clásico (no supervisado y métricas).
- **Brandon Eduardo Alvarado Durán:** CNN (Deep Learning).
- **Óscar Ignacio López Palomino:** Refuerzo e integración.

## Estado del Proyecto
El repositorio contiene las tres fases del proyecto completamente integradas en un solo notebook ejecutable:
- **Entrega 1:** Preparación de datos, EDA, partición 70/15/15 y modelos clásicos (SVM, Random Forest, K-Means).
- **Entrega 2:** Redes Neuronales Convolucionales (CNN propia, Transfer Learning con EfficientNetB0 y ResNet50, Grad-CAM).
- **Entrega 3:** Entorno Gymnasium 10x10 y agente de Aprendizaje por Refuerzo (SARSA) para el patrullaje autónomo del dron.

## Resumen de Resultados
- **Clasificación (E2):** El mejor modelo fue **EfficientNetB0** con un **94.78% de Accuracy** en test, superando ampliamente al mejor modelo clásico (SVM, 58.11%).
- **Refuerzo (E3):** La política **SARSA B** logró una recompensa media de **196.64**, superando a un agente aleatorio (-143.66). Se demostró que la política B aprende a recargar batería (500 recargas útiles), mientras que la política A falla por falta de recarga (agota la batería en el 100% de los episodios).

## Dataset
Se utiliza **Animals-10** (Kaggle, 26,179 imágenes, 10 clases). 
*Nota:* El dataset no se sube al repositorio. El notebook está configurado para descargarlo y verificarlo automáticamente usando `kagglehub` en la primera celda. Si prefieres descargarlo manualmente, colócalo en la ruta `./dataset_animals10/` (una carpeta por clase).

**Clases (carpetas originales en italiano):**
- `cane` = perro
- `cavallo` = caballo
- `elefante` = elefante
- `farfalla` = mariposa
- `gallina` = pollo
- `gatto` = gato
- `mucca` = vaca
- `pecora` = oveja
- `ragno` = araña
- `scoiattolo` = ardilla

##  Estructura del Repositorio
- `processed_data/` — CSVs de partición (train/val/test) y archivos `.npy` de características.
- `models/` — Modelos entrenados (`.keras`, `.pkl`, tablas Q de SARSA) y metadatos. Los archivos pesados están excluidos del control de versiones.
- `resultados/` — Gráficas generadas (EDA, matrices de confusión, curvas de aprendizaje) y el GIF del patrullaje.
- `WildGuard-Fauna.ipynb` — Notebook principal único del proyecto (ejecutar de arriba a abajo).

## Requisitos e Instalación
1. Clona el repositorio y entra a la carpeta:
   ```bash
   git clone https://github.com/tu-usuario/WildGuard-Fauna.git
   cd WildGuard-Fauna