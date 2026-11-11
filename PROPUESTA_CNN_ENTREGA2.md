# Propuesta Técnica de Arquitecturas CNN y Estructura del Módulo (Entrega 2)
**Responsable:** Brandon (Responsable de Redes Neuronales Convolucionales)
**Proyecto:** WildGuard - Dron de Conservación para Identificar Fauna Silvestre
**Fecha:** 11 de Noviembre de 2026

---

## 1. Estructura de Directorios Propuesta para CNN

```text
WildGuard-Fauna/
├── processed_data/              # Generados en E1 (train_split.csv, val_split.csv, test_split.csv)
├── saved_models/                # Modelos exportados (.h5 / .keras)
├── src/
│   └── cnn/
│       ├── __init__.py
│       ├── data_loader.py       # Carga de tensores a partir de los CSV
│       ├── architectures.py     # Definición de CNN propia y modelos preentrenados
│       ├── training.py          # Pipeline de entrenamiento y callbacks
│       └── gradcam.py           # Funciones para interpretabilidad visual
└── notebooks/
    └── 02_cnn_experiments.ipynb # Cuaderno ejecutable principal para la Entrega 2