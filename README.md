# WildGuard-Fauna 🐾
Dron de conservación para identificar fauna silvestre.
Proyecto final de Aprendizaje Automático (UTL, 2026).

## Equipo
- Jasciel: Líder de proyecto y datos
- Osvaldo: ML clásico (supervisado)
- Betzie: ML clásico (no supervisado y métricas)
- Brandon: CNN
- Óscar: Refuerzo e integración

## Estado del proyecto
Entrega 1: datos, EDA inicial, partición y modelos clásicos supervisados.
Las entregas 2 (CNN), 3 (refuerzo) y final (integración) se agregan en las
siguientes semanas.

## Dataset
Animals-10 (Kaggle, 26,179 imágenes, 10 clases).
Descárgalo y colócalo en `./dataset_animals10/` (una carpeta por clase).
No se sube al repo por su tamaño. También puedes descargarlo usando `kagglehub` (ver notebook).

Clases (carpetas en italiano): cane=perro, cavallo=caballo,
elefante=elefante, farfalla=mariposa, gallina=pollo, gatto=gato,
mucca=vaca, pecora=oveja, ragno=araña, scoiattolo=ardilla.

## Estructura del repositorio
- `processed_data/` — CSV de partición (train/val/test) y archivos derivados
- `saved_models/` — modelos entrenados (no se suben los .pkl)
- `dataset_animals10/` — dataset local (no se sube al repo)
- Notebooks: `WildGuard-Fauna.ipynb`

## Instalación
```bash
pip install -r requirements.txt