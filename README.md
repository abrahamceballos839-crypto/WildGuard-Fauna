# WildGuard-Fauna 🐾
Dron de conservación para identificar fauna silvestre.
Proyecto final de Aprendizaje Automático (UTL, 2026).

## Equipo
- Jasciel: Líder de proyecto y datos
- Osvaldo: ML clásico (supervisado)
- Betzie: ML clásico (no supervisado y métricas)
- Brandon: CNN
- Óscar: Refuerzo e integración

## Dataset
Animals-10 (Kaggle, 26,179 imágenes, 10 clases).
Descárgalo y colócalo en `./dataset_animals10/` (una carpeta por clase).
No se sube al repo por su tamaño.
Clases (carpetas en italiano): cane=perro, cavallo=caballo,
elefante=elefante, farfalla=mariposa, gallina=pollo, gatto=gato,
mucca=vaca, pecora=oveja, ragno=araña, scoiattolo=ardilla.

## Instalación
pip install -r requirements.txt

## Reproducibilidad
Semilla aleatoria fija: 42 (partición, modelos y NumPy).
Partición 70/15/15 estratificada en `processed_data/*.csv`.

## Orden de ejecución
1. Partición de datos (Jasciel)
2. Características y modelos clásicos (Osvaldo)
3. Agrupamiento y métricas (Betzie)
