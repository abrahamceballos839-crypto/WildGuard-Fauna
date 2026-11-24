# WildGuard-Fauna 🐾
Dron de conservación para identificar fauna silvestre.
Proyecto final de Aprendizaje Automático (UTL, 2026).

## Equipo
- **Jasciel:** Líder de proyecto y datos
- **Osvaldo:** ML clásico (supervisado)
- **Betzie:** ML clásico (no supervisado y métricas)
- **Brandon:** CNN
- **Óscar:** Refuerzo e integración

## Estado del proyecto
- **Entrega 1:** Datos, EDA inicial, partición y modelos clásicos supervisados.
- **Entregas posteriores:** CNN (Entrega 2), aprendizaje por refuerzo (Entrega 3) e integración final agregadas en las siguientes semanas.

## Dataset
**Animals-10** (Kaggle, 26,179 imágenes, 10 clases).  
*Nota:* No se sube al repositorio debido a su tamaño. Debes descargarlo y colocarlo en `./dataset_animals10/` (una carpeta por clase) o usar la descarga automática mediante `kagglehub` desde el notebook.

**Clases (carpetas en italiano):**
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

## Estructura del Repositorio
- `processed_data/` — CSVs de partición (train/val/test) y archivos derivados.
- `saved_models/` — Modelos entrenados (los archivos pesados `.pkl` están excluidos del control de versiones).
- `dataset_animals10/` — Dataset local (excluido del repositorio).
- Notebooks principales de trabajo (ej. `WildGuard-Fauna.ipynb`, `WildGuard-Fauna2.ipynb`).

## Instalación
1. Clona el repositorio y entra a la carpeta:
   ```bash
   git clone [https://github.com/tu-usuario/WildGuard-Fauna.git](https://github.com/tu-usuario/WildGuard-Fauna.git)
   cd WildGuard-Fauna