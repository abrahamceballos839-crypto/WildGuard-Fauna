import json
import os
import random
import cv2
import kagglehub
import numpy as np
import tensorflow as tf


class WildGuardCNN:
    def __init__(
        self,
        model_path="models/wildguard_cnn_final.keras",
        metadata_path="models/model_metadata.json",
        normalizar_255=True,
    ):
        # 1. Cargar la CNN entrenada una sola vez
        self.model = tf.keras.models.load_model(model_path)
        self.normalizar_255 = normalizar_255

        # Detectar tamaño de imagen que espera el modelo (alto, ancho)
        _, h, w, _ = self.model.input_shape
        self.img_size = (w, h)

        # 2. Leer metadatos de Brandon si existen
        self.metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        self.class_names = self.metadata.get(
            "classes",
            self.metadata.get(
                "class_names",
                [
                    "cane", "cavallo", "elefante", "farfalla", "gallina",
                    "gatto", "mucca", "pecora", "ragno", "scoiattolo",
                ],
            ),
        )

        # 3. Descargar/ubicar el dataset Animals-10 en tu compu
        dataset_path = kagglehub.dataset_download("alessiocorrado99/animals10")
        raw_dir = os.path.join(dataset_path, "raw-img")
        self.dataset_dir = raw_dir if os.path.exists(raw_dir) else dataset_path

        self.imagenes_por_clase = self._indexar_imagenes()
        self._cache = {}

    def _indexar_imagenes(self):
        indice = {}
        for folder in sorted(os.listdir(self.dataset_dir)):
            full_dir = os.path.join(self.dataset_dir, folder)
            if os.path.isdir(full_dir):
                archivos = [
                    os.path.join(full_dir, f)
                    for f in os.listdir(full_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
                if archivos:
                    indice[folder] = archivos
        return indice

    def predecir_imagen(self, ruta_imagen, usar_cache=True):
        if usar_cache and ruta_imagen in self._cache:
            return self._cache[ruta_imagen]

        img = cv2.imread(ruta_imagen)
        if img is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {ruta_imagen}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        arr = img.astype("float32")
        if self.normalizar_255:
            arr = arr / 255.0
        arr = np.expand_dims(arr, axis=0)

        preds = self.model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(preds))
        confianza = float(preds[idx])
        especie = self.class_names[idx]

        if usar_cache:
            self._cache[ruta_imagen] = (especie, confianza)
        return especie, confianza

    def tomar_foto_aleatoria(self, clase_real=None, usar_cache=True):
        if clase_real is None or clase_real not in self.imagenes_por_clase:
            clase_real = random.choice(list(self.imagenes_por_clase.keys()))
        ruta = random.choice(self.imagenes_por_clase[clase_real])
        especie_pred, confianza = self.predecir_imagen(ruta, usar_cache=usar_cache)
        return {
            "clase_real": clase_real,
            "especie_predicha": especie_pred,
            "confianza": confianza,
            "acierto": especie_pred == clase_real,
            "ruta": ruta,
        }


if __name__ == "__main__":
    print("Cargando CNN y descargando/verificando Animals-10...")
    cnn = WildGuardCNN()
    print("Tamaño de entrada del modelo:", cnn.img_size)
    print("Clases cargadas:", cnn.class_names)
    for i in range(3):
        res = cnn.tomar_foto_aleatoria(usar_cache=False)
        print(
            f"Prueba {i+1} -> Real: {res['clase_real']} | "
            f"Predicha: {res['especie_predicha']} | "
            f"Confianza: {res['confianza']:.2%} | Acierto: {res['acierto']}"
        )