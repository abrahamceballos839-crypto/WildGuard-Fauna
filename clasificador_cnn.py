import json
import os
import random
import cv2
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

TRADUCCION_ES = {
    "cane": "Perro",
    "cavallo": "Caballo",
    "elefante": "Elefante",
    "farfalla": "Mariposa",
    "gallina": "Gallina",
    "gatto": "Gato",
    "mucca": "Vaca",
    "pecora": "Oveja",
    "ragno": "Araña",
    "scoiattolo": "Ardilla",
}


def leer_imagen_segura(ruta_imagen):
    """Lee imágenes en Windows incluso si tienen ñ, tildes o formato .webp."""
    datos = np.fromfile(ruta_imagen, dtype=np.uint8)
    img = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    return img


class WildGuardCNN:
    def __init__(
        self,
        model_path="models/wildguard_cnn_final.keras",
        metadata_path="models/model_metadata.json",
        normalizar_255=True,
    ):
        self.model = tf.keras.models.load_model(model_path)
        self.normalizar_255 = normalizar_255

        _, h, w, _ = self.model.input_shape
        self.img_size = (w, h)

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

        self.dataset_dir = None
        self.imagenes_por_clase = {}
        try:
            dataset_path = kagglehub.dataset_download("alessiocorrado99/animals10")
            raw_dir = os.path.join(dataset_path, "raw-img")
            self.dataset_dir = raw_dir if os.path.exists(raw_dir) else dataset_path
            self.imagenes_por_clase = self._indexar_imagenes()
        except Exception as e:
            print(f"Aviso: No se indexó Animals-10 ({e}). Solo disponible inferencia directa.")

        self._cache = {}

    def _indexar_imagenes(self):
        indice = {}
        if not self.dataset_dir or not os.path.exists(self.dataset_dir):
            return indice
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

        img = leer_imagen_segura(ruta_imagen)
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
        if not self.imagenes_por_clase:
            raise RuntimeError("No hay imágenes de Animals-10 indexadas.")
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

    def probar_imagenes_nuevas(
        self,
        carpeta="imagenes_nuevas",
        salida_grafica="resultados_rl/prueba_cnn_imagenes_nuevas.png",
    ):
        os.makedirs(carpeta, exist_ok=True)
        archivos = sorted([
            os.path.join(carpeta, f)
            for f in os.listdir(carpeta)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ])

        if not archivos:
            print(f"\n⚠️ La carpeta '{carpeta}/' está vacía.")
            return []

        print(f"\n=== [E3] 3.2 PRUEBA DE INFERENCIA CON {len(archivos)} IMÁGENES NUEVAS ===")
        resultados = []
        n = len(archivos)
        fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.2))
        if n == 1:
            axes = [axes]

        for ax, ruta in zip(axes, archivos):
            especie, conf = self.predecir_imagen(ruta, usar_cache=False)
            nombre_es = TRADUCCION_ES.get(especie, especie)
            archivo_nombre = os.path.basename(ruta)
            resultados.append((archivo_nombre, especie, nombre_es, conf))
            print(f"📷 {archivo_nombre} -> Predicción: {especie} ({nombre_es}) | Confianza: {conf:.2%}")

            img_bgr = leer_imagen_segura(ruta)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(f"{archivo_nombre}\nPred: {nombre_es} ({especie})\nConf: {conf:.1%}", fontsize=9.5)
            ax.axis("off")

        os.makedirs(os.path.dirname(salida_grafica), exist_ok=True)
        plt.suptitle("[E3] 3.2 Validación CNN con Imágenes Externas Nuevas", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(salida_grafica, dpi=150)
        plt.show()
        print(f"✅ Gráfica guardada en: {salida_grafica}")
        return resultados


if __name__ == "__main__":
    print("Cargando modelo CNN exportado (wildguard_cnn_final.keras)...")
    cnn = WildGuardCNN()
    print("Tamaño de entrada del modelo:", cnn.img_size)
    print("Clases cargadas:", cnn.class_names)

    cnn.probar_imagenes_nuevas("imagenes_nuevas")