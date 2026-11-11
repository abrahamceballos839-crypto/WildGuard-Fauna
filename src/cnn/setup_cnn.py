import os
import pandas as pd

def verificar_integracion_datos():
    """
    Verifica que los archivos CSV generados por Jasciel en processed_data
    esten listos para ser leidos por los generadores de la CNN.
    """
    train_path = "./processed_data/train_split.csv"
    val_path = "./processed_data/val_split.csv"
    test_path = "./processed_data/test_split.csv"
    
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)
        
        print("--- Integracion de Datos para CNN Verificada ---")
        print(f"Imagenes de entrenamiento cargadas: {len(df_train)}")
        print(f"Imagenes de validacion cargadas:    {len(df_val)}")
        print(f"Imagenes de prueba cargadas:        {len(df_test)}")
        return True
    else:
        print("Error: Los archivos CSV en ./processed_data/ no estan listos.")
        return False

if __name__ == "__main__":
    verificar_integracion_datos()