import pandas as pd
import yaml
from src.preprocessing import preprocess_data


def export():
    # 1. Cargar la configuración para saber qué archivos procesar
    config_path = "config/modelo_mar_bueno.yaml"  # o el que estés usando
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    print("Iniciando procesamiento de datos...")

    # 2. Ejecutar el preprocesamiento de tu src/preprocessing.py
    data = preprocess_data(
        filepath_train=config["data_path"],
        filepath_test=config["test_data_path"],
        target_col=config["target_col"],
    )

    # 3. Extraer el set de entrenamiento completo y escalado
    # 'X_train_full_final' contiene todas las filas procesadas y escaladas
    df_final = data["X_train_full_final"].copy()

    # Añadimos la columna objetivo (y) de nuevo al dataframe
    df_final[config["target_col"]] = data["y_train_full"].values

    # 4. Guardar a CSV
    output_path = "data/processed/dataset_final_preprocessed.csv"
    df_final.to_csv(output_path, index=False)

    # También puedes exportar el test si lo necesitas
    data["X_test"].to_csv("data/processed/test_final_preprocessed.csv", index=False)

    print(f"¡Éxito! Dataset guardado en: {output_path}")
    print(f"Dimensiones del dataset: {df_final.shape}")


if __name__ == "__main__":
    export()
