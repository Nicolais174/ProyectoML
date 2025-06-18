import os
import pandas as pd
import numpy as np
import wfdb
from scipy.stats import skew, kurtosis
from biosppy.signals import ecg

# Ruta a la carpeta donde están los archivos .hea y .mat
carpeta_datos = "C:/Users/56984/OneDrive - Universidad de Talca/Escritorio/training2017"
archivo_referencia = "C:/Users/56984/OneDrive - Universidad de Talca/Escritorio/REFERENCE.csv"

# 1. Leer archivo de referencia
df_ref = pd.read_csv(archivo_referencia, header=None, names=["id", "label"])
df_ref['id'] = df_ref['id'].str.strip()  # quitar espacios si existen

# 2. Inicializar lista para almacenar los datos procesados
data = []

# 3. Recorrer archivos en la carpeta
for archivo in os.listdir(carpeta_datos):
    if archivo.endswith(".hea"):
        id_registro = archivo.replace(".hea", "")
        ruta_base = os.path.join(carpeta_datos, id_registro)
        
        try:
            # Leer el registro con wfdb
            record = wfdb.rdrecord(ruta_base)
            signal = record.p_signal[:, 0]  # usar canal 0 (asumiendo que es el ECG)
            fs = record.fs  # frecuencia de muestreo

            # Usar biosppy para procesar y detectar picos R
            out = ecg.ecg(signal=signal, sampling_rate=fs, show=False)
            rpeaks = out['rpeaks']  # índices de los picos R detectados

            if len(rpeaks) < 3:
                continue  # omitir señales con pocos RR detectables

            # Calcular los intervalos RR en milisegundos
            rr_intervals = np.diff(rpeaks) * (1000 / fs)

            # Calcular características
            mean_rr = np.mean(rr_intervals)
            std_rr = np.std(rr_intervals)
            skew_rr = skew(rr_intervals)
            kurt_rr = kurtosis(rr_intervals)

            # Buscar la etiqueta
            fila = df_ref[df_ref['id'] == id_registro]
            if fila.empty:
                continue

            label = fila['label'].values[0]
            if label not in ['N', 'A']:
                continue  # descartar clases O, ~

            # Agregar al dataset
            data.append({
                "id": id_registro,
                "mean_rr": mean_rr,
                "std_rr": std_rr,
                "skew_rr": skew_rr,
                "kurt_rr": kurt_rr,
                "label": "Normal" if label == "N" else "AFib"
            })

        except Exception as e:
            print(f"⚠️ Error procesando {id_registro}: {e}")
            continue

# 4. Crear DataFrame final
df_final = pd.DataFrame(data)

# 5. Guardar como CSV
df_final.to_csv("ecg_rr_features_curado.csv", index=False)

print("✅ Total registros procesados:", len(df_final))
print("📊 Ejemplo de datos:")
print(df_final.head())
