# Proyecto de Segmentación Semántica de Tumores Cerebrales
Este proyecto utiliza modelos de redes neuronales profundas para segmentar tumores cerebrales en imágenes de resonancia magnética (MRI). Incluye el entrenamiento de modelos con diferentes funciones de pérdida (Cross Entropy y Focal Loss) y una aplicación interactiva para visualizar las predicciones.

## Estructura del proyecto
- data/: Contiene el código de descarga del dataset desde Kaggle y algunas imágenes con sus máscaras correspondientes para probar la interfaz.

- dev/: 
    - globalSegmentacionTumores.ipynb: contiene el notebook en donde se desarrolló el entrenamiento de los modelos y la elección del mejor modelo para la interfaz, utilizando métricas.
    - statistics.py: script que elabora los gráficos de pérdida.
    - loss_ce.png: gráfico de la evolución de la pérdida en los modelos en el entrenamiento utilizando Cross-Entropy Loss.
    - loss_fl.png: gráfico de la evolución de la pérdida en los modelos en el entrenamiento utilizando Focal Loss.
    - models_data.csv: datos utilizados para elaborar los gráficos de pérdida

- prod/: Código preparado para el entorno de producción.
    - app.py: Aplicación principal desarrollada en Streamlit.
    - utils.py: Funciones auxiliares, como carga de modelos, preprocesamiento, postprocesamiento y graficación.
    - model.pth: Archivo del modelo entrenado, correspondiente al modelo 2 FCN ResNet101 entrenado con Cross-Entropy Loss.
    - requirements.txt: Dependencias necesarias para ejecutar el proyecto.

## Uso
1. Clonar el repositorio:
 ```bash
 git clone https://github.com/virlop/RNP-global.git
  ```
2. Instalar dependencias:
 ```bash
pip install -r requirements.txt
 ```
3. Ejecutar app de streamlit:
 ```bash
streamlit run prod\app.py
 ```