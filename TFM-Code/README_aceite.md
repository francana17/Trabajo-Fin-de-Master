# TFM Aceite de Oliva — App Streamlit

## Estructura
```
app_aceite.py
model_building_LSTM.py
model_building_prophet.py
model_building_Forecast.py
precios_aceite_clean.csv
requirements_aceite.txt
Dockerfile.aceite
```

## Ejecutar local
```bash
pip install -r requirements_aceite.txt
streamlit run app_aceite.py
```

## Docker
```bash
docker build -t tuusuario/aceite-app -f Dockerfile.aceite .
docker run --rm -p 8501:8501 -e CSV_PATH=/app/precios_aceite_clean.csv tuusuario/aceite-app
```
Puedes montar otro CSV:
```bash
docker run --rm -p 8501:8501 -v $PWD/mi.csv:/data/mi.csv -e CSV_PATH=/data/mi.csv tuusuario/aceite-app
```

## Notas
- La app busca `fecha` o `date` y las columnas de precio `virgen_extra_picual`, `virgen_picual`, `lampante_picual`.
- Prophet y LSTM generan predicciones y permiten descarga. PyCaret entrena el *blender* (para predicción/plots detallados, extiende el módulo).
- Si quieres imágenes separadas por modelo, quita dependencias no usadas de `requirements_aceite.txt`.
