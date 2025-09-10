# TFM Aceite de Oliva — App Streamlit

## Estructura
```
app.py
core/data_loader.py
core/utils.py
core/lstm_module.py
core/prophet_module.py
core/stats_module
precios_aceite_clean.csv
requirements.txt
Dockerfile.aceite
```

## Ejecutar local
```bash
pip install -r requirements.txt
streamlit run app.py
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
- Prophet y LSTM generan predicciones y permiten descarga. 

