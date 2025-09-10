
run:
\tstreamlit run app_aceite.py

docker-build:
\tdocker build -t tuusuario/aceite-app -f Dockerfile.aceite .

docker-run:
\tdocker run --rm -p 8501:8501 -e CSV_PATH=/app/precios_aceite_clean.csv tuusuario/aceite-app
