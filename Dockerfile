# Utilizar Ubuntu
FROM ubuntu:22.04

# Actualizar paquetes y descargar soundfile
RUN apt-get update && apt-get install -y \
	python3 python3-pip \
	libsndfile1 

# Linkear pythons (sino, da error)
RUN ln -s /usr/bin/python3 /usr/bin/python 

# Directorio de la app
WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Actualizar pip e instalar librerias
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt 

# Instalar soundfile
RUN pip install soundfile 

# Copiar el resto de la app
COPY . .

# Exponer el puerto de Gradio, no se si puedes poner otro
EXPOSE 7860

# Variables de entorno
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Run la aplicacion
CMD ["python", "app.py"]
