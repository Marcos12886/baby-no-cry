# Utilizar Ubuntu
FROM ubuntu:22.04

# Eliminar warmings de apt
ENV DEBIAN_FRONTEND=noninteractive

# Instalar: python3.10, pip, y soundfile. Limpiar archivos.
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* 

# Linkear pythons (sino, da error)
RUN ln -s /usr/bin/python3 /usr/bin/python 

# Directorio de la app
WORKDIR /archivo

# Actulizar pip
RUN python3.10 -m pip install --upgrade pip 

# Copiar requirements
COPY requirements.txt .

# Actualizar pip e instalar librerias
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt 

# Instalar soundfile
RUN python3.10 -m pip install soundfile

# Copiar el resto de la app
COPY . .

# Exponer el puerto de Gradio, no se si puedes poner otro
EXPOSE 7861

# Variables de entorno
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Run la aplicacion
CMD ["python", "archivo.py"]
