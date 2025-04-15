FROM tensorflow/tensorflow:latest

# instalar dependencias del proyecto
COPY requirements.txt /tf/
RUN pip install -r /tf/requirements.txt


# Establece el directorio de trabajo dentro del contenedor
WORKDIR /tf

# Copia el contenido del proyecto (opcional, porque montamos volumen igual)
COPY . /tf
