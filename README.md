## Instalación
La versión de Python utilizada es la 3.10.14, modelos entrenados con un gráfica de NVIDIA.

Instalar con: pip install -r requirements.txt

## Estructura
Funcionalidades:
- Clasificador de llantos: conocer por qué llora tu bebé
- Monitor de bebés: identificar si tu bebé llora y por qué
- Chatbot: poder hablar con un llm sobre las preocupaciones con tu bebé

Flujo de archivos:
1. Construir la estructura de los modelos y entrenarlos [model.py](model.py)
2. Chatbot en el que grabar audio y conectar con llama 3 8B [app.py](app.py)

Un modelo ([model.py](model.py)) entrenado con distintos datos:
- Modelo DetectorPredictor.py --n detec
- Modelo Predictor: python model.py --n class

Chatbot [app.py](app.py)

Colaboradores:
- Roberto Martín: creó varias funciones, entre ellas: función para filtrar por decibelios, filtrar por ruido blanco, ...
- Felipe González: diseñó el logo y eligió la fuente y colores de la aplicación.
