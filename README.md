## Instalación
- Versión Python==3.10.14, gráfica: NVIDIA.
- pip install -r requirements.txt

## Estructura
Funcionalidades:
- Clasificador de llantos: conocer por qué llora tu bebé
- Monitor de bebés: identificar si tu bebé está llorando y por qué
- Chatbot: poder hablar con un LLM sobre las preocupaciones con tu bebé

Entrenar modelos:
- Detector: [model.py](model.py) --n detec
- Predictor: [model.py](model.py) --n class

- Chatbot: [app.py](app.py)

Colaboradores:
- Roberto Martín. Creó varias funciones, entre ellas: función para filtrar por decibelios, ...
- Felipe González. Eligió: la fuente y colores de la aplicación.
