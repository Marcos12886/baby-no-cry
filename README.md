## Instalación
- Versión Python==3.10.14, gráfica: NVIDIA.
- pip install -r requirements.txt

## Estructura
Funcionalidades:
- Clasificador de llantos: conocer por qué llora tu bebé
- Monitor de bebés: identificar si tu bebé está llorando y por qué
- Chatbot: poder hablar con un LLM sobre las preocupaciones de tu bebé

Entrenar modelos:
- Detector: [model.py](model.py) --n detec
- Predictor: [model.py](model.py) --n class
- Chatbot: [app.py](app.py)

Parámetros de entrenamiento: [config.json](config.json)
