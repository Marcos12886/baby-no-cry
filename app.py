import torch
import torchaudio # Librería para procesamiento de audio
import torch.nn.functional as F # Importa la API funcional de torch, incluyendo softmax
import gradio as gr # Gradio para crear interfaces web
from dotenv import load_dotenv
from model import predict_params, AudioDataset # Importaciones personalizadas: carga de modelo y procesamiento de audio
#TODO: estaba quitando lo de los decibelios
load_dotenv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_class, id2label_class = predict_params(
    model_path="distilhubert-finetuned-mixed-data", # Ruta al modelo para la predicción de clases de llanto
    dataset_path="data/mixed_data", # Ruta al dataset de audio mixto
    undersample_normal=True # Activa el submuestreo para equilibrar clases
    )
model_mon, id2label_mon = predict_params(
    model_path="distilhubert-finetuned-cry-detector", # Ruta al modelo detector de llanto
    dataset_path="data/baby_cry_detection", # Ruta al dataset de detección de llanto
    undersample_normal=False # No submuestrear datos
    )
# TODO: mirar si se puede quitar el undersample de aqui
def call(audiopath, model, dataset_path, undersample_normal=False):
    model.to(device) # Envía el modelo a la GPU (o CPU si no hay GPU disponible)
    model.eval() # Pone el modelo en modo de evaluación (desactiva dropout, batchnorm)
    audio_dataset = AudioDataset(dataset_path, {}, undersample_normal) # Carga el dataset de audio con parámetros específicos
    processed_audio = audio_dataset.preprocess_audio(audiopath) # Preprocesa el audio según la configuración del dataset
    inputs = {"input_values": processed_audio.to(device).unsqueeze(0)} # Prepara los datos para el modelo (envía a GPU y ajusta dimensiones)
    with torch.no_grad(): # Desactiva el cálculo del gradiente para ahorrar memoria
        outputs = model(**inputs) # Realiza la inferencia con el modelo
        logits = outputs.logits # Obtiene las predicciones del modelo
    return logits # Retorna los logits (valores sin procesar)

def predict(audio_path_pred):
    with torch.no_grad(): # Desactiva gradientes para la inferencia
        logits = call(audio_path_pred, model=model_class, dataset_path="data/mixed_data", undersample_normal=False) # Llama a la función de inferencia
        predicted_class_ids_class = torch.argmax(logits, dim=-1).item() # Obtiene la clase predicha a partir de los logits
        label_class = id2label_class[predicted_class_ids_class] # Convierte el ID de clase en una etiqueta de texto
        label_mapping = {0: 'Cansancio/Incomodidad', 1: 'Dolor', 2: 'Hambre', 3: 'Problemas para respirar'} # Mapea las etiquetas
        label_class = label_mapping.get(predicted_class_ids_class, label_class) # Si hay una etiqueta personalizada, la usa
    return f"""
        <div style='text-align: center; font-size: 1.5em'>
            <span style='display: inline-block; min-width: 300px;'>{label_class}</span>
        </div>
    """ # Retorna el resultado formateado para mostrar en la interfaz

def predict_stream(audio_path_stream):
    with torch.no_grad(): # Desactiva gradientes durante la inferencia
        logits = call(audio_path_stream, model=model_mon, dataset_path="data/baby_cry_detection", undersample_normal=False) # Llama al modelo de detección de llanto
        probabilities = F.softmax(logits, dim=-1) # Aplica softmax para convertir los logits en probabilidades
        crying_probabilities = probabilities[:, 1] # Obtiene las probabilidades asociadas al llanto
        avg_crying_probability = crying_probabilities.mean()*100 # Calcula la probabilidad promedio de llanto
        if avg_crying_probability < 15: # Si la probabilidad de llanto es menor a un 15%, se predice la razón
            label_class = predict(audio_path_stream) # Llama a la predicción para determinar la razón del llanto
            return f"Está llorando por: {label_class}" # Retorna el resultado indicando por qué llora
        else:
            return "No está llorando" # Si la probabilidad es mayor, indica que no está llorando

def decibelios(audio_path_stream):
    waveform, _ = torchaudio.load(audio_path_stream) # Carga el audio y su forma de onda
    rms = torch.sqrt(torch.mean(torch.square(waveform))) # Calcula el valor RMS del audio
    db_level = 20 * torch.log10(rms + 1e-6).item() # Convierte el RMS en decibelios (añade un pequeño valor para evitar log(0))
    min_db = -80 # Nivel mínimo de decibelios esperado
    max_db = 0 # Nivel máximo de decibelios esperado
    scaled_db_level = (db_level - min_db) / (max_db - min_db) # Escala el nivel de decibelios a un rango entre 0 y 1
    normalized_db_level = scaled_db_level * 100 # Escala el nivel de decibelios a un porcentaje
    return normalized_db_level # Retorna el nivel de decibelios normalizado

def mostrar_decibelios(audio_path_stream, visual_threshold):
    db_level = decibelios(audio_path_stream)# Obtiene el nivel de decibelios del audio
    if db_level > visual_threshold: # Si el nivel de decibelios supera el umbral visual
        status = "Prediciendo..." # Cambia el estado a "Prediciendo"
    else:
        status = "Esperando..." # Si no supera el umbral, indica que está "Esperando"
    return f"""
        <div style='text-align: center; font-size: 1.5em'>
            <span>{status}</span>
            <span style='display: inline-block; min-width: 120px;'>Decibelios: {db_level:.2f}</span>
        </div>
    """ # Retorna una cadena HTML con el estado y el nivel de decibelios

def predict_stream_decib(audio_path_stream, visual_threshold):
    db_level = decibelios(audio_path_stream) # Calcula el nivel de decibelios
    if db_level > visual_threshold: # Si supera el umbral, hace una predicción
        prediction = display_prediction_stream(audio_path_stream) # Llama a la función de predicción
    else:
        prediction = "" # Si no supera el umbral, no muestra predicción
    return f"""
        <div style='text-align: center; font-size: 1.5em; min-height: 2em;'>
            <span style='display: inline-block; min-width: 300px;'>{prediction}</span>
        </div>
    """ # Retorna el resultado o nada si no supera el umbral

def chatbot_config(message, history: list[tuple[str, str]]):
    system_message = "You are a Chatbot specialized in baby health and care." # Mensaje inicial del chatbot
    max_tokens = 512 # Máximo de tokens para la respuesta
    temperature = 0.5 # Controla la aleatoriedad de las respuestas
    top_p = 0.95 # Top-p sampling para filtrar palabras
    messages = [system_message] # Configura el mensaje del sistema para el chatbot
    for val in history: # Añade el historial de la conversación al mensaje
        if val[0]:
            messages.append(val[0]) # Añade los mensajes del usuario
        if val[1]:
            messages.append(val[1]) # Añade las respuestas del asistente
    messages.append(message) # Añade el mensaje actual del usuario
    full_prompt = "/n".join(messages)
    inputs = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)
    ouputs = model.generate(
        inputs,
        max_length=max_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response # Retorna la respuesta generada por el modelo

def cambiar_pestaña():
    return gr.update(visible=False), gr.update(visible=True) # Esta función cambia la visibilidad de las pestañas en la interfaz

def display_prediction(audio, prediction_func):
    prediction = prediction_func(audio) # Llama a la función de predicción para obtener el resultado
    return f"<h3 style='text-align: center; font-size: 1.5em;'>{prediction}</h3>" # Retorna el resultado formateado en HTML

def display_prediction_wrapper(audio):
    return display_prediction(audio, predict) # Envuelve la función de predicción "predict" en la función "display_prediction"

def display_prediction_stream(audio):
    return display_prediction(audio, predict_stream) # Envuelve la función de predicción "predict_stream" en la función "display_prediction"

my_theme = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="green",
    neutral_hue="slate",
    text_size="sm",
    spacing_size="sm",
    font=[gr.themes.GoogleFont('Bubblegum Sans'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=[gr.themes.GoogleFont('Bubblegum Sans'), 'ui-monospace', 'Consolas', 'monospace'],
).set(
    body_background_fill='*neutral_50',
    body_text_color='*neutral_600',
    body_text_size='*text_sm',
    embed_radius='*radius_md',
    shadow_drop='*shadow_spread',
    shadow_spread='*button_shadow_active'
)

with gr.Blocks(theme=my_theme, fill_height=True, fill_width=True) as demo:
    with gr.Column(visible=True) as inicial:
        gr.HTML(
            """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Lobster&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Roboto&display=swap');

            h1 {
                font-family: 'Lobster', cursive;
                font-size: 5em !important;
                text-align: center;
                margin: 0;
            }

            .gr-button {
                background-color: #4CAF50 !important;
                color: white !important;
                border: none;
                padding: 25px 50px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-family: 'Lobster', cursive;
                font-size: 2em !important;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 12px;
            }

            .gr-button:hover {
                background-color: #45a049;
            }
            h2 {
                font-family: 'Lobster', cursive;
                font-size: 3em !important;
                text-align: center;
                margin: 0;
            }
            p.slogan, h4, p, h3 {
                font-family: 'Roboto', sans-serif;
                text-align: center;
            }
            </style>
            <h1>baby-no-cry</h1>
            <h4 style='text-align: center; font-size: 1.5em'>El mejor aliado para el bienestar de tu bebé</h4>
            """
        )
        gr.Markdown(
            "<h4 style='text-align: left; font-size: 1.5em;'>¿Qué es baby-no-cry?</h4>"
            "<p style='text-align: left'>baby-no-cry es un proyecto llevado a cabo por un grupo de estudiantes interesados en el desarrollo de modelos de inteligencia artificial, enfocados específicamente en casos de uso relevantes para ayudar a cuidar a los más pequeños de la casa.</p>"
            "<h4 style='text-align: left; font-size: 1.5em;'>Nuestra misión</h4>"
            "<p style='text-align: left'>Sabemos que la paternidad puede suponer un gran desafío. Nuestra misión es brindarles a todos los padres unas herramientas de última tecnología que los ayuden a navegar esos primeros meses de vida tan cruciales en el desarrollo de sus pequeños.</p>"
            "<h4 style='text-align: left; font-size: 1.5em;'>¿Qué ofrece baby-no-cry?</h4>"
            "<p style='text-align: left'>Chatbot: Pregunta a nuestro asistente que te ayudará con cualquier duda que tengas sobre el cuidado de tu bebé.</p>"
            "<p style='text-align: left'>Predictor: Con nuestro modelo de inteligencia artificial somos capaces de predecir por qué tu bebé está llorando.</p>"
            "<p style='text-align: left'>Monitor: Nuestro monitor no es como otros que hay en el mercado, ya que es capaz de reconocer si un sonido es un llanto del bebé o no; y si está llorando, predice automáticamente la causa. Dándote la tranquilidad de saber siempre qué pasa con tu pequeño, ahorrándote tiempo y horas de sueño.</p>"
        )
        boton_inicial = gr.Button("¡Prueba nuestros modelos!")
    with gr.Column(visible=False) as chatbot: # Columna para la pestaña del chatbot
        gr.Markdown("<h2>Asistente</h2>") # Título de la pestaña del chatbot
        gr.Markdown("<h4 style='text-align: center; font-size: 1.5em'>Pregunta a nuestro asistente cualquier duda que tengas sobre el cuidado de tu bebé</h4>")  # Descripción de la pestaña del chatbot
        gr.ChatInterface(
            chatbot_config, # Función de configuración del chatbot
            theme=my_theme, # Tema personalizado para la interfaz
            submit_btn="Enviar", # Botón de enviar mensaje
            autofocus=True, # Enfocar automáticamente el campo de entrada de texto
            fill_height=True, # Rellenar el espacio verticalmente
        )
        with gr.Row(): # Fila para los botones de cambio de pestaña
            with gr.Column(): # Columna para el botón del predictor
                gr.Markdown("<h2>Predictor</h2>") # Título de la pestaña del chatbot
                boton_predictor = gr.Button("Probar predictor") # Botón para cambiar a la pestaña del predictor
            with gr.Column(): # Columna para el botón del monitor
                gr.Markdown("<h2>Monitor</h2>") # Título de la pestaña del chatbot
                boton_monitor = gr.Button("Probar monitor") # Botón para cambiar a la pestaña del monitor
        boton_volver_inicio = gr.Button("Volver al inicio") # Botón para volver a la pestaña inicial
    with gr.Column(visible=False) as pag_predictor: # Columna para la pestaña del predictor
        gr.Markdown("<h2>Predictor</h2>") # Título de la pestaña del predictor
        gr.Markdown("<h4 style='text-align: center; font-size: 1.5em'>Descubre por qué tu bebé está llorando</h4>") # Descripción de la pestaña del predictor
        audio_input = gr.Audio(
            min_length=1.0, # Duración mínima del audio requerida
            format="wav", # Formato de audio admitido
            label="Baby recorder", # Etiqueta del campo de entrada de audio
            type="filepath", # Tipo de entrada de audio (archivo)
        )
        prediction_output = gr.Markdown() # Salida para mostrar la predicción
        gr.Button("¿Por qué llora?").click(
            display_prediction_wrapper, # Función de predicción para el botón
            inputs=audio_input, # Entrada de audio para la función de predicción
            outputs=gr.Markdown() # Salida para mostrar la predicción
        )
        gr.Button("Volver").click(cambiar_pestaña, outputs=[pag_predictor, chatbot]) # Botón para volver a la pestaña del chatbot
    with gr.Column(visible=False) as pag_monitor: # Columna para la pestaña del monitor
        gr.Markdown("<h2>Monitor</h2>") # Título de la pestaña del monitor
        gr.Markdown("<h4 style='text-align: center; font-size: 1.5em'>Detecta en tiempo real si tu bebé está llorando y por qué</h4>")  # Descripción de la pestaña del monitor
        audio_stream = gr.Audio(
            format="wav", # Formato de audio admitido
            label="Baby recorder", # Etiqueta del campo de entrada de audio
            type="filepath", # Tipo de entrada de audio (archivo)
            streaming=True # Habilitar la transmisión de audio en tiempo real
        )
        threshold_db = gr.Slider(
            minimum=0, # Valor mínimo del umbral de ruido
            maximum=120, # Valor máximo del umbral de ruido
            step=1, # Paso del umbral de ruido
            value=20, # Valor inicial del umbral de ruido
            label="Umbral de ruido para activar la predicción:" # Etiqueta del control deslizante del umbral de ruido
        )
        volver = gr.Button("Volver") # Botón para volver a la pestaña del chatbot
        audio_stream.stream(
            mostrar_decibelios, # Función para mostrar el nivel de decibelios
            inputs=[audio_stream, threshold_db], # Entradas para la función de mostrar decibelios
            outputs=gr.HTML() # Salida para mostrar el nivel de decibelios
        )
        audio_stream.stream(
            predict_stream_decib, # Función para realizar la predicción en tiempo real
            inputs=[audio_stream, threshold_db], # Entradas para la función de predicción en tiempo real
            outputs=gr.HTML() # Salida para mostrar la predicción en tiempo real
        )
        volver.click(cambiar_pestaña, outputs=[pag_monitor, chatbot]) # Botón para volver a la pestaña del chatbot
    boton_inicial.click(cambiar_pestaña, outputs=[inicial, chatbot]) # Botón para cambiar a la pestaña inicial
    boton_volver_inicio.click(cambiar_pestaña, outputs=[chatbot, inicial]) # Botón para volver a la pestaña inicial desde el chatbot
    boton_predictor.click(cambiar_pestaña, outputs=[chatbot, pag_predictor]) # Botón para cambiar a la pestaña del predictor
    boton_monitor.click(cambiar_pestaña, outputs=[chatbot, pag_monitor]) # Botón para cambiar a la pestaña del monitor
demo.launch() # Lanzar la interfaz gráfica
