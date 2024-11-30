import torch
import gradio as gr
from dotenv import load_dotenv
from model import predict_params, AudioDataset

load_dotenv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_class, id2label_class = predict_params(
    model_path="distilhubert-finetuned-mixed-data", # Ruta al modelo para la predicción de clases de llanto
    dataset_path="data/mixed_data", # Ruta al dataset de audio mixto
    undersample_normal=True # Activa el submuestreo para equilibrar clases
    )
model_detec, id2label_mon = predict_params(
    model_path="distilhubert-finetuned-cry-detector", # Ruta al modelo detector de llanto
    dataset_path="data/baby_cry_detection", # Ruta al dataset de detección de llanto
    undersample_normal=False # No submuestrear datos
    )

def call(audiopath, model, dataset_path, undersample_normal=False):
    model.to(device) # Envía el modelo a la GPU (o CPU si no hay GPU disponible)
    model.eval() # Pone el modelo en modo de evaluación (desactiva dropout, batchnorm)
    audio_dataset = AudioDataset(dataset_path, {}, undersample_normal) # Carga el dataset de audio con parámetros específicos
    processed_audio = audio_dataset.preprocess_audio(audiopath) # Preprocesa el audio según la configuración del dataset
    inputs = {"input_values": processed_audio.to(device).unsqueeze(0)}
    with torch.no_grad(): # Desactiva el cálculo del gradiente para ahorrar memoria
        outputs = model(**inputs) # Realiza la inferencia con el modelo
        logits = outputs.logits # Obtiene las predicciones del modelo
    return logits # Retorna los logits (valores sin procesar)

def predict(audio_path_pred):
    with torch.no_grad():
        logits = call(audio_path_pred, model=model_class, dataset_path="data/mixed_data", undersample_normal=False)
        predicted_class_ids_class = torch.argmax(logits, dim=-1).item() # Obtiene la clase predicha a partir de los logits
        label_class = id2label_class[predicted_class_ids_class] # Convierte el ID de clase en una etiqueta de texto
        label_mapping = {0: 'Cansancio/Incomodidad', 1: 'Dolor', 2: 'Hambre', 3: 'Problemas para respirar'} # Mapea las etiquetas
        label_class = label_mapping.get(predicted_class_ids_class, label_class) # Si hay una etiqueta personalizada, la usa
    return f"""
        <div style='text-align: center; font-size: 1.5em'>
            <span style='display: inline-block; min-width: 300px;'>{label_class}</span>
        </div>
    """

def predict_stream(audio_path_stream):
    with torch.no_grad(): # Desactiva gradientes durante la inferencia
        logits = call(audio_path_stream, model=model_detec, dataset_path="data/baby_cry_detection", undersample_normal=False)
        probabilities = torch.nn.functional.softmax(logits, dim=-1) # Aplica softmax para convertir los logits en probabilidades
        crying_probabilities = probabilities[:, 1] # Obtiene las probabilidades asociadas al llanto
        avg_crying_probability = crying_probabilities.mean()*100 # Calcula la probabilidad promedio de llanto
        if avg_crying_probability < 15: # Si la probabilidad de llanto es menor a un 15%, se predice la razón
            label_class = predict(audio_path_stream) # Llama a la predicción para determinar la razón del llanto
            return f"Está llorando por: {label_class}" # Retorna el resultado indicando por qué llora
        else:
            return "No está llorando"

def chatbot_config(history, type='messages'):
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
    response = tokenizer.decode(outputs[0])
    return response # Retorna la respuesta generada por el modelo

def cambiar_pestaña():
    return gr.update(visible=False), gr.update(visible=True) # Esta función cambia la visibilidad de las pestañas en la interfaz

my_theme = gr.themes.Soft(
    primary_hue="lime",  # Light purple for a calming effect
    neutral_hue="slate",
    text_size="sm",
    spacing_size="sm",
    font=[gr.themes.GoogleFont('Bubblegum Sans'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=[gr.themes.GoogleFont('Bubblegum Sans'), 'ui-monospace', 'Consolas', 'monospace'],
).set(
    body_background_fill='*neutral_100',  # Lighter background
    body_text_color='*neutral_700',  # Slightly darker text for better readability
    body_text_size='*text_sm',
    embed_radius='*radius_md',
    shadow_drop='*shadow_spread',
    shadow_spread='*button_shadow_active'
)

with gr.Blocks(theme=my_theme, fill_height=True, fill_width=True) as demo:
    with gr.Column(visible=True) as chatbot: # Columna para la pestaña del chatbot
        gr.Markdown("<h2>Asistente</h2>") # Título de la pestaña del chatbot
        gr.Markdown(
            "<h4 style='text-align: center;"
            "font-size: 1.5em'>"
            "Pregunta a nuestro asistente cualquier duda que tengas sobre el cuidado de tu bebé</h4>"
        )
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
    with gr.Column(visible=False) as pag_predictor: # Columna para la pestaña del predictor
        gr.Markdown("<h2>Predictor</h2>") # Título de la pestaña del predictor
        gr.Markdown(
            "<h4 style='text-align: center;"
            "font-size: 1.5em'>"
            "Descubre por qué tu bebé está llorando</h4>"
        )
        audio_input = gr.Audio(
            min_length=1.0, # Duración mínima del audio requerida
            format="wav", # Formato de audio admitido
            label="Baby recorder", # Etiqueta del campo de entrada de audio
            type="filepath", # Tipo de entrada de audio (archivo)
        )
        gr.Button("¿Por qué llora?").click(
            predict, # Función de predicción para el botón
            inputs=audio_input, # Entrada de audio para la función de predicción
            outputs=gr.Markdown() # Salida para mostrar la predicción
        )
        gr.Button("Volver").click(cambiar_pestaña, outputs=[pag_predictor, chatbot]) # Botón para volver a la pestaña del chatbot
    with gr.Column(visible=False) as pag_monitor: # Columna para la pestaña del monitor
        gr.Markdown("<h2>Monitor</h2>") # Título de la pestaña del monitor
        gr.Markdown(
            "<h4 style='text-align: center;"
            "font-size: 1.5em'>"
            "Detecta en tiempo real si tu bebé está llorando y por qué</h4>"
        )
        audio_stream = gr.Audio(
            format="wav", # Formato de audio admitido
            label="Baby recorder", # Etiqueta del campo de entrada de audio
            type="filepath", # Tipo de entrada de audio (archivo)
            streaming=True # Habilitar la transmisión de audio en tiempo real
        )
        audio_stream.stream(
            predict_stream, # Función para realizar la predicción en tiempo real
            inputs=audio_stream, # Entradas para la función de predicción en tiempo real
            outputs=gr.HTML() # Salida para mostrar la predicción en tiempo real
        )
        gr.Button("Volver").click(cambiar_pestaña, outputs=[pag_monitor, chatbot]) # Botón para volver a la pestaña del chatbot
    boton_predictor.click(cambiar_pestaña, outputs=[chatbot, pag_predictor]) # Botón para cambiar a la pestaña del predictor
    boton_monitor.click(cambiar_pestaña, outputs=[chatbot, pag_monitor]) # Botón para cambiar a la pestaña del monitor
demo.launch() # Lanzar la interfaz gráfica
