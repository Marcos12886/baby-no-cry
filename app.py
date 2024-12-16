import torch
import gradio as gr
from dotenv import load_dotenv
from model import AudioDataset, train_params

load_dotenv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_class, _, _, id2label_class = train_params(
    dataset_path="data/mixed_data",
    model_type="class",
    )
model_detec, _, _, _ = train_params(
    dataset_path="data/baby_cry_detection",
    model_type="detec",
    )

def call(audiopath, model, dataset_path, undersample_normal=False):
    model.to(device)
    model.eval() # (desactiva dropout, batchnorm)
    audio_dataset = AudioDataset(dataset_path, {}, undersample_normal) # Carga dataset de audio
    processed_audio = audio_dataset.preprocess_audio(audiopath) # Preprocesa el audio
    inputs = {"input_values": processed_audio.to(device).unsqueeze(0)}
    with torch.no_grad(): # Desactivar cálculo del gradiente para ahorrar memoria
        outputs = model(**inputs) # Inferir
        logits = outputs.logits # Obtener las predicciones
    return logits # Logits (valores sin procesar)

def predict(audio_path_pred):
    with torch.no_grad():
        logits = call(audio_path_pred, model=model_class, dataset_path="data/mixed_data", undersample_normal=False)
        predicted_class_ids_class = torch.argmax(logits, dim=-1).item() # Obtener clase
        label_class = id2label_class[predicted_class_ids_class] # Convierte el ID de clase en una etiqueta de texto
        label_mapping = {0: 'Cansancio/Incomodidad', 1: 'Dolor', 2: 'Hambre', 3: 'Problemas para respirar'}
        label_class = label_mapping.get(predicted_class_ids_class, label_class) # Aplicar etiquetas cambiadas
    return f"""
        <div style='text-align: center; font-size: 1.5em'>
            <span style='display: inline-block; min-width: 300px;'>{label_class}</span>
        </div>
    """

def predict_stream(audio_path_stream):
    with torch.no_grad(): # Desactivar gradientes
        logits = call(audio_path_stream, model=model_detec, dataset_path="data/baby_cry_detection", undersample_normal=False)
        probabilities = torch.nn.functional.softmax(logits, dim=-1) # Softmax para convertir logits en probabilidades
        crying_probabilities = probabilities[:, 1] # Obtener probabilidades
        avg_crying_probability = crying_probabilities.mean()*100 # Probabilidad media de llanto
        if avg_crying_probability < 15: # Si probabilidad de predicción > 15 dar la razón
            label_class = predict(audio_path_stream) # Hacer la predicción
            return f"Está llorando por: {label_class}" # Razon lloro
        else:
            return "No está llorando"

def chatbot_config(history, type='messages'):
    system_message = "You are a Chatbot specialized in baby health and care."
    max_tokens = 512 # Máximo de tokens por respuesta
    messages = [system_message] # Mensaje del sistema del chatbot
    for val in history: # Añade el historial de la conversación
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
        temperature=0.5,
        top_p=0.95,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0])
    return response

def cambiar_pestaña():
    return gr.update(visible=False), gr.update(visible=True) # Cambiar que pagina mostrar

my_theme = gr.themes.Soft(
    primary_hue="lime",
    neutral_hue="slate",
    text_size="sm",
    spacing_size="sm",
    font=[gr.themes.GoogleFont('Bubblegum Sans'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=[gr.themes.GoogleFont('Bubblegum Sans'), 'ui-monospace', 'Consolas', 'monospace'],
).set(
    body_background_fill='*neutral_100',
    body_text_color='*neutral_700',
    body_text_size='*text_sm',
    embed_radius='*radius_md',
    shadow_drop='*shadow_spread',
    shadow_spread='*button_shadow_active'
)

with gr.Blocks(theme=my_theme, fill_height=True, fill_width=True) as demo:
    with gr.Column(visible=True) as chatbot:
        gr.Markdown("<h2>Asistente</h2>") # Título chatbot
        gr.Markdown(
            "<h4 style='text-align: center;"
            "font-size: 1.5em'>"
            "Pregunta a nuestro asistente cualquier duda que tengas sobre el cuidado de tu bebé</h4>"
        )
        gr.ChatInterface(
            fn=chatbot_config, # Función de configuración del chatbot
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
            fn=predict, # Función de predicción para el botón
            inputs=audio_input, # Entrada de audio para la función de predicción
            outputs=gr.Markdown() # Salida para mostrar la predicción
        )
        gr.Button("Volver").click(cambiar_pestaña, outputs=[pag_predictor, chatbot]) # Mostrar chatbot
    with gr.Column(visible=False) as pag_monitor: # Columna para la pestaña del monitor
        gr.Markdown("<h2>Monitor</h2>") # Título de la pestaña del monitor
        gr.Markdown(
            "<h4 style='text-align: center;"
            "font-size: 1.5em'>"
            "Detecta en tiempo real si tu bebé está llorando y por qué</h4>"
        )
        audio_stream = gr.Audio(
            streaming=True, # Habilitar la transmisión de audio en tiempo real
            format="wav", # Formato de audio admitido
            label="Baby recorder", # Etiqueta del campo de entrada de audio
            type="filepath", # Tipo de entrada de audio (archivo)
        )
        audio_stream.stream(
            fn=predict_stream, # Función para predecir en tiempo real
            inputs=audio_stream, # Entradas para predecir en tiempo real
            outputs=gr.Markdown() # Salida para mostrar la predicción en tiempo real
        )
        gr.Button("Volver").click(cambiar_pestaña, outputs=[pag_monitor, chatbot]) # Mostrar chatbot
    boton_predictor.click(cambiar_pestaña, outputs=[chatbot, pag_predictor]) # Mostrar predictor
    boton_monitor.click(cambiar_pestaña, outputs=[chatbot, pag_monitor]) # Mostrar monitor
demo.launch() # Lanzar la interfaz gráfica
