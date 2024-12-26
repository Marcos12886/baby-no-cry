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
    model.eval()
    audio_dataset = AudioDataset(dataset_path, {}, undersample_normal)
    processed_audio = audio_dataset.preprocess_audio(audiopath)
    inputs = {"input_values": processed_audio.to(device).unsqueeze(0)}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    return logits

def predict(audio_path_pred):
    with torch.no_grad():
        logits = call(audio_path_pred, model=model_class, dataset_path="data/mixed_data", undersample_normal=False)
        predicted_class_ids_class = torch.argmax(logits, dim=-1).item()
        label_class = id2label_class[predicted_class_ids_class]
        label_mapping = {0: 'Cansancio/Incomodidad', 1: 'Dolor', 2: 'Hambre', 3: 'Problemas para respirar'}
        label_class = label_mapping.get(predicted_class_ids_class, label_class)
    return f"""
        <div style='text-align: center; font-size: 1.5em'>
            <span style='display: inline-block; min-width: 300px;'>{label_class}</span>
        </div>
    """

def predict_stream(audio_path_stream):
    with torch.no_grad():
        logits = call(audio_path_stream, model=model_detec, dataset_path="data/baby_cry_detection", undersample_normal=False)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        crying_probabilities = probabilities[:, 1]
        avg_crying_probability = crying_probabilities.mean()*100
        if avg_crying_probability < 15:
            label_class = predict(audio_path_stream)
            return f"Está llorando por: {label_class}"
        else:
            return "No está llorando"

def chatbot_config(history, type='messages'):
    system_message = "You are a Chatbot specialized in baby health and care."
    max_tokens = 512
    messages = [system_message]
    for val in history:
        if val[0]:
            messages.append(val[0])
        if val[1]:
            messages.append(val[1])
    messages.append(message)
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
    return gr.update(visible=False), gr.update(visible=True)

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
        gr.Markdown("<h2>Asistente</h2>")
        gr.Markdown(
            "<h4 style='text-align: center;"
            "font-size: 1.5em'>"
            "Pregunta a nuestro asistente cualquier duda que tengas sobre el cuidado de tu bebé</h4>"
        )
        gr.ChatInterface(
            fn=chatbot_config,
            submit_btn="Enviar",
            autofocus=True,
            fill_height=True,
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("<h2>Predictor</h2>")
                boton_predictor = gr.Button("Probar predictor")
            with gr.Column():
                gr.Markdown("<h2>Monitor</h2>")
                boton_monitor = gr.Button("Probar monitor")
    with gr.Column(visible=False) as pag_predictor:
        gr.Markdown("<h2>Predictor</h2>")
        gr.Markdown(
            "<h4 style='text-align: center;"
            "font-size: 1.5em'>"
            "Descubre por qué tu bebé está llorando</h4>"
        )
        audio_input = gr.Audio(
            min_length=1.0,
            format="wav",
            label="Baby recorder",
            type="filepath",
        )
        gr.Button("¿Por qué llora?").click(
            fn=predict,
            inputs=audio_input,
            outputs=gr.Markdown()
        )
        gr.Button("Volver").click(cambiar_pestaña, outputs=[pag_predictor, chatbot])
    with gr.Column(visible=False) as pag_monitor:
        gr.Markdown("<h2>Monitor</h2>")
        gr.Markdown(
            "<h4 style='text-align: center;"
            "font-size: 1.5em'>"
            "Detecta en tiempo real si tu bebé está llorando y por qué</h4>"
        )
        audio_stream = gr.Audio(
            streaming=True,
            format="wav",
            label="Baby recorder",
            type="filepath",
        )
        audio_stream.stream(
            fn=predict_stream,
            inputs=audio_stream,
            outputs=gr.Markdown()
        )
        gr.Button("Volver").click(cambiar_pestaña, outputs=[pag_monitor, chatbot])
    boton_predictor.click(cambiar_pestaña, outputs=[chatbot, pag_predictor])
    boton_monitor.click(cambiar_pestaña, outputs=[chatbot, pag_monitor])
demo.launch()
