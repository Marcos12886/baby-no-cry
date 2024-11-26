import os
import json
import random
import argparse
import torch
import torchaudio
from dotenv import load_dotenv
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from huggingface_hub import upload_folder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import Counter
from transformers import (
    Wav2Vec2FeatureExtractor, HubertConfig, HubertForSequenceClassification,
    Trainer, TrainingArguments
    )

load_dotenv() # Cargar variables de entorno
MODEL = "ntu-spml/distilhubert" # Nombre del modelo base utilizado
FEATURE_EXTRACTOR = Wav2Vec2FeatureExtractor.from_pretrained(MODEL) # Extracción de características del modelo base
seed = 123 # Semilla para reproducibilidad
MAX_DURATION = 1.00 # Máxima duración de los audios
SAMPLING_RATE = FEATURE_EXTRACTOR.sampling_rate # 16kHz
config_file = "models_config.json" # Archivo con los argumentos de entrenamiento
batch_size = 1024 # Tamaño del batch para hacer el split solo 1 vez 
num_workers = 12 # Número de núcleos de CPU utilizados en la carga de datos

class AudioDataset(Dataset):
    """Dataset de audios"""
    def __init__(self, dataset_path, label2id, filter_white_noise, undersample_normal):
        """Inicializar dataset"""
        self.dataset_path = dataset_path
        self.label2id = label2id
        self.file_paths = []
        self.filter_white_noise = filter_white_noise
        self.labels = []
        # Recorremos los directorios de etiquetas y asignamos sus archivos
        for label_dir, label_id in self.label2id.items():
            label_path = os.path.join(self.dataset_path, label_dir)
            if os.path.isdir(label_path): # Verificamos que sea un directorio válido
                for file_name in os.listdir(label_path):
                    audio_path = os.path.join(label_path, file_name)
                    self.file_paths.append(audio_path)
                    self.labels.append(label_id)
        # Submuestreamos la clase normal para el clasficador
        if undersample_normal and self.label2id:
            self.undersample_normal_class()

    def undersample_normal_class(self):
        """Método para submuestrear la clase normal y equilibrar el dataset"""
        normal_label = self.label2id.get('1s_normal')
        label_counts = Counter(self.labels)
        other_counts = [count for label, count in label_counts.items() if label != normal_label]
        if other_counts:
            target_count = max(other_counts) # Establecemos el tamaño objetivo igual al mayor número de ejemplos no normales
            normal_indices = [i for i, label in enumerate(self.labels) if label == normal_label]
            keep_indices = random.sample(normal_indices, target_count)
            new_file_paths = []
            new_labels = []
            for i, (path, label) in enumerate(zip(self.file_paths, self.labels)):
                if label != normal_label or i in keep_indices: # Mantenemos solo los ejemplos necesarios
                    new_file_paths.append(path)
                    new_labels.append(label)
            self.file_paths = new_file_paths
            self.labels = new_labels

    def __len__(self):
        """Devuelve el número de audios en el dataset"""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """Obtiene un audio y su etiqueta correspondiente"""
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        input_values = self.preprocess_audio(audio_path) # Preprocesa el audio
        return {
            "input_values": input_values,
            "labels": torch.tensor(label) # Convertimos la etiqueta en tensor
        }

    def preprocess_audio(self, audio_path):
        """Preprocesamiento de los archivos de audio"""
        waveform, sample_rate = torchaudio.load(
            audio_path,
            normalize=True # Convertir a float32
            )
        if sample_rate != SAMPLING_RATE: # Resamplear si no es 16kHz
            resampler = torchaudio.transforms.Resample(sample_rate, SAMPLING_RATE)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1: # Convertimos a mono si es estéreo
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-6) # Normalizar evitando dividir por 0
        max_length = int(SAMPLING_RATE * MAX_DURATION)
        if waveform.shape[1] > max_length:
            waveform = waveform[:, :max_length] # Truncamos si supera la duración máxima
        else:
            waveform = torch.nn.functional.pad(waveform, (0, max_length - waveform.shape[1])) # Padding si audio más corto
        inputs = FEATURE_EXTRACTOR(
            waveform.squeeze(), # Aplanamos el tensor a una dimensión
            sampling_rate=SAMPLING_RATE, # Nos aseguramos que sea 16kHz
            return_tensors="pt" # Devolvemos los tensores de PyTorch
        )
        return inputs.input_values.squeeze()

def is_white_noise(audio):
    """Comprueba si un audio es ruido blanco según su media y desviación estándar"""
    mean = torch.mean(audio)
    std = torch.std(audio)
    return torch.abs(mean) < 0.001 and std < 0.01 # Definimos las condiciones para considerar ruido blanco

def seed_everything():
    """Fijar semillas para reproducibilidad"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_label_mappings(dataset_path):
    """Construir diccionarios de mapeo de etiquetas"""
    label2id = {}
    id2label = {}
    label_id = 0
    for label_dir in os.listdir(dataset_path):
        if os.path.isdir(os.path.join(dataset_path, label_dir)):
            label2id[label_dir] = label_id
            id2label[label_id] = label_dir
            label_id += 1
    return label2id, id2label

def compute_class_weights(labels):
    """Calcula los pesos de las clases para balancear el dataset"""
    class_counts = Counter(labels) # Contamos las ocurrencias de cada clase
    total_samples = len(labels) # Número total de muestras
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()} # Calculamos los pesos
    return [class_weights[label] for label in labels] # Devolvemos los pesos correspondientes a las etiquetas

def create_dataloader(dataset_path, filter_white_noise, undersample_normal, test_size=0.2, shuffle=True, pin_memory=True):
    """Hacer particiones de entrenamiento y validación"""
    label2id, id2label = build_label_mappings(dataset_path) # Construimos el mapeo de etiquetas
    dataset = AudioDataset(dataset_path, label2id, filter_white_noise, undersample_normal) # Creamos el dataset
    dataset_size = len(dataset) # Obtenemos el tamaño del dataset
    indices = list(range(dataset_size))
    random.shuffle(indices) # Mezclamos los índices para dividir aleatoriamente
    split_idx = int(dataset_size * (1 - test_size)) # Calculamos el índice de partición
    train_indices = indices[:split_idx] # Índices de entrenamiento
    test_indices = indices[split_idx:] # Índices de validación
    train_dataset = Subset(dataset, train_indices) # Subconjunto para entrenamiento
    test_dataset = Subset(dataset, test_indices) # Subconjunto para validación
    labels = [dataset.labels[i] for i in train_indices]
    class_weights = compute_class_weights(labels) # Calculamos los pesos de las clases
    sampler = WeightedRandomSampler(
        weights=class_weights, # Usamos los pesos calculados
        num_samples=len(train_dataset), # El número de muestras es el tamaño del dataset de entrenamiento
        replacement=True # Permitimos la repetición de ejemplos para balanceo
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory
    )
    return train_dataloader, test_dataloader, id2label # Devolvemos los dataloaders y el mapeo de etiquetas

def load_model(model_path, id2label, num_labels):
    """Carga el modelo preentrenado"""
    config = HubertConfig.from_pretrained(
        pretrained_model_name_or_path=model_path, # Cargamos la configuración del modelo
        num_labels=num_labels, # Especificamos el número de etiquetas
        id2label=id2label, # Mapeo de id a etiquetas
        finetuning_task="audio-classification" # Indicamos que el finetuning es para clasificación de audio
    )
    device = torch.device("cuda") # Utilizamos GPU
    model = HubertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_path,
        config=config,
        torch_dtype=torch.float32 # Necesario para evitar errores de precisión
    )
    model.to(device)
    return model

def train_params(dataset_path, filter_white_noise, undersample_normal):
    """Entrenar modelo"""
    train_dataloader, test_dataloader, id2label = create_dataloader(dataset_path, filter_white_noise, undersample_normal)
    model = load_model(MODEL, id2label, num_labels=len(id2label))
    return model, train_dataloader, test_dataloader, id2label

def predict_params(dataset_path, model_path, filter_white_noise, undersample_normal):
    """Predecir en app.py"""
    _, _, id2label = create_dataloader(dataset_path, filter_white_noise, undersample_normal)
    model = load_model(model_path, id2label, num_labels=len(id2label))
    return model, id2label

def compute_metrics(pred):
    """Calcular métricas"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        }

def main(training_args, output_dir, dataset_path, filter_white_noise, undersample_normal):
    """Entrenar modelo"""
    seed_everything()
    model, train_dataloader, test_dataloader, _ = train_params(dataset_path, filter_white_noise, undersample_normal)
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataloader.dataset,
        eval_dataset=test_dataloader.dataset,
    )
    torch.cuda.empty_cache() # Liberamos memoria de la GPU
    trainer.train() # Entrenamos el modelo
    os.makedirs(output_dir, exist_ok=True) # Creamos el directorio de salida si no existe
    trainer.save_model(output_dir) # Guardamos el modelo entrenado en local
    eval_results = trainer.evaluate() # Evaluamos el modelo en el conjunto de validación
    print(f"Evaluation results: {eval_results}") # Imprimimos los resultados de evaluación

def load_config(model_name):
    """Cargar configuración del modelo"""
    with open(config_file, 'r') as f:
        config = json.load(f)  # Cargamos el archivo JSON de configuración
    model_config = config[model_name]  # Obtenemos la configuración específica del modelo
    training_args = TrainingArguments(**model_config["training_args"])  # Creamos los argumentos de entrenamiento
    model_config["training_args"] = training_args  # Actualizamos la configuración con los argumentos de entrenamiento
    return model_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser() # Creamos el parser para argumentos de línea de comandos
    parser.add_argument(
        "--n", choices=["detec", "class"],
        required=True, help="Elegir qué modelo entrenar"
        )
    args = parser.parse_args()
    config = load_config(args.n) # Cargamos la configuración según el modelo seleccionado
    training_args = config["training_args"]
    output_dir = config["output_dir"]
    dataset_path = config["dataset_path"]
    # Ajustamos los parámetros según el modelo elegido
    if args.n == "detec":
        filter_white_noise = False
        undersample_normal = False
    elif args.n == "class":
        filter_white_noise = True
        undersample_normal = True
    # Iniciamos el proceso de entrenamiento
    main(training_args, output_dir, dataset_path, filter_white_noise, undersample_normal)
