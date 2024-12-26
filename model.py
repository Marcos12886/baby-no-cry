import os
import json
import random
import argparse
import torch
import torchaudio
from dotenv import load_dotenv
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    Wav2Vec2FeatureExtractor, HubertConfig, HubertForSequenceClassification,
    Trainer, TrainingArguments
    )

load_dotenv()
MODEL = "ntu-spml/distilhubert"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL)
seed = 123
max_duration = 1.00
sampling_rate = feature_extractor.sampling_rate
config_file = "config.json"
batch_size = 1024
num_workers = 12

class AudioDataset(Dataset):
    """Dataset de audios"""
    def __init__(self, dataset_path, label2id, undersample_normal):
        """Inicializar dataset"""
        self.dataset_path = dataset_path
        self.label2id = label2id
        self.file_paths = []
        self.labels = []
        for label_dir, label_id in self.label2id.items():
            label_path = os.path.join(self.dataset_path, label_dir)
            if os.path.isdir(label_path):
                for file_name in os.listdir(label_path):
                    audio_path = os.path.join(label_path, file_name)
                    self.file_paths.append(audio_path)
                    self.labels.append(label_id)
        if undersample_normal and self.label2id:
            self.undersample_normal_class()

    def undersample_normal_class(self):
        """Método para submuestrear la clase normal y equilibrar el dataset"""
        normal_label = self.label2id.get('1s_normal')
        label_counts = {}
        for label in self.labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        other_counts = [count for label, count in label_counts.items() if label != normal_label]
        if other_counts:
            target_count = max(other_counts)
            normal_indices = [i for i, label in enumerate(self.labels) if label == normal_label]
            keep_indices = random.sample(normal_indices, target_count)
            new_file_paths = []
            new_labels = []
            for i, (path, label) in enumerate(zip(self.file_paths, self.labels)):
                if label != normal_label or i in keep_indices:
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
        input_values = self.preprocess_audio(audio_path)
        return {
            "input_values": input_values,
            "labels": torch.tensor(label)
        }

    def preprocess_audio(self, audio_path):
        """Preprocesamiento de los archivos de audio"""
        waveform, sample_rate = torchaudio.load(
            audio_path,
            normalize=True
            )
        if sample_rate != sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, sampling_rate)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-6)
        max_length = int(sampling_rate * max_duration)
        if waveform.shape[1] > max_length:
            waveform = waveform[:, :max_length]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, max_length - waveform.shape[1]))
        inputs = feature_extractor(
            waveform.squeeze(),
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )
        return inputs.input_values.squeeze()

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
    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    class_weights = {cls: len(labels) / count for cls, count in class_counts.items()}
    return [class_weights[label] for label in labels]

def create_dataloader(dataset_path, undersample_normal, test_size=0.2, shuffle=True, pin_memory=True):
    """Hacer particiones de entrenamiento y validación"""
    label2id, id2label = build_label_mappings(dataset_path)
    dataset = AudioDataset(dataset_path, label2id, undersample_normal)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    split_idx = int(dataset_size * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    labels = [dataset.labels[i] for i in train_indices]
    class_weights = compute_class_weights(labels)
    sampler = WeightedRandomSampler(
        weights=class_weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory
    )
    return train_dataloader, test_dataloader, id2label

def load_model(MODEL, id2label):
    """Carga el modelo preentrenado"""
    config = HubertConfig.from_pretrained(
        pretrained_model_name_or_path=MODEL,
        num_labels=len(id2label),
        id2label=id2label,
        finetuning_task="audio-classification"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HubertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=MODEL,
        config=config,
        torch_dtype=torch.float32
    )

    model.to(device)
    return model

def train_params(dataset_path, model_type):
    """Entrenar modelo"""
    undersample_normal = (model_type == "class")
    train_dataloader, test_dataloader, id2label = create_dataloader(dataset_path, undersample_normal)
    model = load_model(MODEL, id2label)
    return model, train_dataloader, test_dataloader, id2label

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

def main(training_args, output_dir, dataset_path, model_type):
    """Entrenar modelo"""
    seed_everything()
    model, train_dataloader, test_dataloader, _ = train_params(dataset_path, model_type)
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataloader.dataset,
        eval_dataset=test_dataloader.dataset,
    )
    torch.cuda.empty_cache()
    trainer.train()
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

def load_config(model_name):
    """Cargar configuración del modelo"""
    with open(config_file, 'r') as f:
        config = json.load(f)
    model_config = config[model_name]
    training_args = TrainingArguments(**model_config["training_args"])
    model_config["training_args"] = training_args
    return model_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n", choices=["detec", "class"],
        required=True, help="Elegir qué modelo entrenar"
        )
    args = parser.parse_args()
    config = load_config(args.n)
    training_args = config["training_args"]
    output_dir = config["output_dir"]
    dataset_path = config["dataset_path"]
    if args.n == "detec":
        undersample_normal = False
    elif args.n == "class":
        undersample_normal = True
    main(training_args, output_dir, dataset_path, args.n)
