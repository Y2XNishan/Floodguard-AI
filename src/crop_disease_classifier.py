import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


DATA_DIR = Path("data/raw/crop_disease/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train")
MODEL_PATH = Path("models/crop_disease_classifier.pth")
CLASSES_PATH = Path("models/crop_disease_classes.json")

INDIAN_CROPS = [
    "Corn_(maize)",
    "Tomato",
    "Potato",
    "Pepper,_bell",
    "Rice",
    "Grape",
    "Apple",
    "Strawberry",
]


def build_model(num_classes):
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def is_readable_image(path):
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


class FilteredDataset(torch.utils.data.Dataset):
    def __init__(self, samples, label_map, transform):
        self.samples = samples
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), self.label_map[label]


def train_crop_disease_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_t, val_t = get_transforms()

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Crop disease dataset not found: {DATA_DIR}")

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_t)

    indian_indices = []
    for idx, (path, label) in enumerate(full_dataset.samples):
        class_name = full_dataset.classes[label]
        for crop in INDIAN_CROPS:
            if crop in class_name:
                indian_indices.append(idx)
                break

    print(f"Found {len(indian_indices)} Indian crop images")

    class_counts = defaultdict(list)
    for idx in indian_indices:
        label = full_dataset.targets[idx]
        class_counts[label].append(idx)

    limited_indices = []
    skipped = 0
    for label, indices in class_counts.items():
        kept_for_class = 0
        for idx in indices:
            path = full_dataset.samples[idx][0]
            if not is_readable_image(path):
                skipped += 1
                continue
            limited_indices.append(idx)
            kept_for_class += 1
            if kept_for_class >= 300:
                break

    if skipped:
        print(f"Skipped {skipped} missing or unreadable images")

    if not limited_indices:
        raise ValueError("No Indian crop classes found in the dataset.")

    used_labels = sorted(set(full_dataset.targets[i] for i in limited_indices))
    label_map = {old: new for new, old in enumerate(used_labels)}
    class_names = [full_dataset.classes[label] for label in used_labels]

    print(f"Classes: {len(class_names)}")
    for class_name in class_names:
        print(f"  - {class_name}")

    CLASSES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CLASSES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    rng = np.random.default_rng(42)
    shuffled_indices = list(limited_indices)
    rng.shuffle(shuffled_indices)
    train_size = int(0.8 * len(shuffled_indices))
    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:]

    train_samples = [full_dataset.samples[i] for i in train_indices]
    val_samples = [full_dataset.samples[i] for i in val_indices]

    train_set = FilteredDataset(train_samples, label_map, train_t)
    val_set = FilteredDataset(val_samples, label_map, val_t)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)

    model = build_model(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(15):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        acc = 100 * correct / total
        scheduler.step()
        print(f"Epoch {epoch + 1}/15 - Loss: {train_loss:.3f} - Val Acc: {acc:.1f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state": model.state_dict(),
                "num_classes": len(class_names),
                "class_names": class_names,
            }, MODEL_PATH)
            print(f"  Saved: {acc:.1f}%")

    print(f"\nDone! Best accuracy: {best_acc:.1f}%")
    return best_acc


DISEASE_INFO = {
    "healthy": {
        "status": "Healthy",
        "severity": "None",
        "description": "Plant appears healthy with no visible disease symptoms.",
        "treatment": ["Continue regular care", "Monitor regularly", "Maintain good drainage"],
        "flood_connection": "Keep monitoring during flood season as waterlogging weakens plants.",
        "prevention": ["Ensure good drainage", "Avoid overwatering", "Regular inspection"],
    },
    "early_blight": {
        "status": "Diseased",
        "severity": "Moderate",
        "description": "Early Blight caused by Alternaria fungus. Brown spots with concentric rings on leaves.",
        "treatment": [
            "Remove affected leaves immediately",
            "Apply copper-based fungicide",
            "Improve air circulation",
            "Avoid wetting leaves",
        ],
        "flood_connection": "Flooding creates excess moisture that significantly accelerates Early Blight spread.",
        "prevention": ["Crop rotation every season", "Avoid overhead irrigation", "Space plants properly"],
    },
    "late_blight": {
        "status": "Diseased",
        "severity": "High",
        "description": "Late Blight caused by Phytophthora. Dark water-soaked lesions on leaves and stems.",
        "treatment": [
            "Apply systemic fungicide immediately",
            "Remove and destroy infected plants",
            "Do not compost infected material",
            "Contact agriculture department",
        ],
        "flood_connection": "Flood conditions are ideal for Late Blight and can destroy an entire crop within days.",
        "prevention": ["Use resistant varieties", "Ensure field drainage", "Early fungicide application"],
    },
    "brown_spot": {
        "status": "Diseased",
        "severity": "Moderate",
        "description": "Brown Spot caused by Helminthosporium. Oval brown spots on leaves.",
        "treatment": [
            "Apply Mancozeb fungicide",
            "Improve soil nutrition",
            "Ensure proper drainage",
            "Remove affected leaves",
        ],
        "flood_connection": "Waterlogged soil from flooding weakens plant immunity, increasing Brown Spot risk by 3x.",
        "prevention": ["Balanced fertilization", "Avoid water stress", "Use certified seeds"],
    },
    "bacterial_spot": {
        "status": "Diseased",
        "severity": "High",
        "description": "Bacterial Spot spreads rapidly in wet conditions. Water-soaked dark spots on leaves.",
        "treatment": [
            "Apply copper bactericide",
            "Remove infected plant parts",
            "Avoid working in wet fields",
            "Disinfect tools regularly",
        ],
        "flood_connection": "Flood water actively spreads bacterial infections between plants across entire fields.",
        "prevention": ["Use disease-free seeds", "Avoid overhead watering", "Proper field sanitation"],
    },
    "common_rust": {
        "status": "Diseased",
        "severity": "Moderate",
        "description": "Common Rust caused by Puccinia. Orange/brown pustules on both leaf surfaces.",
        "treatment": [
            "Apply fungicide at first sign",
            "Use triazole fungicides",
            "Remove severely infected leaves",
        ],
        "flood_connection": "High humidity from flooding accelerates rust spore germination and spread.",
        "prevention": ["Plant resistant varieties", "Early season planting", "Monitor regularly"],
    },
    "default": {
        "status": "Diseased",
        "severity": "Moderate",
        "description": "Disease detected on plant leaf. Consult local agriculture expert.",
        "treatment": [
            "Consult agriculture department",
            "Apply appropriate pesticide",
            "Remove affected parts",
            "Improve drainage",
        ],
        "flood_connection": "Flooding increases disease risk. Monitor closely after flood events.",
        "prevention": ["Regular monitoring", "Good field hygiene", "Proper drainage"],
    },
}


def get_disease_flood_connection():
    return {
        "Early_blight": "High moisture from flooding significantly increases Early Blight spread.",
        "Late_blight": "Flooding creates ideal wet conditions for Late Blight.",
        "Brown_spot": "Waterlogged soil weakens plants, making them susceptible.",
        "Bacterial_spot": "Flood water spreads bacterial infections between plants.",
        "healthy": "Healthy plant - monitor during flood season.",
    }


def classify_crop_image(image):
    device = torch.device("cpu")

    if not MODEL_PATH.exists():
        return {
            "crop": "Unknown",
            "disease": "Model not trained",
            "status": "Error",
            "confidence": 0,
            "severity": "N/A",
            "description": "Please train the model first.",
            "treatment": ["Run training script"],
            "flood_connection": "N/A",
            "prevention": [],
        }

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]

    model = build_model(num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img_tensor = transform(image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    predicted_idx = probs.argmax().item()
    confidence = float(probs.max()) * 100

    if confidence < 80.0:
        return {
            "crop": "Unknown",
            "disease": "Crop not supported",
            "status": "Error",
            "confidence": confidence,
            "severity": "N/A",
            "description": "This crop is not supported. Supported crops: Corn/Maize, Tomato, Potato, Pepper, Rice, Grape, Apple, Strawberry.",
            "treatment": ["Please upload a leaf from a supported crop"],
            "flood_connection": "N/A",
            "prevention": [],
        }

    predicted_class = class_names[predicted_idx]

    parts = predicted_class.split("___")
    crop = parts[0].replace("_", " ").replace("(maize)", "").replace(",", "").strip()
    disease_raw = parts[1] if len(parts) > 1 else "Unknown"
    disease = disease_raw.replace("_", " ").strip()

    disease_key = disease_raw.lower()
    info = None
    for key in DISEASE_INFO:
        if key in disease_key:
            info = DISEASE_INFO[key]
            break
    if not info:
        if "healthy" in disease_key:
            info = DISEASE_INFO["healthy"]
        else:
            info = DISEASE_INFO["default"]

    return {
        "crop": crop.title(),
        "disease": disease.title(),
        "status": info["status"],
        "confidence": round(confidence, 1),
        "severity": info["severity"],
        "description": info["description"],
        "treatment": info["treatment"],
        "flood_connection": info["flood_connection"],
        "prevention": info["prevention"],
    }


if __name__ == "__main__":
    train_crop_disease_model()
