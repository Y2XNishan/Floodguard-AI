from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torch.utils.data import DataLoader, WeightedRandomSampler


CLASSES = ["mild", "moderate", "no_flood", "severe"]
MODEL_PATH = Path("models/flood_classifier.pth")
DATA_DIR = Path("data/processed/flood_classified")
EPOCHS = 20
RESUME_START_EPOCH = 8
RESUME_CLASS_COUNTS = {
    "mild": 165,
    "moderate": 534,
    "severe": 374,
    "no_flood": 3355,
}


def build_model(num_classes=len(CLASSES), pretrained=True):
    if pretrained:
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = efficientnet_b0(weights=weights)
    else:
        model = efficientnet_b0(weights=None)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )

    # Fine-tune the classification head and the last EfficientNet blocks.
    for param in list(model.parameters())[-30:]:
        param.requires_grad = True

    return model


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.3,
            hue=0.1,
        ),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def get_class_counts(dataset):
    counts = torch.zeros(len(dataset.classes), dtype=torch.float)
    for _, label in dataset.samples:
        counts[label] += 1
    return counts


def build_weighted_sampler(train_data):
    counts = get_class_counts(train_data)
    class_weights = 1.0 / counts.clamp(min=1)
    sample_weights = [class_weights[label] for _, label in train_data.samples]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def train_classifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_t, val_t = get_transforms()

    train_data = datasets.ImageFolder(DATA_DIR / "train", transform=train_t)
    val_data = datasets.ImageFolder(DATA_DIR / "val", transform=val_t)

    sampler = build_weighted_sampler(train_data)
    train_loader = DataLoader(
        train_data,
        batch_size=32,
        sampler=sampler,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    model = build_model(num_classes=len(train_data.classes), pretrained=True).to(device)

    class_counts = get_class_counts(train_data)
    class_weights = 1.0 / class_counts.clamp(min=1)
    class_weights = class_weights / class_weights.sum()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6,
    )

    print(f"Classes: {train_data.classes}")
    print(
        "Class counts: "
        + ", ".join(
            f"{name}={int(class_counts[idx].item())}"
            for idx, name in enumerate(train_data.classes)
        )
    )

    best_acc = 0
    for epoch in range(EPOCHS):
        if epoch == 10:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=EPOCHS - epoch,
                eta_min=1e-6,
            )
            print("Unfroze all EfficientNetB0 layers for fine-tuning.")

        # Training
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        acc = 100 * correct / total
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch + 1}/{EPOCHS} - "
            f"Loss: {train_loss:.3f} - "
            f"Val Acc: {acc:.1f}% - "
            f"LR: {current_lr:.6f}"
        )

        if acc > best_acc:
            best_acc = acc
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  Best model saved: {acc:.1f}%")

    print(f"\nTraining complete! Best accuracy: {best_acc:.1f}%")
    return best_acc


def evaluate_classifier(model, val_loader, class_names, device):
    model.eval()
    correct = total = 0
    per_class = {
        class_name: {"correct": 0, "total": 0, "accuracy": 0.0}
        for class_name in class_names
    }

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for label, pred in zip(labels.cpu().tolist(), predicted.cpu().tolist()):
                class_name = class_names[label]
                per_class[class_name]["total"] += 1
                if label == pred:
                    per_class[class_name]["correct"] += 1

    for stats in per_class.values():
        if stats["total"]:
            stats["accuracy"] = 100 * stats["correct"] / stats["total"]

    overall_acc = 100 * correct / total if total else 0.0
    return overall_acc, per_class


def format_per_class_accuracy(per_class):
    return ", ".join(
        f"{class_name}={stats['accuracy']:.1f}% "
        f"({stats['correct']}/{stats['total']})"
        for class_name, stats in per_class.items()
    )


def resume_training(start_epoch=RESUME_START_EPOCH, end_epoch=EPOCHS, checkpoint_epoch=None):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Resuming training on: {device}")

    train_t, val_t = get_transforms()
    train_data = datasets.ImageFolder(DATA_DIR / "train", transform=train_t)
    val_data = datasets.ImageFolder(DATA_DIR / "val", transform=val_t)

    sampler = build_weighted_sampler(train_data)
    train_loader = DataLoader(
        train_data,
        batch_size=32,
        sampler=sampler,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    model = build_model(num_classes=len(train_data.classes), pretrained=False).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    class_counts = torch.tensor(
        [RESUME_CLASS_COUNTS[class_name] for class_name in train_data.classes],
        dtype=torch.float,
    )
    class_weights = 1.0 / class_counts.clamp(min=1)
    class_weights = class_weights / class_weights.sum()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    if start_epoch >= 10:
        for param in model.parameters():
            param.requires_grad = True
        trainable_params = model.parameters()
        initial_lr = 0.0001
    else:
        trainable_params = [param for param in model.parameters() if param.requires_grad]
        initial_lr = 0.0005

    optimizer = torch.optim.Adam(trainable_params, lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6,
    )

    print(f"Classes: {train_data.classes}")
    print(
        "Resume class counts: "
        + ", ".join(
            f"{name}={int(class_counts[idx].item())}"
            for idx, name in enumerate(train_data.classes)
        )
    )

    best_acc, best_per_class = evaluate_classifier(
        model,
        val_loader,
        train_data.classes,
        device,
    )
    best_epoch = checkpoint_epoch if checkpoint_epoch is not None else start_epoch - 1
    print(
        f"Loaded checkpoint baseline - "
        f"Val Acc: {best_acc:.1f}% - "
        f"Per-class: {format_per_class_accuracy(best_per_class)}"
    )

    for epoch in range(start_epoch, end_epoch + 1):
        if epoch == 10:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=EPOCHS,
                eta_min=1e-6,
            )
            print("Unfroze all EfficientNetB0 layers for fine-tuning.")

        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        acc, per_class = evaluate_classifier(
            model,
            val_loader,
            train_data.classes,
            device,
        )
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch}/{EPOCHS} - "
            f"Loss: {train_loss:.3f} - "
            f"Val Acc: {acc:.1f}% - "
            f"LR: {current_lr:.6f} - "
            f"Per-class: {format_per_class_accuracy(per_class)}"
        )

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            best_per_class = per_class
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  Best checkpoint saved: {acc:.1f}% at epoch {epoch}")

    print(
        f"\nResume training complete! "
        f"Best val acc: {best_acc:.1f}% at epoch {best_epoch}"
    )
    print(f"Best per-class breakdown: {format_per_class_accuracy(best_per_class)}")
    return best_acc


def classify_flood_image(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not MODEL_PATH.exists():
        return rule_based_classify(image)

    model = build_model(num_classes=len(CLASSES), pretrained=False).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    probs_dict = {
        CLASSES[i].replace("_", " ").title(): float(probs[i]) * 100
        for i in range(len(CLASSES))
    }
    predicted_class = CLASSES[probs.argmax().item()]
    confidence = float(probs.max()) * 100

    if predicted_class == "no_flood":
        return {
            "severity": "No Flooding",
            "confidence": confidence,
            "probabilities": probs_dict,
            "description": "No flood detected in this image.",
            "recommendations": [
                "Area appears safe",
                "Continue monitoring weather",
            ],
        }

    descriptions = {
        "mild": "Minor flooding detected. Water levels are low. Roads may be waterlogged.",
        "moderate": "Moderate flooding detected. Significant water accumulation visible. Take precautions.",
        "severe": "SEVERE flooding detected! Extensive water coverage. Immediate action required!",
    }

    recommendations = {
        "mild": [
            "Monitor water levels closely",
            "Avoid low-lying areas",
            "Keep emergency kit ready",
            "Stay updated with local alerts",
        ],
        "moderate": [
            "Move valuables to higher floors",
            "Avoid unnecessary travel",
            "Contact local authorities",
            "Prepare for possible evacuation",
        ],
        "severe": [
            "EVACUATE IMMEDIATELY",
            "Call 112 or NDMA: 1078",
            "Move to nearest relief camp",
            "Do NOT enter floodwater",
            "Alert neighbors",
        ],
    }

    return {
        "severity": predicted_class.title(),
        "confidence": confidence,
        "probabilities": probs_dict,
        "description": descriptions[predicted_class],
        "recommendations": recommendations[predicted_class],
    }


def rule_based_classify(image):
    import numpy as np

    img_array = np.array(image.convert("RGB"))

    # PIL arrays are RGB.
    red = img_array[:, :, 0].mean()
    green = img_array[:, :, 1].mean()
    blue = img_array[:, :, 2].mean()
    brightness = img_array.mean()

    if blue > red + 20 and brightness < 100:
        severity = "severe"
    elif blue > red + 10 or brightness < 130:
        severity = "moderate"
    elif brightness > 150 and abs(red - blue) < 35:
        severity = "no_flood"
    else:
        severity = "mild"

    descriptions = {
        "mild": "Minor flooding indicators detected.",
        "moderate": "Moderate flooding indicators detected.",
        "severe": "Severe flooding indicators detected!",
        "no_flood": "No flood detected in this image.",
    }

    if severity == "no_flood":
        return {
            "severity": "No Flooding",
            "confidence": 65.0,
            "probabilities": {
                "Mild": 11.7,
                "Moderate": 11.7,
                "Severe": 11.6,
                "No Flood": 65.0,
            },
            "description": descriptions[severity],
            "recommendations": [
                "Area appears safe",
                "Continue monitoring weather",
            ],
        }

    return {
        "severity": severity.title(),
        "confidence": 65.0,
        "probabilities": {
            "Mild": 33.3,
            "Moderate": 33.3,
            "Severe": 33.3,
            "No Flood": 0.1,
        },
        "description": descriptions[severity],
        "recommendations": [
            "Contact local authorities",
            "Call 112 for emergency",
        ],
    }


if __name__ == "__main__":
    train_classifier()
