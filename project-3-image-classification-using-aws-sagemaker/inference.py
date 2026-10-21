# scripts/inference.py
import io
import json
import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 3

# Same as eval_tf in train_model.py; built once, not per request
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

IMAGE_TYPES = {"image/jpeg", "image/png", "application/x-image"}


def model_fn(model_dir):
    state_dict = torch.load(os.path.join(model_dir, "model.pth"),
                            map_location="cpu", weights_only=True)
    num_classes = state_dict["fc.3.weight"].shape[0]  # read from checkpoint, can't drift from training

    model = models.resnet50(weights=None)  # no download: weights come from model.pth
    model.fc = nn.Sequential(              # identical to net() in train_model.py
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()  # eval() disables Dropout

    # Optional class names saved at training time (falls back to indices if absent)
    classes_path = os.path.join(model_dir, "classes.json")
    model.classes = json.load(open(classes_path)) if os.path.exists(classes_path) else None
    return model


def input_fn(body, content_type):
    if content_type not in IMAGE_TYPES:
        raise ValueError(f"Unsupported content type: {content_type}. Send one of {sorted(IMAGE_TYPES)}")
    image = Image.open(io.BytesIO(body)).convert("RGB")
    return TRANSFORM(image).unsqueeze(0)


def predict_fn(x, model):
    with torch.no_grad():
        probs = torch.softmax(model(x.to(DEVICE)), dim=1)
        top_p, top_i = probs.topk(TOP_K, dim=1)
    return {"probs": top_p[0].tolist(), "indices": top_i[0].tolist(), "classes": model.classes}


def output_fn(pred, accept):
    classes = pred["classes"]
    results = [
        {"index": i, "breed": classes[i] if classes else None, "probability": round(p, 4)}
        for i, p in zip(pred["indices"], pred["probs"])
    ]
    return json.dumps({"predictions": results})
