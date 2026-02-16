
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import torch.nn.functional as F
import os

# Import the model architecture from model.py
# (Assuming model.py is in the same directory or accessible via PYTHONPATH)
from model import CropStressResNet50

def preprocess_image(image_path):
    # Define the same transformations used during validation
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return val_transform(image).unsqueeze(0) # Add batch dimension

def load_model(model_path, num_classes, device):
    model = CropStressResNet50(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def load_class_names(class_names_path):
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    return class_names

def predict_crop_stress(image_path, model_path='crop_stress_model.pth', class_names_path='class_names.json'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load class names first to determine num_classes
    class_names = load_class_names(class_names_path)
    num_classes = len(class_names)
    
    model = load_model(model_path, num_classes, device)
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    predicted_class_name = class_names[predicted_idx.item()]
    confidence_score = confidence.item()

    structured_output = {
        "stress_type": None,
        "disease_name": None,
        "stress_cause": None,
        "confidence": confidence_score
    }

    if predicted_class_name == 'Healthy':
        structured_output["stress_type"] = "Healthy"
    elif predicted_class_name.endswith('_Disease'):
        structured_output["stress_type"] = "Biotic"
        structured_output["disease_name"] = predicted_class_name
    elif predicted_class_name == 'Heat_Stress': # Assuming Drought is also part of Abiotic if present
        structured_output["stress_type"] = "Abiotic"
        structured_output["stress_cause"] = predicted_class_name
    # Add more conditions for other stress types if applicable

    return structured_output

if __name__ == '__main__':
    # Example usage - ensure an image path exists for testing
    # You might need to provide a real image path here, or create a dummy image
    # For demonstration, let's try to find an image in the processed dataset
    example_image_path = None
    dataset_root = 'dataset_processed'
    train_dir = os.path.join(dataset_root, 'train')
    val_dir = os.path.join(dataset_root, 'val')

    # Try to find an image in the val directory for a class that exists
    if os.path.exists(val_dir) and os.listdir(val_dir):
        first_class_dir = os.path.join(val_dir, os.listdir(val_dir)[0])
        if os.path.exists(first_class_dir) and os.listdir(first_class_dir):
            example_image_path = os.path.join(first_class_dir, os.listdir(first_class_dir)[0])

    if example_image_path and os.path.exists(example_image_path):
        print(f"Performing inference on example image: {example_image_path}")
        result = predict_crop_stress(example_image_path)
        print("Inference Result:")
        print(json.dumps(result, indent=4))
    else:
        print("No example image found for inference. Please provide a valid image path to test the function.")
