import torch
from torchvision import models, transforms
from PIL import Image
import json
import os
import urllib.request

# Download the ImageNet class index if it doesn't exist
LABELS_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
LABELS_FILENAME = "imagenet_class_index.json"

if not os.path.exists(LABELS_FILENAME):
    print("Downloading ImageNet labels...")
    urllib.request.urlretrieve(LABELS_URL, LABELS_FILENAME)

with open(LABELS_FILENAME) as f:
    imagenet_classes = json.load(f)

# Load the pretrained ResNet18 model
model = models.resnet18(pretrained=True)
model.eval()

# Define the preprocessing transformation
preprocess = transforms.Compose([
    transforms.Resize(256),               # Resize the image to 256 pixels on the shorter side
    transforms.CenterCrop(224),           # Crop to 224x224 pixels as expected by ResNet18
    transforms.ToTensor(),                # Convert image to PyTorch tensor
    transforms.Normalize(                 # Normalize using ImageNet mean and std
         mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225])
])

def predict(image_path, topk=5):
    image = Image.open(image_path).convert("RGB")
    
    # Preprocess the image and add a batch dimension
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    
    # Perform inference
    with torch.no_grad():
        output = model(input_batch)
    
    # Convert the output logits to probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Get the top-k predictions
    top_probs, top_indices = torch.topk(probabilities, topk)
    
    # Map indices to class labels
    results = []
    for i in range(topk):
        idx = str(top_indices[i].item())
        label = imagenet_classes[idx][1]
        prob = top_probs[i].item()
        results.append((label, prob))
    return results

if __name__ == "__main__":
    image_paths = ["/Users/Yoni Ben-Zvi/VSCodeProjects/CV3/capsicum.jpeg",
                   "/Users/Yoni Ben-Zvi/VSCodeProjects/CV3/dog.jpeg",
                   "/Users/Yoni Ben-Zvi/VSCodeProjects/CV3/tiger-shark.jpg"]
    for img_path in image_paths:
        if os.path.exists(img_path):
            print(f"\nInference results for {img_path}:")
            predictions = predict(img_path)
            for label, prob in predictions:
                print(f"{label}: {prob:.4f}")
        else:
            print(f"\nFile {img_path} not found. Please check the image path.")