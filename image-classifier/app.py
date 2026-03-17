import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Class names
classes = ['airplane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Load model
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load('resnet_cifar10.pth', 
                          map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Transform
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Prediction function
def predict(image):
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    # Return top 3 predictions
    top3_prob, top3_idx = torch.topk(probabilities, 3)
    result = {}
    for i in range(3):
        result[classes[top3_idx[i]]] = float(top3_prob[i])
    return result

# Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="🐱 CIFAR-10 Image Classifier",
    description="Upload an image and I'll predict what it is! (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck)"
)

demo.launch()