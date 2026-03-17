# image-classifier
🐱 CIFAR-10 Image Classifier built with PyTorch + ResNet50 Transfer Learning. Upload an image to classify it into 10 categories: airplane, car, bird, cat, deer, dog, frog, horse, ship or truck. 91% test accuracy!
# 🐱 CIFAR-10 Image Classifier

A deep learning web app that classifies images into 10 categories.

## 🚀 Live Demo
[Click here to try it!]([https://huggingface.co/spaces/srutha4/cifar10-image-classifier])

## 📊 Model Performance
- Dataset: CIFAR-10 (60,000 images, 10 classes)
- Model: ResNet50 + Transfer Learning + Fine Tuning
- Test Accuracy: 91.03%

## 🛠️ Tech Stack
- Python
- PyTorch
- torchvision
- ResNet50 (pretrained)
- Gradio
- HuggingFace Spaces

## 💡 How It Works
1. Image uploaded and resized to 224x224
2. ResNet50 extracts 2048 features
3. Final layer classifies into 10 classes
4. Top 3 predictions shown with confidence scores

## 📁 Classes
airplane, car, bird, cat, deer, dog, frog, horse, ship, truck

## 🧠 Key Learnings
- Transfer Learning vs training from scratch
- Fine tuning frozen layers
- Overfitting and how to prevent it
