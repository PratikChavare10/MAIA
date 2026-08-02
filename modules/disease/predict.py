
import os
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
from config import DISEASE_MODEL_PATH
# 1. Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Load the saved checkpoint
checkpoint_path = DISEASE_MODEL_PATH   # Update if your path is different
print(f"Checkpoint path: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)

class_names = checkpoint['classes']
print(f"Loaded {len(class_names)} classes successfully.")

# 3. Re-initialize the model architecture and load weights
weights = models.ViT_B_16_Weights.DEFAULT
model = models.vit_b_16(weights=weights)

# Replace the head to match your training configuration
model.heads.head = torch.nn.Linear(model.heads.head.in_features, len(class_names))

# Load the trained weights
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()  # Set model to evaluation mode

# 4. Define the inference transforms (must match validation preprocessing)
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 5. Function to predict a single image
def predict_disease(image_path):
    print("Hello")
    abs_image_path = os.path.abspath(image_path)
    print(f"Loading image from: {abs_image_path}")

    # 1. Path चेक करा
    if not os.path.exists(abs_image_path):
        raise FileNotFoundError(f"Image not found at {abs_image_path}")

    # 2. Safe PIL Reading (Lazy Loading टाळण्यासाठी image.load() वापरा)
    with Image.open(abs_image_path) as img:
        img.load()  # Image मेमरीमध्ये फोर्सफुली लोड करा
        image = img.convert('RGB')

    print("Image loaded successfully!")  # <-- आता हे १००% प्रिंट होईल!

    # 3. Transform & Inference
    input_tensor = inference_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_label = class_names[predicted_idx.item()]
    confidence_score = confidence.item() * 100

    print(f"\n--- Prediction Results ---")
    print(f"Predicted Class: {predicted_label}")
    print(f"Confidence: {confidence_score:.2f}%")

    return {"disease": predicted_label, "confidence": confidence_score}

# --- Example Usage ---
# Replace with the path to a real test image on your machine or Colab
# test_image_path = 'image2.jpg'
#
# print(predict_disease(test_image_path))

