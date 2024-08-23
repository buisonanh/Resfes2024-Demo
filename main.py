import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from model import ResNet, BasicBlock




#checkpoint_path = 'model/resnet18-epoch109fer2013.pth'
checkpoint_path = 'model/resnet18-epoch140ferplus.pth'

# Instantiate the model
num_classes = 7  # Based on FER datasets
model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Get the model state dict
model_dict = model.state_dict()

# 1. Filter out unnecessary keys
pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}

# 2. Overwrite entries in the existing state dict
model_dict.update(pretrained_dict) 

# 3. Load the new state dict into the model
model.load_state_dict(model_dict)

# Set the model to evaluation mode
model.eval()



target_size = 48
mean = 0
std = 255

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((target_size, target_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])



# Define FER2013 labels
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Extract the face region
        face = frame[y:y+h, x:x+w]
        
        # Preprocess the face
        face_tensor = transform(face)
        face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            outputs, _ = model(face_tensor)
            output = outputs[0]  # Select the output from the last fully connected layer
            _, predicted = torch.max(output, 1)
            label = labels[predicted.item()]
        
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Write the label above the bounding box
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Facial Expression Recognition', frame)
    
    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()



