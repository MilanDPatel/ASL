import cv2
import numpy as np
import torch
from torch import nn
from cvzone.HandTrackingModule import HandDetector
import math
import torchvision.transforms as transforms

# Create the same transform as used in training
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class SimpleModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, 
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) 
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7, 
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x

# Initialize camera
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleModel(3, 10, 26)
model.load_state_dict(torch.load('Model/ASLModels.pth', map_location=device))
model.to(device)
model.eval()

imgSize = 300
offset = 20
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Create debug window
debug_window = np.ones((300, 900, 3), dtype=np.uint8) * 255

while True:
    try:
        success, img = cap.read()
        if not success:
            print("Failed to read from camera")
            break
            
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            
            y_min = max(y - offset, 0)
            y_max = min(y + h + offset, img.shape[0])
            x_min = max(x - offset, 0)
            x_max = min(x + w + offset, img.shape[1])
            
            imgCrop = img[y_min:y_max, x_min:x_max]
            
            if imgCrop.size != 0:
                aspectRatio = h / w
                
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    if wGap + wCal <= imgSize:
                        imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    if hGap + hCal <= imgSize:
                        imgWhite[hGap:hCal + hGap, :] = imgResize
                
                # Resize to 28x28 and apply the same transforms as training
                imgWhite = cv2.resize(imgWhite, (28, 28))
                
                # Convert to PIL Image for transforms
                imgWhite_tensor = transform(imgWhite).unsqueeze(0)
                imgWhite_tensor = imgWhite_tensor.to(device)

                try:
                    with torch.no_grad():
                        predictions = model(imgWhite_tensor)
                    
                    probabilities = torch.nn.functional.softmax(predictions, dim=1)
                    confidence, index = torch.max(probabilities, 1)
                    confidence = confidence.item() * 100
                    
                    # Draw prediction on output image
                    cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                                (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
                    cv2.putText(imgOutput, labels[index.item()], (x, y -26), 
                              cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                    cv2.rectangle(imgOutput, (x-offset, y-offset),
                                (x + w+offset, y + h+offset), (255, 0, 255), 4)
                    
                    # Update debug window
                    debug_window.fill(255)
                    
                    # Draw probability bars
                    bar_height = 50
                    for i, prob in enumerate(probabilities[0]):
                        prob_value = prob.item() * 100
                        bar_width = int(prob_value * 3)  # Scale bar width
                        cv2.rectangle(debug_window, 
                                    (50, 50 + i * 60),
                                    (50 + bar_width, 50 + i * 60 + bar_height),
                                    (0, 255, 0), -1)
                        cv2.putText(debug_window, 
                                  f"{labels[i]}: {prob_value:.1f}%",
                                  (50, 40 + i * 60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    
                    # Show processed images
                    cv2.imshow("Debug Window", debug_window)
                    cv2.imshow("Processed Input", imgWhite)
                    
                except Exception as e:
                    print(f"Prediction error: {str(e)}")
        
        cv2.imshow("Image", imgOutput)
        key = cv2.waitKey(1)
        
        if key == ord('q'):
            break
            
    except Exception as e:
        print(f"Error in main loop: {str(e)}")
        continue

cap.release()
cv2.destroyAllWindows()