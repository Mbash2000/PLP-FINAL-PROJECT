import torch
import torchvision.models as models
import torch.nn as nn

# Number of classes
num_classes = 39

# Load ResNet50 model
resnet50 = models.resnet50(pretrained=False)

# Replace the final fully connected layer to match number of classes
resnet50.fc = nn.Linear(resnet50.fc.in_features, num_classes)

# Load your trained model weights
state_dict = torch.load("trained_model.pth", map_location=torch.device('cpu'))  # Adjust device if needed
resnet50.load_state_dict(state_dict)

# Set model to evaluation mode
resnet50.eval()

# Test with dummy input
dummy_input = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 channels, 224x224 image
output = resnet50(dummy_input)

print("Output shape:", output.shape)
print("Output:", output)
