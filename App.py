from flask import Flask, request, jsonify, render_template
import torch
import torchvision.transforms as transforms
import torch.nn as nn 
import torch.nn.functional as F 
from PIL import Image
import io

app = Flask(__name__)

# Load the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
net.load_state_dict(torch.load('cifar_net.pth'))
net.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = net(tensor)
    _, predicted = torch.max(outputs.data, 1)
    return classes[predicted]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "Please upload a file", 400
        file = request.files['file']
        if file.filename == '':
            return "No file selected", 400
        if file:
            img_bytes = file.read()
            class_name = get_prediction(image_bytes=img_bytes)
            return render_template('index.html', prediction=class_name)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
