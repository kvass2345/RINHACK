import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

class SpectrogramDataset(Dataset):
    def init(self, root_dir):
        self.root_dir = root_dir
        self.file_list = os.listdir(root_dir)
        self.transform = transforms.Compose([
            transforms.Resize((360, 480)),
            transforms.ToTensor(),
        ])

    def len(self):
        return len(self.file_list)

    def getitem(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.file_list[idx])
        image = Image.open(img_name)

        label = 0 if 'real' in img_name else 1

        if self.transform:
            image = self.transform(image)

        return image, label

        return image, label

# Определение параметров обучения
num_epochs = 10
batch_size = 32

# Преобразования изображений
transform = transforms.Compose([
    transforms.Resize((360, 480)),  # Изменение размера изображения
    transforms.ToTensor(),  # Преобразование изображения в тензор
])

# Загрузка данных для обучения и тестирования
train_dataset = SpectrogramDataset(root_dir='C:\\Users\\Kerim\\Documents\\rthbv\\Выявление синтезированного голосаречи с использованием методов машинного обучения (ООО «Даталаб»)\\звук\\spect png train', transform=transform)
test_dataset = SpectrogramDataset(root_dir='C:\\Users\\Kerim\\Documents\\rthbv\\Выявление синтезированного голосаречи с использованием методов машинного обучения (ООО «Даталаб»)\\звук\\spect png testing', transform=transform)


# Создание DataLoader для обучения и тестирования
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Создание нейронной сети для классификации спектрограмм
class CNNAudioClassifier(nn.Module):
    def init(self):
        super(CNNAudioClassifier, self).init()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 90 * 120, 256)  # После трех пулинговых слоев размеры изображения уменьшаются в 8 раз по высоте и ширине
        self.fc2 = nn.Linear(256, 2)  # 2 класса: настоящий голос и сгенерированный голос

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 64 * 90 * 120)  # После трех пулинговых слоев размеры изображения уменьшаются в 8 раз по высоте и ширине
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Создаем экземпляр модели после определения всех ее слоев
model = CNNAudioClassifier()

# Проверяем наличие обучаемых параметров
params = list(model.parameters())
if not params:
    print("Модель не содержит обучаемых параметров")

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение нейронной сети
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2%}')

print('Finished Training')