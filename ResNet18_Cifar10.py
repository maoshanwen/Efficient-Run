import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from google.colab import drive
drive.mount('/content/drive')

# 配置参数
config = {
    "drive_path": "/content/drive/My Drive",
    "dataset_path": "data/cifar10",
    "num_classes": 10,
    "learning_rate": 0.01,
    "num_epochs": 100,
    "batch_size": 128,
    "test_batch_size": 100,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "save_model_threshold": 90.0,  # 90%准确率保存模型
}

# 切换到Google Drive的目录
os.chdir(config["drive_path"])

# 定义基本残差块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

# 定义ResNet18模型
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 实例化ResNet18
def Resnet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

model = Resnet18(config["num_classes"]).to(config["device"])
print(model)

# 设置优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
criterion = nn.CrossEntropyLoss().to(config["device"])

# 定义数据转换，包含数据增强
transforms = {
    "train": Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ]),
    "test": Compose([
        ToTensor(),
        Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ]),
}

# 数据加载
train_dataset = CIFAR10(root=config["dataset_path"], train=True, download=True, transform=transforms["train"])
train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=True)

test_dataset = CIFAR10(root=config["dataset_path"], train=False, download=True, transform=transforms["test"])
test_dataloader = DataLoader(test_dataset, batch_size=config["test_batch_size"], shuffle=False, num_workers=0, pin_memory=True)

def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device, verbose=True) -> tuple:
    model.eval()
    num_samples = 0
    num_correct = 0
    total_loss = 0.0
    all_targets = []
    all_preds = []
    for inputs, targets in tqdm(dataloader, desc="eval", leave=False, disable=not verbose):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        num_samples += targets.size(0)
        num_correct += (preds == targets).sum().item()
        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = (num_correct / num_samples) * 100
    return accuracy, avg_loss, all_targets, all_preds

def train(model, train_loader, test_loader, config, optimizer, criterion):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    lrs = []
    best_accuracy = 0.0

    for epoch in range(config["num_epochs"]):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            inputs, targets = inputs.to(config["device"]), targets.to(config["device"])

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100. * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        lrs.append(optimizer.param_groups[0]['lr'])

        print(f"Epoch [{epoch+1}] Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # 在每个epoch结束后进行一次评估
        test_accuracy, test_loss, all_targets, all_preds = evaluate(model, test_loader, criterion, config["device"])
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch [{epoch+1}] Test Loss: {test_loss:.4f},Test Accuracy: {test_accuracy:.2f}%")

        # 如果测试集准确率高于当前最高准确率，保存模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), f'best_resnet18_epoch{epoch+1}_acc{test_accuracy:.2f}.pth')
            print(f'Best model saved at epoch {epoch+1} with accuracy {test_accuracy:.2f}%')

    # 绘制训练和验证损失
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')

    # 绘制训练和验证准确率
    plt.subplot(2, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Over Epochs')

       # 绘制学习率变化
    plt.subplot(2, 2, 3)
    plt.plot(lrs, label='Learning Rate')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Over Epochs')

    # 绘制混淆矩阵
    plt.subplot(2, 2, 4)
    cm = confusion_matrix(all_targets, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(config["num_classes"])])
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
    plt.title('Confusion Matrix')

    plt.tight_layout()
    plt.show()

# 开始训练模型
train(model, train_dataloader, test_dataloader, config, optimizer, criterion)

print("Training completed.")
