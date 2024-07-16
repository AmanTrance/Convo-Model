import torch.nn 
from torchvision import transforms as tf
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import pathlib

root = pathlib.Path('Convo_Model/public')
transforms = v2.Compose([
    tf.Resize((224,224)),
    tf.ToTensor()
])
train_directory = str(root) + r'\train'
test_directory = str(root) + r'\test'

torch.manual_seed(42)
train_data = ImageFolder(root = train_directory,
                         transform = transforms,
                         target_transform= None)
test_data = ImageFolder(root = test_directory,
                        transform = transforms)

train_dataloader = DataLoader(dataset = train_data,
                              batch_size = 1,
                              shuffle = False)
test_dataloader = DataLoader(dataset = test_data,
                             batch_size = 1,
                             shuffle = False)
test_img = None
test_label = None
for batch, (x,y) in enumerate(test_dataloader):
    if batch == 8:
        test_img = x
        test_label = y

class ConvoModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 3,
                            out_channels = 20,
                            kernel_size = 3,
                            stride = 1,
                            padding = 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels = 20,
                            out_channels = 20,
                            kernel_size = 3,
                            stride = 1,
                            padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2,
                               stride = 2) 
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 20,
                            out_channels = 20,
                            kernel_size = 3,
                            stride = 1,
                            padding = 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels = 20,
                            out_channels = 20,
                            kernel_size = 3,
                            stride = 1,
                            padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2,
                               stride = 2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 20,
                            out_channels = 20,
                            kernel_size = 3,
                            stride = 2,
                            padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2,
                               stride = 2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features = 3920,
                            out_features = 3)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.classifier(x)
        return x
    
model0 = ConvoModel() 
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model0.parameters(),
                             lr = 0.001)

epochs = 40
model0.train()
train_acc = 0

for i in range(epochs):
    for (x,y) in train_dataloader:
        y_pred = model0(x)

        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)

        if i == 39 and y_pred_class == y :
            train_acc += 1  
       
train_acc = train_acc / len(train_dataloader)

x = f'{train_acc * 100} %'
print(x)

with torch.inference_mode():
    model0.eval()
    y_test_pred = model0(test_img)
    y_test_pred_class = torch.argmax(torch.softmax(y_test_pred, dim=1), dim=1)
    if y_test_pred_class == test_label:
        print('Beautiful!')
    print(y_test_pred_class)      