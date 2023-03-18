# Pytorch Blog 3 CNN实战与迁移学习
> 参考数据集&&代码仓库: [pytorch-learn](https://github.com/openhe-hub/pytorch-learn.git)
> PoweredBy **CHATGPT**
## CNN经典网络模型介绍
### AlexNet
AlexNet是一种深度卷积神经网络模型，由Alex Krizhevsky、Ilya Sutskever和Geoffrey Hinton在2012年提出。它是第一个在ImageNet数据集上取得卓越性能的深度学习模型，并推动了深度学习在计算机视觉领域的发展。

AlexNet由8个卷积层和3个全连接层组成，共计约600万个参数。相对于以前的神经网络，AlexNet具有以下几个创新点：

* 使用了ReLU激活函数。相对于传统的sigmoid激活函数，ReLU在计算速度和性能方面更优秀。

* 使用了数据增强技术。AlexNet在训练数据中增加了一些随机的变化，如图像翻转、裁剪和旋转等，以增加数据的多样性，减轻过拟合的风险。

* 使用了GPU加速。AlexNet使用了两个NVIDIA GTX 580 GPU来加速训练，大大缩短了训练时间。

* 使用了Dropout技术。Dropout可以随机地将一些神经元的输出设置为0，以减少网络的过拟合。
### Vgg
VGG是一种深度卷积神经网络模型，由Oxford大学的VGG（Visual Geometry Group）团队于2014年提出。VGG通过增加网络深度来提高准确性，使用了多个小卷积核代替单个大卷积核，以及使用更少的池化层来减少信息丢失。

VGG网络的核心思想是使用多个3x3的小卷积核代替一个更大的卷积核，以增加网络的深度。VGG网络共分为5个部分，每个部分包含若干个卷积层和一个池化层，最后通过3个全连接层进行分类。其中VGG16和VGG19是最为著名的两个版本，分别使用了16个和19个卷积层。

相对于AlexNet，VGG网络具有以下几个优点：

* VGG网络更加深度，可以提取更丰富的特征。

* 使用小卷积核可以减少网络参数，降低过拟合的风险。

* 使用更少的池化层可以减少信息的丢失。
### ResNet
ResNet是一种深度卷积神经网络模型，由Microsoft Research团队于2015年提出。ResNet通过引入残差块（Residual Block）来解决网络深度增加导致的梯度消失问题，使得网络可以更容易地训练超过1000层。

传统的深度神经网络中，随着网络深度的增加，梯度信息逐渐消失，导致模型难以训练。ResNet通过使用残差块来构建深层网络，可以让网络学习到残差（即跨越多层的映射），从而避免梯度消失问题。在ResNet中，每个残差块都包含了一个跳跃连接（skip connection），允许梯度在整个网络中自由传递。

ResNet共有几个版本，其中最著名的是ResNet-50、ResNet-101和ResNet-152，它们分别使用50、101和152个卷积层来构建网络。在ResNet中，每个残差块通常由两个或三个卷积层和一个跳跃连接组成。

相比于以往的深度神经网络，ResNet具有以下几个优点：

* 能够构建更深的神经网络，提取更丰富的特征。

* 能够更容易地训练深度神经网络。

* 能够在不增加参数数量的情况下提高准确性。
## 迁移学习
1. 特征提取：从一个预训练模型中提取特征，并将这些特征用于新任务中。这是最常见的迁移学习操作之一，通常可以通过去掉预训练模型的分类器层，并在其顶部添加一个新的分类器层来实现。

2. 微调：使用新数据集来微调预训练模型的权重，以便更好地适应新任务。在微调期间，预训练模型的前几层通常被锁定，只有最后几层的权重会被更新。

3. 多任务学习：使用预训练模型来同时处理多个任务。在这种情况下，模型通常被设计成共享底层特征提取器，但具有不同的顶层分类器。

4. 深度迁移学习：将一个预训练模型的部分或全部权重迁移到一个新的更深的模型中。这可以通过在新模型的底部添加与预训练模型相同的层来实现。

5. 知识蒸馏：使用一个大的、复杂的预训练模型来教授一个小的、简单的模型。这可以通过将大模型的输出作为小模型的目标标签来实现。

* 对于入门者来说，主要是：在模型全连接层修改参数；冻结前面的卷积层，来快速训练网络，数据预处理对齐
## 数据整理与预处理
### 数据集说明
这次数据集为牛津102-flower数据集，可从[102-flower](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)下载
### 数据集分类：ImageFolder格式
* ImageFolder是PyTorch中常用的数据集格式，用于读取文件夹中的图像数据。该格式要求将数据集按类别分别存储在不同的文件夹中，每个文件夹的名称即为类别名称，同时要求数据集中的每个图像文件都包含该图像所属的类别信息。
* 整理数据脚本：[dataset-gen](https://github.com/openhe-hub/pytorch-learn/blob/master/src/cnn/flower102/dataset-gen.ipynb)
### cv常用数据增强
```python
data_transform = {
    'train': transforms.Compose([transforms.RandomRotation(45),
                                 transforms.CenterCrop(224),
                                 transforms.RandomVerticalFlip(p=0.5),
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
                                 transforms.RandomGrayscale(p=0.025),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'valid': transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'test': transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}
```
* RandomRotation(45)：随机旋转图像45度以内的角度，增加数据的多样性和鲁棒性。
* CenterCrop(224)：对图像进行中心裁剪，将其缩放为指定大小，以保留图像的中心信息。
* RandomVerticalFlip(p=0.5)：以0.5的概率随机进行图像垂直翻转，增加数据的多样性和鲁棒性。
* RandomHorizontalFlip(p=0.5)：以0.5的概率随机进行图像水平翻转，增加数据的多样性和鲁棒性。
* ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1)：对图像的亮度、对比度、饱和度和色调进行随机的调整，增加数据的多样性。
* RandomGrayscale(p=0.025)：以0.025的概率将图像转换为灰度图像，增加数据的多样性。
* ToTensor()：将图像数据转换为Tensor类型。
* Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])：对图像数据进行标准化处理，均值为[0.485, 0.456, 0.406]，标准差为[0.229, 0.224, 0.225]，与resnet对齐
## Pytorch实战
### 数据载入
```python
# load data
batch_size = 48
train_datasets = datasets.ImageFolder(os.path.join(base_dir, "train"), data_transform["train"])
test_datasets = datasets.ImageFolder(os.path.join(base_dir, "test"), data_transform["test"])
valid_datasets = datasets.ImageFolder(os.path.join(base_dir, "valid"), data_transform["valid"])
train_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_datasets, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_datasets, batch_size=batch_size, shuffle=True)

train_datasets, test_datasets, valid_datasets
```
### 迁移ResNet152
```
class FlowerModel(nn.Module):
    def __init__(self, model):
        super(FlowerModel, self).__init__()
        self.resnet = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Linear(in_features=2048, out_features=102)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
    
resnet152_model = models.resnet152(pretrained=True)
model = FlowerModel(resnet152_model)
```
### 训练参数准备
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
epochs = 40
```
### 训练
```python
def train_model(model,criterion,optimizer,epochs):
    model=model.to(device)
    model.train()

    epoch_loss=0.0
    for epoch in range(epochs):
        losses=0.0
        for (idx,(inputs,labels)) in enumerate(train_dataloader):
            inputs=inputs.to(device)
            labels=labels.to(device)

            optimizer.zero_grad() # 梯度清空
            outputs=model(inputs) # 前向传播
            loss=criterion(outputs,labels) # 计算损失函数
            loss.backward() # 反向传播
            optimizer.step() # 更新参数

            losses+=loss.item()

            if idx % 20==0:
                print(f"epoch={epoch}/{epochs}, {(idx+1)*batch_size}/{len(train_datasets)}, loss={losses/((idx+1)*batch_size)}")

        epoch_loss=losses/len(train_datasets)
        print(f"epoch={epoch}/{epochs}, losses={epoch_loss}")

train_model(model,criterion,optimizer,epochs)
```
### 测试
```python
def eval_model(model, criterion, dataset, dataloader):
    best_acc = 0.0
    running_loss = 0.0
    running_corrects = 0

    model = model.to(device)
    for (idx, (inputs, labels)) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        model.eval()

        with torch.no_grad():
            outputs = model(inputs)

        loss = criterion(outputs, labels)
        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)  # val, idx
        running_corrects += torch.sum(preds == labels)

        print(f"{(idx + 1) * batch_size / len(dataset)}")

    epoch_loss = running_loss / len(dataset)
    epoch_acc = running_corrects / len(dataset)

    print(f"loss={epoch_loss}, acc={epoch_acc}")
```
```python
eval_model(model,criterion,test_datasets,test_dataloader)
```
> result: `acc=0.997`