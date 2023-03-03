# Pytorch Blog 1: 基础使用与NN实战

参考数据集&&代码仓库: [pytorch-learn](https://github.com/openhe-hub/pytorch-learn.git)

## 1. Pytorch快速入门

### Tensor张量
[tensor详解](https://zhuanlan.zhihu.com/p/142859846)
### 格式转换
* numpy ndarray转tensor：`arr_tensor=torch.from_numpy(arr_numpy)` 
* tensor转numpy：`arr_numpy=arr_tensor.numpy()`
### Autograd 自动求导机制
[自动求导讲解](https://zhuanlan.zhihu.com/p/148669484)
### Model 

### 代价函数与优化器
* 代价函数
    [Pytorch常用代价函数](https://blog.csdn.net/shanglianlm/article/details/85019768)
    对于新手而言：回归任务选：均方差误差损失`MSELoss`，分类任务选交叉熵损失`CrossEntropyLoss`
* 优化器
    [Pytorch常用优化器](https://www.jianshu.com/p/1a1339c4acd7)
    对于新手而言：
        * `SGD`: 原始的梯度下降优化器
        * `Adam`：实时修正学习率的梯度下降优化器
### CUDA加速
* 检测是否有可用的cuda，然后将模型和数据移到cuda，保持数据在cpu和gpu运算的一致，不可跨设备运算
    ```
    device="cuda:0" if torch.cuda.is_available() else "cpu"
    x=x.to(device)
    ```
* n卡监控常用命令
    * nvidia-smi：nvidia官方命令
    * nvtop： 第三方可视化命令，类似于htop
     
### Dataset 与 Dataloader

## 2. NN实战1（回归任务demo） ： 气温预测

### 数据读入与可视化

* csv数据集读入
    ```python
    dataset = pd.read_csv("../../../assets/temp-prediction-dataset.csv")
    dataset.head()
    ```
* 数据集列名注解
    * temp1 : 前天气温
    * temp2 : 昨天气温
    * average：往年平均
    * actual：实际预测
    * friend：朋友预测
* matplotlib画图
    ```python
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))

    axs[0][0].plot(dates, dataset['temp_2'])
    axs[0][1].plot(dates, dataset['temp_1'])
    axs[1][0].plot(dates, dataset['average'])
    axs[1][1].plot(dates, dataset['actual'])

    plt.show()
    ```
    ![](https://raw.githubusercontent.com/openhe-hub/my-img-repo/main/img/uTools_1677395039141.png?token=ASKA7IP5IDECJCR5XACCXVTD7MCKC)

### 数据预处理  
* 处理时间格式  
  ```python
    import datetime

    years = dataset['year']
    months = dataset['month']
    days = dataset['day']

    dates = [f"{year}-{month}-{day}" for year, month, day in zip(years, months, days)]
    dates = [datetime.datetime.strptime(date, "%Y-%m-%d") for date in dates]
    dates[:5]
  ``` 
* 离散化星期：one-hot decoding
  星期字符串无法进行模型训练，使用one-hot编码将其离散化
  ```python
  dataset = pd.get_dummies(dataset)
  dataset.head()
  ``` 
  效果如下:
  ![](https://raw.githubusercontent.com/openhe-hub/my-img-repo/main/img/20230226151206.png)
* 归一化
    ```python
    # 提取features和labels，转换成numpy格式
    labels = np.array(dataset['actual'])
    features = np.array(dataset.drop('actual', axis=1))
    labels.shape, features.shape
    # 预处理：数据标准化
    ```
    ```python
    from sklearn import preprocessing

    std_features = preprocessing.StandardScaler().fit_transform(features)
    std_features[:5]
    ``` 
### 模型与参数
* 模型：
    > nn.Sequential 使用顺序连接，省去写前向传播的代码
    ```python
    tempPredictionModel = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size),
        torch.nn.Sigmoid(),
        torch.nn.Linear(hidden_size, output_size)
    )
    ``` 
* 优化器与代价函数
    ```python
    cost = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(tempPredictionModel.parameters(), lr=learning_rate)
    ``` 
* 参数
    ```python
    input_size = std_features.shape[1]
    hidden_size = 128
    output_size = 1
    batch_size = 12
    epochs = 2000
    learning_rate=0.001
    ``` 
### 训练
* cuda加速
    ```python
    # data
    x = torch.tensor(std_features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.float)
    #cuda
    device="cuda:0" if torch.cuda.is_available() else "cpu"
    x=x.to(device)
    y=y.to(device)
    tempPredictionModel=tempPredictionModel.to(device)
    ```
* 训练
    ```python
    # train
    for epoch in range(epochs):
        batch_loss = []
        for start in range(0, len(std_features), batch_size):
            # 抽取batch数据
            end = start + batch_size if (start + batch_size) < len(std_features) else len(std_features)
            xx = torch.tensor(std_features[start:end], dtype=torch.float, requires_grad=True)
            yy = torch.tensor(labels[start:end], dtype=torch.float, requires_grad=True)

            prediction = tempPredictionModel(xx)  # 前向传播
            prediction.squeeze_(-1)
            loss = cost(prediction, yy)  # 计算代价函数
            optimizer.zero_grad()  # 梯度清零
            loss.backward(retain_graph=True) # 反向传播
            optimizer.step() # 更新

            batch_loss.append(loss.data.numpy())

        if epoch % 100 ==0:
            print(f"{epoch}/{epochs}, loss={np.mean(batch_loss)}")
    ```
    注意：  
        1. 使用batch训练，而不是全部数据
        2. 训练步骤为：梯度清零，前向传播，计算代价函数，反向传播，更新
### 结果与可视化
* 评估和可视化
    ```python
    # eval
    prediction=tempPredictionModel(x)
    plt.plot(dates,dataset['actual'])
    plt.plot(dates,prediction.data.numpy())
    ```
    ![](https://raw.githubusercontent.com/openhe-hub/my-img-repo/main/img/20230226151944.png)
## 3. NN实战2 （分类任务demo）：MNIST手写数字识别
### 数据读入与可视化
* 使用torch提供的数据集
    ```python
    # download dataset
    from torch.utils.data import DataLoader
    import torchvision

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    batch_size = 64

    train_dataset = torchvision.datasets.MNIST(root="../../../assets/", train=True, download=True,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),  # 转换成张量
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 标准化
                                            ]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataset = torchvision.datasets.MNIST(root="../../../assets/", train=False, download=True,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),  # 转换成张量
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 标准化
                                            ]))
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    ``` 
* matplotlib画图
    ```python
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(5, 5))
    for i, (x, y) in enumerate(train_loader):
        if i == 0:
            for j in range(9):
                idx, idy = int(j / 3), j % 3
                axs[idx][idy].imshow(x[j].reshape((28, 28)), cmap="gray")
                axs[idx][idy].axis('off')
                axs[idx][idy].set_title(y[j].item())
        else:
            break
    ```
    效果
    ![](https://raw.githubusercontent.com/openhe-hub/my-img-repo/main/img/20230226152705.png)

### 模型与参数
* 模型
    使用两层线形层，用`relu`做激活函数，输出层矩阵代表各个数字的识别概率，最大值为结果
    ```python
    from torch import nn
    import torch.nn.functional as F


    class MnistClassificationModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden1 = nn.Linear(28 * 28, 128)
            self.hidden2 = nn.Linear(128, 256)
            self.out = nn.Linear(256, 10)

        def forward(self, x):
            x = F.relu(self.hidden1(x))
            x = F.relu(self.hidden2(x))
            x = self.out(x)
            return x
    ```
    ![](https://raw.githubusercontent.com/openhe-hub/my-img-repo/main/img/20230226152942.png)
* 代价函数和优化器
    ```python
    # optimizer & cost function
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    cost = F.cross_entropy
    ``` 
### 训练
* cuda加速
    ```python
    use_cuda = torch.cuda.is_available()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    ``` 
* 训练
    ```python
    # train
    def train():
        model.train()
        for epoch in range(epochs):
            batch_loss = []
            for batch_id, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                batch_x = batch_x.reshape(len(batch_x), 28 * 28)

                optimizer.zero_grad()  # 梯度清零
                output = model(batch_x)  # 前向传播
                loss = cost(output, batch_y)  # 计算代价函数
                loss.backward()  #反向传播
                optimizer.step()  #更新
                if use_cuda:
                    batch_loss.append(loss.data.cpu().numpy())
                else:
                    batch_loss.append(loss.data.numpy())

                if batch_id % 100 == 0 and len(batch_x) == batch_size:
                    print(
                        f"\tepoch={epoch}, {batch_id * batch_size}/{len(train_loader.dataset)},loss={np.mean(batch_loss)}")

    train()
    ```
### 结果与可视化
* 评价模型
    ```python
    # test
    def test():
        model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for (test_x, test_y) in test_loader:
                test_x = test_x.to(device)
                test_y = test_y.to(device)

                test_x = test_x.reshape(len(test_x), 28 * 28)
                output = model(test_x)
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(test_y.data.view_as(pred)).sum()

        print(f"accuracy={correct / len(test_loader.dataset)}")


    test()
    # result: accuracy=0.907800018787384
    ``` 
* 测试集分类结果可视化
    ```python
    # visualize
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))

    for i, (test_x, test_y) in enumerate(test_loader):
        test_x = test_x.to(device)
        test_y = test_y.to(device)

        test_x = test_x.reshape(len(test_x), 28 * 28)
        output = model(test_x)
        pred = output.data.max(1, keepdim=True)[1]
        if i == 0:
            for j in range(9):
                idx, idy = int(j / 3), j % 3
                axs[idx][idy].imshow(test_x[j].reshape((28, 28)), cmap="gray")
                axs[idx][idy].axis('off')
                axs[idx][idy].set_title(f"pred={pred[j].item()}, actual={test_y[j].item()}")
        else:
            break
    ```
    ![](https://raw.githubusercontent.com/openhe-hub/my-img-repo/main/img/20230226153345.png)









