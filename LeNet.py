#

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary


class FashionClass(nn.Module):
    def __init__(self):
        super(FashionClass, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)# padding in conv1 changed its dim from 16*4*4
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
       # x = x.view(-1, 16 * 5 * 5)
    

loss_functn = torch.nn.CrossEntropyLoss()

def train(argms, model, device, training_loader, optimizer, epoch):
    model.train()
    active_loss = 0.
    b_loss = 0.
    for batch_idx1, (data, labels) in enumerate(training_loader):
        #print(batch_idx1)
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(data)
        #print(output)
        #print(target)

        loss =  loss_functn(output, labels)

        loss.backward()
        optimizer.step()
        active_loss += loss.item()

        if batch_idx1 % 1000 == 999:
            b_loss = active_loss / 1000 # loss per batch

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tbatch no: {} \tbatch Loss: {:.6f}'.format(
                epoch, batch_idx1 * len(data), len(training_loader.dataset),
                100. * batch_idx1 / len(training_loader), loss.item(),batch_idx1, b_loss))
            active_loss = 0.

          


def test(model, device, testing_loader):
    model.eval()
    test_loss = 0
    correct_pre = 0
    active_vloss = 0.0
    with torch.no_grad():
        for i,(data, labels) in enumerate(testing_loader):
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            test_loss =  loss_functn(output, labels)
            active_vloss += test_loss

            predictd = output.argmax(dim=1, keepdim=True)  
            correct_pre += predictd.eq(labels.view_as(predictd)).sum().item()

    average_vloss = active_vloss / (i + 1)


    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        average_vloss, correct_pre, len(testing_loader.dataset),
        100. * correct_pre / len(testing_loader.dataset)))



                     
def main():
    parser = argparse.ArgumentParser(description='fashionMNIST LeNet example using pytorch')
    parser.add_argument('--train-batch-size', type=int, default=8, metavar='N',
                        help='Input trainnig batch size default:4')
    parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                        help='Input test batch size default:4')
    parser.add_argument('--no-epochs', type=int, default=10, metavar='N',
                        help='Total number of epochs to train default:5')
    parser.add_argument('--learning-rate', type=float, default=0.001, metavar='LR',
                        help='Learning rate default:0.001')
    parser.add_argument('--nocuda', action='store_true', default=False,
                        help='disabling CUDA training')
    parser.add_argument('--store-model', action='store_true', default=True,
                        help='To save the Model')
    argms = parser.parse_args()
    
    useCuda = not argms.nocuda and torch.cuda.is_available()


    if useCuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")



                        
    training_kwargms = {'batch_size': argms.train_batch_size}
    testing_kwargms = {'batch_size': argms.test_batch_size}
    if useCuda:
        cuda_kwargms = {'num_workers': 1,
                       'shuffle': True}
        training_kwargms.update(cuda_kwargms)
        testing_kwargms.update(cuda_kwargms)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4,), (0.4,))
        ])
    dataset_train = datasets.FashionMNIST('./data', train=True, download=True,
                       transform=transform)
    dataset_test = datasets.FashionMNIST('./data', train=False,
                       transform=transform)
    
    training_loader = torch.utils.data.DataLoader(dataset_train,**training_kwargms)
    testing_loader = torch.utils.data.DataLoader(dataset_test, **testing_kwargms)
    
       
    
    # split sizes
    print('Training data set has {} exampeles'.format(len(training_loader)))
    print('Validation data set has {} exmaples'.format(len(testing_loader)))
    print(torch.__version__)
    print(device)

    model = FashionClass().to(device)
    optimizer = optim.Adam(model.parameters(), lr=argms.learning_rate)

    #print(model)
    summary(model, input_size=(1, 28, 28))

    for epoch in range(1, argms.no_epochs + 1):
        train(argms, model, device, training_loader, optimizer, epoch)
        test(model, device, testing_loader)
        optimizer.step()

    if argms.store_model:
        torch.save(model.state_dict(), "C:/Users/USER/Desktop/training_save/model_saving/fashoinmnist_cnnLeNet.pt")


if __name__ == '__main__':
    main()