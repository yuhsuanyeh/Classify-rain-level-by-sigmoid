#---------------------------------------------------Hi Andy! This code is for estimate rain is Heavy or Medium or Light By sigmoid---------------------------------------------------
from __future__ import print_function, division
import os
from tqdm import tqdm
import numpy as np
from torch.autograd import Variable
# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms, utils
import torch.nn.functional as F
import torch.nn as nn

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 32
LR = 0.001              # learning rate
DOWNLOAD_MNIST = False

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def progbar(curr, total, full_progbar, is_done) :
    """
        Plot progress bar on terminal
        Args :
            curr (int) : current progress
            total (int) : total progress
            full_progbar (int) : length of progress bar
            is_done (bool) : is already done
    """
    frac = curr/total
    filled_progbar = round(frac*full_progbar)

    if is_done == True :
        print('\r|'+'#'*full_progbar + '|  [{:>7.2%}]'.format(1) , end='')
    else :
        print('\r|'+'#'*filled_progbar + '-'*(full_progbar-filled_progbar) + '|  [{:>7.2%}]'.format(frac) , end='')


'''
# Mnist digits dataset
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,
)
'''
# train_data = torchvision.datasets.CIFAR10('./data', train=True, download=False,
#                                            transform=transforms.Compose([
#                                            transforms.ToTensor()]))
train_data = torchvision.datasets.ImageFolder('./project_derain/training_data/',
                                                 transform=transforms.Compose([
                                                 transforms.ToTensor()]))
test_data = torchvision.datasets.ImageFolder('./project_derain/testing_data/',
                                                 transform=transforms.Compose([
                                                 transforms.ToTensor()]))
# test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transforms.Compose([
#                                         transforms.ToTensor()]))
print(len(train_data)) #24000
data_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle =True,num_workers = 2)
print(len(data_loader)) #750
#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse','ship', 'truck')
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers = 2)

classes = ('nonrain','rain')

def imshow(img):
    #img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

dataiter = iter(data_loader)
images, labels = dataiter.next()
print(images[0].shape) #torch.Size([3, 512, 512])
print(labels) #tensor([ 1,  1,  1,  1,  0,  1,  1,  0,  0,  0,  0,  0,  1,  1,  1,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  1,  1,  1,  0,  1,  1])

for j in range(BATCH_SIZE) :
    print('{}    '.format(classes[labels[j]]), end = '')

#imshow(torchvision.utils.make_grid(images))
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (3, 512, 512)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 512, 512)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 256, 256)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 256, 256)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 128, 128)
        )
        self.conv3 = nn.Sequential(         # input shape (32, 128, 128)
            nn.Conv2d(32, 64, 5, 1, 2),     # output shape (64, 128, 128)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (64, 64, 64)
        )
        self.out = nn.Linear(64 * 64 * 64, 1)   # fully connected layer, output 2 classes

    def forward(self, x):
        y = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 64 * 64 * 64)
        x = self.out(x)
#        print("input size: ",y.size(),
#            "output size: ",x.size())
        x = F.sigmoid(x)
        return x    # return x for visualization
'''
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, , 5, 1, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5 , 1, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))

        return x
'''
cnn = CNN()
print(cnn)  # net architecture
cnn.cuda()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.BCELoss()                     # the target label is not one-hotted

#criterion = nn.CrossEntropyLoss()

for epoch in range(1):  # loop over the dataset multiple times
    #pbar = tqdm(total = 2000)
    progress = 0
    running_loss = 0.0
    print('epoch : {} '.format(epoch))

    for i, data in enumerate(data_loader, 0):
        # get the inputs
        inputs, labels = data
        labels = labels.type(torch.FloatTensor)
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        # zero the parameter gradients
        optimizer.zero_grad()
        
        #print('inputs : {}'.format(len(inputs)))
        #print('inputs : {}'.format(inputs.shape))
        #print('outputs : {}'.format((outputs[0][0])))
        #print('labels : {}'.format(len(labels)))
        #print('labels : {}'.format(labels.shape))
        #input()
        # forward + backward + optimize
        outputs = cnn(inputs)
        # print('inputs : {}'.format(len(inputs)))
        # print('outputs : {}'.format(outputs))
        # print('labels : {}'.format(len(labels)))
        #input()
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        #pbar.update(1)
        # print statistics
#        running_loss += loss.item()
        running_loss += loss.data[0]
        progbar(progress, 750, 40, (progress == 750-1))
        progress += 1
        if i % 750 == 749:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 750))
            running_loss = 0.0
   # pbar.close()
    save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'my_cnn',
            'state_dict': cnn.state_dict(),
            'optimizer' : optimizer.state_dict(),
        })     
            
  
print('Finished Training')

# load test data----------------------------------------------------------
dataiter = iter(data_loader)
images, labels = dataiter.next()
#labels.cuda()
# print images

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(BATCH_SIZE)))

########################################################################
# Okay, now let us see what the neural network thinks these examples above are:
outputs = cnn(Variable(images.cuda()))
#outputs = cnn(images)
predicted = []
#_, predicted = torch.max(outputs, 1)
for i in range(BATCH_SIZE):
    if outputs[i] > 0.5:
        predicted.append(classes[1]) 
    else:
        predicted.append(classes[0])

print('Predicted: ', ' '.join('%5s' % predicted[j] for j in range(BATCH_SIZE)))
print('Predicted Prob : ', ' '.join('%f' % outputs[j]for j in range(BATCH_SIZE)))

imshow(torchvision.utils.make_grid(images))

correct = 0
total = 0
with torch.no_grad():
    for data in data_loader:
        images, labels = data
        #labels = labels.type(torch.FloatTensor)
        outputs = cnn(Variable(images.cuda()))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))
with torch.no_grad():
    for data in data_loader:
        images, labels = data
        #labels = labels.type(torch.FloatTensor)
        outputs = cnn(Variable(images.cuda()))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels.cuda()).squeeze()
        for i in range(BATCH_SIZE):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(2):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assume that we are on a CUDA machine, then this should print a CUDA device:

print(device)
'''
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (3, 512, 512)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 512, 512)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 256, 256)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 256, 256)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 128, 128)
        )
        self.out = nn.Linear(32 * 128 * 128, 2)   # fully connected layer, output 2 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization
'''

'''
# following function (plot_with_labels) is for visualization, can be ignored if not interested
from matplotlib import cm
try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; print('Please install sklearn for layer visualization')
def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

plt.ion()
# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(data_loader):   # gives batch data, normalize x when iterate train_loader

        output = cnn(b_x)[0]               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step % 50 == 0:
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze().numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
            if HAS_SK:
                # Visualization of trained flatten layer (T-SNE)
                tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                plot_only = 500
                low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                labels = test_y.numpy()[:plot_only]
                plot_with_labels(low_dim_embs, labels)
plt.ioff()

# print 10 predictions from test data
test_output, _ = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')
'''