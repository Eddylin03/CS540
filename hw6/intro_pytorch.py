import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

def get_data_loader(training=True):
    """
    INPUT: 
        An optional boolean argument (default value is True for training dataset)
    RETURNS:
        DataLoader for the training set (if training=True) or the test set (if training=False)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    if training:
        data_set = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        loader = torch.utils.data.DataLoader(data_set, batch_size=64, shuffle=True)
    else:
        data_set = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
        loader = torch.utils.data.DataLoader(data_set, batch_size=64, shuffle=False)
    
    return loader

def build_model():
    """
    INPUT: 
        None
    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),  # Input size is 28*28=784, output size is 128
        nn.ReLU(),
        nn.Linear(128, 64),     # Input size is 128, output size is 64
        nn.ReLU(),
        nn.Linear(64, 10)       # Input size is 64, output size is 10
    )
    return model

def train_model(model, train_loader, criterion, T):
    """
    INPUT: 
        model - the model produced by the previous function
        train_loader - the train DataLoader produced by the first function
        criterion - cross-entropy loss function
        T - number of epochs for training
    RETURNS:
        None
    """
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    
    for epoch in range(T):
        correct = 0
        total_loss = 0.0
        total_samples = 0
        
        for data, target in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            # Update statistics
            _, predicted = outputs.max(1)
            correct += predicted.eq(target).sum().item()
            total_loss += loss.item() * data.size(0)  # Multiply by batch size
            total_samples += data.size(0)
        
        avg_loss = total_loss / total_samples
        accuracy = 100. * correct / total_samples
        print('Train Epoch: {} Accuracy: {}/{}({:.2f}%) Loss: {:.3f}'.format(
            epoch, correct, total_samples, accuracy, avg_loss))

def evaluate_model(model, test_loader, criterion, show_loss=True):
    """
    INPUT: 
        model - the trained model produced by the previous function
        test_loader - the test DataLoader
        criterion - cross-entropy loss function
    RETURNS:
        None
    """
    model.eval()
    correct = 0
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            loss = criterion(outputs, target)
            total_loss += loss.item() * data.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(target).sum().item()
            total_samples += data.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = 100. * correct / total_samples
    
    if show_loss:
        print('Average loss: {:.4f}'.format(avg_loss))
    print('Accuracy: {:.2f}%'.format(accuracy))

def predict_label(model, test_images, index):
    """
    INPUT: 
        model - the trained model
        test_images - a tensor of shape Nx1x28x28
        index - specific index i of the image to be tested: 0 <= i <= N - 1
    RETURNS:
        None
    """
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
                   'Sandal','Shirt','Sneaker','Bag','Ankle Boot']
    model.eval()
    with torch.no_grad():
        image = test_images[index].unsqueeze(0)  # Add batch dimension
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        prob, predicted = probabilities.topk(3, dim=1)
        prob = prob.squeeze()
        predicted = predicted.squeeze()
        for i in range(3):
            label = class_names[predicted[i].item()]
            prob_percent = prob[i].item() * 100
            print('{}: {:.2f}%'.format(label, prob_percent))

if __name__ == '__main__':
    '''
    Feel free to write your own test code here to examine the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
    train_loader = get_data_loader()
    test_loader = get_data_loader(False)
    model = build_model()
    train_model(model, train_loader, criterion, T=5)
    evaluate_model(model, test_loader, criterion, show_loss=True)
    test_images, _ = next(iter(test_loader))
    predict_label(model, test_images, index=0)
