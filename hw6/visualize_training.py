import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def get_data_loader(training=True):
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
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),  # Input size is 28*28=784, output size is 128
        nn.ReLU(),
        nn.Linear(128, 64),     # Input size is 128, output size is 64
        nn.ReLU(),
        nn.Linear(64, 10)       # Input size is 64, output size is 10
    )
    return model

def train_and_visualize(model, train_loader, test_loader, criterion, optimizer, epochs):
    model.train()
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(epochs):
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
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
        
        avg_loss = total_loss / total_samples
        accuracy = 100. * correct / total_samples
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
        # Evaluate on test data
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, return_metrics=True)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        print('Epoch {}: Train Loss: {:.4f}, Train Acc: {:.2f}%, Test Loss: {:.4f}, Test Acc: {:.2f}%'.format(
            epoch, avg_loss, accuracy, test_loss, test_accuracy))
    
    # Plot loss and accuracy curves
    plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies)
    
    # Visualize weights and activations
    visualize_weights(model)
    visualize_activations(model, test_loader)

def evaluate_model(model, test_loader, criterion, show_loss=True, return_metrics=False):
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
    
    if not return_metrics:
        if show_loss:
            print('Average loss: {:.4f}'.format(avg_loss))
        print('Accuracy: {:.2f}%'.format(accuracy))
    else:
        return avg_loss, accuracy

def plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies):
    epochs = range(len(train_losses))
    
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, test_losses, 'r-', label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, test_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_weights(model):
    for idx, layer in enumerate(model):
        if isinstance(layer, nn.Linear):
            weights = layer.weight.data
            plt.figure(figsize=(10, 5))
            plt.hist(weights.numpy().flatten(), bins=50)
            plt.title('Weight Distribution in Layer {}'.format(idx))
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
            plt.show()

def visualize_activations(model, test_loader):
    data_iter = iter(test_loader)
    images, labels = data_iter.next()
    images = images[:5]  # Take first 5 images
    labels = labels[:5]
    
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks to capture activations
    model[1].register_forward_hook(get_activation('fc1'))
    model[3].register_forward_hook(get_activation('fc2'))
    
    # Forward pass
    outputs = model(images)
    
    # Visualize activations
    for i in range(len(images)):
        img = images[i].squeeze().numpy()
        plt.imshow(img, cmap='gray')
        plt.title('Input Image - Label: {}'.format(labels[i].item()))
        plt.show()
        
        fc1_act = activations['fc1'][i]
        fc2_act = activations['fc2'][i]
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.bar(range(len(fc1_act)), fc1_act.numpy())
        plt.title('Activations of Layer fc1')
        
        plt.subplot(1, 2, 2)
        plt.bar(range(len(fc2_act)), fc2_act.numpy())
        plt.title('Activations of Layer fc2')
        
        plt.show()

if __name__ == '__main__':
    # Initialize Data Loaders
    train_loader = get_data_loader(training=True)
    test_loader = get_data_loader(training=False)
    
    # Build Model
    model = build_model()
    
    # Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Train the Model and Visualize
    train_and_visualize(model, train_loader, test_loader, criterion, optimizer, epochs=5)
