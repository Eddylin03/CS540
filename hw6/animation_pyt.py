import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
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
        nn.Linear(28*28, 128),  # Layer index 1
        nn.ReLU(),
        nn.Linear(128, 64),     # Layer index 3
        nn.ReLU(),
        nn.Linear(64, 10)       # Layer index 5
    )
    return model
def train_model_with_recording(model, train_loader, criterion, optimizer, epochs):
    model.train()
    weight_history = []  # To store weights of a specific layer over epochs

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
        print('Epoch {}: Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch, avg_loss, accuracy))
        
        # Record weights of a specific layer (e.g., the first Linear layer)
        # Access the layer using its index in the Sequential model
        layer_weights = model[1].weight.data.clone().cpu().numpy()
        weight_history.append(layer_weights)
    
    return weight_history
def animate_weights(weight_history, layer_name='Layer 1'):
    fig, ax = plt.subplots()
    ims = []

    for i, weights in enumerate(weight_history):
        im = ax.hist(weights.flatten(), bins=50, range=(-0.5, 0.5), color='blue')
        ax.set_title('{} Weight Distribution at Epoch {}'.format(layer_name, i))
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Frequency')
        ims.append(im)
        # Clear the axis to prevent overlapping histograms
        ax.cla()
    
    # Create the animation
    def update_hist(num):
        ax.hist(weight_history[num].flatten(), bins=50, range=(-0.5, 0.5), color='blue')
        ax.set_title('{} Weight Distribution at Epoch {}'.format(layer_name, num))
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Frequency')
    
    ani = animation.FuncAnimation(fig, update_hist, frames=len(weight_history), repeat=False)
    plt.show()
    # Save the animation as a GIF or MP4
    ani.save('{}_weights_animation.gif'.format(layer_name.replace(' ', '_')), writer='imagemagick')
def get_activation(model, data):
    activations = {}
    
    def hook_fn(module, input, output):
        activations[module] = output.detach().cpu().numpy()
    
    hooks = []
    for layer in model:
        if isinstance(layer, nn.Linear):
            hooks.append(layer.register_forward_hook(hook_fn))
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(data)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return activations
def train_model_with_activation_recording(model, train_loader, fixed_input, criterion, optimizer, epochs):
    model.train()
    activation_history = []

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
        print('Epoch {}: Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch, avg_loss, accuracy))
        
        # Record activations for the fixed input
        activations = get_activation(model, fixed_input)
        activation_history.append(activations)
    
    return activation_history
def animate_activations(activation_history, layer_index=1, neuron_index=0):
    fig, ax = plt.subplots()
    
    activations_over_time = []
    for activations in activation_history:
        # Since activations is a dictionary with layers as keys
        layer = list(activations.keys())[layer_index]
        activation = activations[layer][0][neuron_index]  # First sample, specified neuron
        activations_over_time.append(activation)
    
    epochs = np.arange(len(activation_history))
    
    def update_line(num, data, line):
        line.set_data(epochs[:num], data[:num])
        return line,
    
    ax.set_xlim(0, len(activation_history)-1)
    ax.set_ylim(min(activations_over_time), max(activations_over_time))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Activation Value')
    ax.set_title('Activation of Neuron {} in Layer {} Over Epochs'.format(neuron_index, layer_index))
    line, = ax.plot([], [], 'r-')
    
    ani = animation.FuncAnimation(fig, update_line, frames=len(epochs), fargs=(activations_over_time, line), interval=500, blit=True)
    plt.show()
    ani.save('activation_layer{}_neuron{}_animation.gif'.format(layer_index, neuron_index), writer='imagemagick')
if __name__ == '__main__':
    # Initialize Data Loaders
    train_loader = get_data_loader(training=True)
    test_loader = get_data_loader(training=False)
    
    # Build Model
    model = build_model()
    
    # Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Get a fixed input image (first image from test set)
    test_iter = iter(test_loader)
    fixed_input, _ = next(test_iter)
    fixed_input = fixed_input[0].unsqueeze(0)  # Shape [1, 1, 28, 28]
    
    # Train the Model and Record Weights
    epochs = 5
    weight_history = train_model_with_recording(model, train_loader, criterion, optimizer, epochs)
    # Animate Weights
    animate_weights(weight_history, layer_name='Layer 1')
    
    # Re-initialize the Model and Optimizer
    model = build_model()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Train the Model and Record Activations
    activation_history = train_model_with_activation_recording(model, train_loader, fixed_input, criterion, optimizer, epochs)
    # Animate Activations
    animate_activations(activation_history, layer_index=1, neuron_index=0)
