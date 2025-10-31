import torch
import torch.nn as nn
import time
import numpy as np
import torch.optim as optim
def evaluate(model, data_loader, device):
    model.to(device)
    model.eval()
    try:
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_dtype = torch.float32
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.to(dtype=model_dtype)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def measure_latency(model, device, dummy_input):
    """Measures average inference latency."""
    model.to(device)
    model.eval()
    dummy_input = dummy_input.to(device)
    
    # Warm-up runs
    for _ in range(10):
        _ = model(dummy_input)
        
    # Measurement runs
    latencies = []
    with torch.no_grad():
        for _ in range(100):
            start_time = time.time()
            _ = model(dummy_input)
            torch.cuda.synchronize() # Wait for the operation to complete
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000) # milliseconds
            
    return np.mean(latencies)
DEVICE = "cuda"

def train_baseline(model, train_loader, test_loader, epochs=20, lr=0.01):
    """A simple training loop for the baseline model."""
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        val_acc = evaluate(model, test_loader, DEVICE)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.3f}, Val Acc: {val_acc:.2f}%')
        scheduler.step()
    return model