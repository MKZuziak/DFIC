import torch
import timm

def train_epoch(
    train_loader,
    model,
    optimizer,
    loss_func
    ):    
    total_training_images = 0
    total_training_loss = 0
    total_training_correct = 0
    
    # Enumerating the data loader
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs.to('cuda:0')
        labels.to('cuda:0')
        # Removing previous gradient
        optimizer.zero_grad()
        # Using model inference
        outputs = model(inputs)
        # Calculating the loss
        loss = loss_func(outputs, labels)
        # Computing the gradients
        loss.backward()
        # Adjusting networking weights
        optimizer.step()
                
        # Training metrics
        total_training_images += labels.size(0)
        _, predicted = torch.max(outputs.detach(), 1)
        total_training_correct += (predicted == labels).sum().items()
        total_training_loss += loss.item()
    
    return {"total_training_images":total_training_images,
            "total_training_loss":total_training_loss,
            "total_training_correct":total_training_correct
            }