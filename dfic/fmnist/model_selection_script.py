import os

import timm
import pandas as pd
import torch
import datasets
from torchvision import transforms

DEVICE = 'cuda:0'

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
        inputs = data['image']
        labels = data['label']
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
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
        total_training_correct += (predicted == labels).sum().item()
        total_training_loss += loss.item()
    
    return {"total_training_images":total_training_images,
            "total_training_loss":total_training_loss,
            "total_training_correct":total_training_correct
            }

def validation_epoch(
    validation_loader,
    model,
    loss_func
    ):    
    total_validation_images = 0
    total_validation_loss = 0
    total_validation_correct = 0
    model.eval()
    
    # Enumerating the data loader
    with torch.no_grad():
        for i, data in enumerate(validation_loader):
            inputs = data['image']
            labels = data['label']
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            # Using model inference
            outputs = model(inputs)

            # Testing metrics
            total_validation_loss += loss_func(outputs, labels).item()
            total_validation_images += labels.size(0)
            _, predicted = torch.max(outputs.detach(), 1)
            total_validation_correct += (predicted == labels).sum().item()
        
    return {"total_validation_images":total_validation_images,
            "total_validation_loss":total_validation_loss,
            "total_validation_correct":total_validation_correct
            }


def model_evaluation():
    # MODEL SELECTION
    models = ['resnet18', 'resnet26', 'resnet34', 'resnet50', 'mobilenetv3_small_050', 'mobilenetv3_small_075', 'mobilenetv3_small_100', 'convmixer_768_32']
    
    # DATASET LOADING
    train_set = datasets.load_dataset('zalando-datasets/fashion_mnist', split='train')
    out_set = datasets.load_dataset('zalando-datasets/fashion_mnist', split='test')
    out_set = out_set.train_test_split(test_size=0.5)
    test_set = out_set['test']
    validation_set = out_set['train']
    convert_tensor = transforms.ToTensor()
    train_set = train_set.map(lambda sample: {'image': convert_tensor(sample['image'])})
    train_set.set_format('pt', columns=['image'], output_all_columns=True)
    validation_set = validation_set.map(lambda sample: {'image': convert_tensor(sample['image'])})
    validation_set.set_format('pt', columns=['image'], output_all_columns=True)
    test_set = test_set.map(lambda sample: {'image': convert_tensor(sample['image'])})
    test_set.set_format('pt', columns=['image'], output_all_columns=True)
    training_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)
    
    # RESULTS ARCHIVES
    final_test = pd.DataFrame(index=models, columns=['test loss', 'test accuracy'])
    evaluation_frame = {
    model:
        {
            "training_loss": [],
            "validation_loss": [],
            "training_acc": [],
            'validation_acc': [],
        } for model in models
    }
    
    # TRAINING
    num_classes = 10
    in_chans = 1
    learning_rate = 0.001
    
    for model_name in models:
        model = timm.create_model(model_name, num_classes=num_classes, in_chans=in_chans, pretrained=False)
        model.to(DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        loss_func = torch.nn.CrossEntropyLoss()
        print(f"Training model {model_name}")
        
        for epoch in range(100):
            ## TRAIN
            train_results = train_epoch(
                train_loader=training_loader,
                model=model,
                optimizer=optimizer,
                loss_func=loss_func
            )
            evaluation_frame[model_name]['training_loss'].append(train_results['total_training_loss'])
            evaluation_frame[model_name]['training_acc'].append(train_results['total_training_correct'] / train_results['total_training_images'])
            print(f"Epoch {epoch} training loss: {train_results['total_training_loss']:.3f}")
            print(f"Epoch {epoch} training accuracy: {train_results['total_training_correct'] / train_results['total_training_images']:.3f}")
            
            ## VALIDATE
            validation_results = validation_epoch(
                validation_loader=validation_loader,
                model=model,
                loss_func=loss_func
            )
            evaluation_frame[model_name]['validation_loss'].append(validation_results['total_validation_loss'])
            evaluation_frame[model_name]['validation_acc'].append(validation_results['total_validation_correct'] / validation_results['total_validation_images'])
            print(f"Epoch {epoch} validation loss: {validation_results['total_validation_loss']:.3f}")
            print(f"Epoch {epoch} validation accuracy: {validation_results['total_validation_correct'] / validation_results['total_validation_images']:.3f}")

        ## TEST
        test_results = validation_epoch(
            validation_loader=test_loader,
            model=model,
            loss_func=loss_func
        )
        print(f"Final test loss: {test_results['total_validation_loss']:.3f}")
        print(f"Final test accuracy: {test_results['total_validation_correct'] / test_results['total_validation_images']:.3f}")

        final_test.loc[model_name, 'test loss'] = test_results['total_validation_loss']
        final_test.loc[model_name, 'test accuracy'] = test_results['total_validation_correct'] / test_results['total_validation_images']
    
    for frame, results in evaluation_frame.items():
        with open(os.path.join(os.getcwd(), 'model_selection', f'{frame}.csv'), 'w') as file:
            file.write('epoch,training_loss,validation_loss,training_acc,validation_acc\n')
            for epoch in range(len(results['training_loss'])):
                file.write(f"{epoch},{results['training_loss'][epoch]},{results['validation_loss'][epoch]},{results['training_acc'][epoch]},{results['validation_acc'][epoch]}\n")

    with open(os.path.join(os.getcwd(), 'model_selection', "final_test_results.csv"), 'w') as file:
        file.write(final_test.to_csv())
        

if __name__ == "__main__":
    model_evaluation()