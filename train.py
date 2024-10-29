import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import os

def get_input_args():
    parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model as a checkpoint.')
    parser.add_argument('data_dir', type=str, help='Directory with training data')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture (vgg16, alexnet, resnet18)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    return parser.parse_args()

def load_data(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')

    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)

    return trainloader, validloader, train_data

def save_checkpoint(model, save_dir, arch, class_to_idx, epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint_path = os.path.join(save_dir, f'checkpoint_{arch}_{epoch}.pth')
    torch.save({
        'arch': arch,
        'class_to_idx': class_to_idx,
        'state_dict': model.state_dict()
    }, checkpoint_path)

def main():
    args = get_input_args()
    trainloader, validloader, train_data = load_data(args.data_dir)

    model = getattr(models, args.arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    # Adjusting the classifier according to the architecture
    if args.arch.startswith('vgg'):
        input_features = model.classifier[0].in_features
    elif args.arch == 'alexnet':
        input_features = model.classifier[1].in_features
    elif 'resnet' in args.arch:
        input_features = model.fc.in_features
    else:
        raise ValueError("Unsupported architecture")

    model.classifier = nn.Sequential(
        nn.Linear(input_features, args.hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(args.hidden_units, 102), 
        nn.LogSoftmax(dim=1)
    )

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    epochs = args.epochs
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Training loss: {running_loss/len(trainloader):.3f}")

        # Evaluate the model on the validation set
        model.eval()
        validation_loss = 0
        accuracy = 0
        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model(inputs)
                batch_loss = criterion(logps, labels)
                validation_loss += batch_loss.item()

                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Validation loss: {validation_loss/len(validloader):.3f} - "
              f"Accuracy: {accuracy/len(validloader):.3f}")

        # Save checkpoints periodically
        if epoch % 5 == 0 or epoch == epochs - 1:
            save_checkpoint(model, args.save_dir, args.arch, train_data.class_to_idx, epoch)

if __name__ == '__main__':
    main()
