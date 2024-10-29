import argparse
import torch
from torchvision import models, transforms
from PIL import Image
import json

def get_input_args():
    parser = argparse.ArgumentParser(description='Predict flower name from an image along with the probability of that name.')
    parser.add_argument('image_path', type=str, help='Path to image file')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to JSON file mapping categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    model_architecture = checkpoint['arch']
    model = getattr(models, model_architecture)(pretrained=True)
    if 'classifier' in checkpoint:
        model.classifier = checkpoint['classifier']
    elif 'fc' in checkpoint:  
        model.fc = checkpoint['fc']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.eval()  
    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Tensor.
    '''
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image

def predict(image_path, model, top_k, device):
    ''' Predict the class of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    image = image.to(device)
    
    with torch.no_grad():
        output = model.forward(image)
        probs, classes = torch.exp(output).topk(top_k)
    
    return probs.cpu().numpy(), classes.cpu().numpy()

def main():
    args = get_input_args()
    
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint)
    model.to(device)

    probs, classes = predict(args.image_path, model, args.top_k, device)

    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name.get(str(cls), 'Unknown category') for cls in classes[0]]

    print('Probabilities:', probs[0])
    print('Classes:', classes)

if __name__ == '__main__':
    main()
