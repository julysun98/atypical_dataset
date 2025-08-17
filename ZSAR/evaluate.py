import torch
from tqdm import tqdm
from config import Config
from data.dataset import get_dataloader
from models.zsar_model import ZSARModel

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    class_correct = {}
    class_total = {}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            frames = batch['frames'].to(device)
            class_names = batch['class_name']
            labels = batch['class_idx'].to(device)
            
            # Forward pass
            similarity = model(frames, class_names)
            
            # Calculate accuracy
            _, predicted = similarity.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Per-class accuracy
            for pred, label, class_name in zip(predicted, labels, class_names):
                if class_name not in class_correct:
                    class_correct[class_name] = 0
                    class_total[class_name] = 0
                
                class_total[class_name] += 1
                if pred == label:
                    class_correct[class_name] += 1
    
    # Calculate overall accuracy
    accuracy = 100. * correct / total
    
    # Calculate per-class accuracy
    class_accuracies = {
        class_name: 100. * class_correct[class_name] / class_total[class_name]
        for class_name in class_correct.keys()
    }
    
    return accuracy, class_accuracies

def main():
    # Set device
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    
    # Create test dataloader
    test_loader = get_dataloader(Config.UCF101_PATH, split='test')
    
    # Initialize model
    model = ZSARModel().to(device)
    
    # Load best model
    checkpoint = torch.load(f"{Config.CHECKPOINT_DIR}/best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    accuracy, class_accuracies = evaluate(model, test_loader, device)
    
    print(f"\nOverall Test Accuracy: {accuracy:.2f}%")
    print("\nPer-class Test Accuracy:")
    for class_name, acc in sorted(class_accuracies.items()):
        print(f"{class_name}: {acc:.2f}%")

if __name__ == '__main__':
    main()
