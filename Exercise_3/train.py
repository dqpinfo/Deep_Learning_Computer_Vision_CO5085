import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_data_loaders
from models import ImageRNNClassifier

# Tạo thư mục "output" nếu chưa tồn tại
SAVE_DIR = "output"
os.makedirs(SAVE_DIR, exist_ok=True)

def train_model(rnn_type='LSTM', epochs=5, lr=0.001, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- Đang huấn luyện mô hình: {rnn_type} trên thiết bị: {device} ---")
    
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)
    
    model = ImageRNNClassifier(rnn_type=rnn_type, input_size=28, hidden_size=128, num_layers=1, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 1. Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 2. Backward pass và tối ưu hóa thủ công
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc*100:.2f}%")
        
    # Lưu checkpoint vào thư mục "output"
    weight_filename = os.path.join(SAVE_DIR, f"{rnn_type.lower()}_model.pth")
    torch.save(model.state_dict(), weight_filename)
    print(f"[+] Đã lưu trọng số mô hình tại: {weight_filename}")
    
    return model, test_loader

if __name__ == '__main__':
    # Huấn luyện LSTM và lưu vào output/lstm_model.pth
    train_model(rnn_type='LSTM', epochs=5)
    
    # Huấn luyện GRU và lưu vào output/gru_model.pth
    train_model(rnn_type='GRU', epochs=5)