###############################################
#   Tạo các lớp mô hình Softmax, MLP và CNN   #
###############################################

import torch
import torch.nn as nn

# 1. Softmax Classifier
class SoftmaxClassifier(nn.Module):

    def __init__(self):
        super(SoftmaxClassifier, self).__init__()
        self.flatten = nn.Flatten()          # Chuyển đổi ảnh từ [B, 1, 28, 28] sang [B, 784]
        self.linear = nn.Linear(28 * 28, 10) # Lớp Linear với 784 đầu vào và 10 đầu ra (tương ứng với 10 lớp nhãn của MNIST)

    def forward(self, x):
        x = self.flatten(x)                  # Làm phẳng ảnh
        out = self.linear(x)                 # Tính toán đầu ra dạng điểm số (logits) qua lớp tuyến tính
        return out

# 2. MLP Classifier
class MLPClassifier(nn.Module):

    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.flatten = nn.Flatten()          # Chuyển đổi ảnh từ [B, 1, 28, 28] sang [B, 784]
        self.fc1 = nn.Linear(28 * 28, 128)   # Lớp fully connected đầu tiên: 784 đầu vào, 128 đầu ra
        self.fc2 = nn.Linear(128, 64)        # Lớp fully connected thứ hai: 128 đầu vào, 64 đầu ra
        self.fc3 = nn.Linear(64, 10)         # Lớp fully connected cuối cùng: 64 đầu vào, 10 đầu ra (tương ứng với 10 lớp nhãn của MNIST)
        self.relu = nn.ReLU()                # Hàm kích hoạt ReLU: f(x) = max(0, x)

    def forward(self, x):
        x = self.flatten(x)                  # Làm phẳng ảnh
        x = self.relu(self.fc1(x))           # Chuyển qua lớp fully connected đầu tiên và áp dụng ReLU
        x = self.relu(self.fc2(x))           # Chuyển qua lớp fully connected thứ hai và áp dụng ReLU
        out = self.fc3(x)                    # Chuyển qua lớp fully connected cuối cùng để nhận đầu ra dạng điểm số (logits)
        return out

# 3. CNN Classifier
class CNNClassifier(nn.Module):

    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = 3, padding = 1),  # [B, 16, 28, 28] => Lớp tích chập 2D nhận đầu vào 1 kênh (ảnh xám), trích xuất ra 16 kênh đặc trưng, sử dụng kernel 3×3 và giữ nguyên kích thước ảnh
            nn.BatchNorm2d(16),                              # Chuẩn hóa batch cho 16 kênh đặc trưng
            nn.ReLU(),                                       # Hàm kích hoạt ReLU: f(x) = max(0, x)
            nn.MaxPool2d(2, 2),                              # [B, 16, 14, 14] => Lớp gộp cực đại (Max Pooling) với kernel 2x2, giảm kích thước ảnh xuống một nửa 28x28 => 14x14
            
            nn.Conv2d(16, 32, kernel_size = 3, padding = 1), # [B, 32, 14, 14] => Lớp tích chập 2D nhận đầu vào 16 kênh, trích xuất ra 32 kênh đặc trưng, sử dụng kernel 3×3 và giữ nguyên kích thước ảnh
            nn.BatchNorm2d(32),                              # Chuẩn hóa batch cho 32 kênh đặc trưng
            nn.ReLU(),                                       # Hàm kích hoạt ReLU: f(x) = max(0, x)
            nn.MaxPool2d(2, 2)                               # [B, 32, 7, 7] => Lớp gộp cực đại (Max Pooling) với kernel 2×2, giảm kích thước ảnh xuống một nửa 14x14 => 7x7
        )
        self.fc_block = nn.Sequential(                       # Khối phân loại cuối cùng, làm phẳng các đặc trưng không gian từ CNN ([B, 32, 7, 7]) thành vector 1D rồi đưa qua các tầng Linear để phân thành 10 lớp nhãn
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.conv_block(x)
        out = self.fc_block(x)
        return out