###################################
#   Tải ảnh và chuẩn bị dữ liệu   #
###################################

# <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1032)>
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from torchvision import datasets, transforms # datasets: chứa các tập dữ liệu phổ biến (MNIST, CIFAR-10, ...), transformers: giúp chuyển đổi dữ liệu (ví dụ: từ ảnh sang tensor, chuẩn hóa, ...)
from torch.utils.data import DataLoader      # DataLoader: giúp chia dữ liệu thành từng nhóm nhỏ (batch) và xáo trộn dữ liệu khi huấn luyện

def get_data_loaders(batch_size = 64):
    
    # Chuyển đổi ảnh sang Tensor và chuẩn hóa
    transform = transforms.Compose([
        transforms.ToTensor(),               # Chuyển đổi ảnh từ PIL Image sang Tensor, chuẩn hoá giá trị pixel về khoảng [0, 1]
        transforms.Normalize((0.5,), (0.5,)) # Chuẩn hóa về khoảng [-1, 1]
    ])

    # Tải tập huấn luyện và tập kiểm tra
    train_dataset = datasets.MNIST(root = './data', train = True, download = True, transform = transform)
    test_dataset = datasets.MNIST(root = './data', train = False, download = True, transform = transform)

    # Tạo đối tượng cung cấp dữ liệu cho tập huấn luyện (cần xáo trộn) và tập kiểm tra (không cần xáo trộn)
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

    return train_loader, test_loader