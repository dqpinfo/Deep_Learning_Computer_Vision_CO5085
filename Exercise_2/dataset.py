###################################
#   Tải ảnh và chuẩn bị dữ liệu   #
###################################

# <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1032)>
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_data_loaders(batch_size = 64, subset_train = 5000, subset_test = 1000):
    
    # Chuyển đổi dữ liệu ảnh sang tensor
    transform = transforms.Compose([transforms.ToTensor()])

    # Tải tập huấn luyện và tập kiểm tra
    train_dataset = datasets.MNIST(root = './data', train = True, download = True, transform = transform)
    test_dataset = datasets.MNIST(root = './data', train = False, download = True, transform = transform)

    # Chỉ lấy subset nếu tập dữ liệu quá lớn
    if subset_train and subset_train < len(train_dataset):
        train_dataset = torch.utils.data.Subset(train_dataset, range(subset_train))
    if subset_test and subset_test < len(test_dataset):
        test_dataset = torch.utils.data.Subset(test_dataset, range(subset_test))

    # Tạo đối tượng cung cấp dữ liệu cho tập huấn luyện (cần xáo trộn) và tập kiểm tra (không cần xáo trộn)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    return train_loader, test_loader