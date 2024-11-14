# Học Tăng Cường Mô Hình Thế Giới (World Model Reinforcement Learning)

Dự án này triển khai một hệ thống học tăng cường dựa trên mô hình thế giới sử dụng PyTorch. Nó kết hợp mô hình không gian trạng thái tuần hoàn (RSSM), tìm kiếm quỹ đạo và lập kế hoạch dựa trên mô hình để học tập và ra quyết định hiệu quả.

## Tính năng

- Mô hình Không gian Trạng thái Tuần hoàn (RSSM) để mô hình hóa động lực học môi trường
- Tìm kiếm quỹ đạo với lấy mẫu Gumbel-Top-K
- Kiến trúc Actor-Critic cho việc học chính sách
- Bộ đệm phát lại kinh nghiệm cho việc học off-policy
- Hỗ trợ cả thiết bị CPU và CUDA

## Yêu cầu

- Python 3.7+
- PyTorch >= 2.0.0
- Gymnasium >= 0.29.0
- NumPy >= 1.24.3
- Matplotlib >= 3.7.1
- TorchVision >= 0.15.0
- TorchInfo >= 1.8.0
- tqdm >= 4.65.0
- pytest >= 7.3.1

## Cài đặt

1. Sao chép kho lưu trữ này:
```bash
   git clone https://github.com/RandomMike1280/FishZero.git
   cd FishZero
```

2. Cài đặt các thư viện phụ thuộc:
```bash
   pip install -r requirements.txt
```

## Huấn luyện

Để huấn luyện mô hình, hãy chạy:
```bash
    python train.py
```

CẦN LÀM:
Sửa lỗi mất mát động lực học (dynamics loss)
Thêm mất mát chính sách (policy loss) và mất mát giá trị (value loss)
