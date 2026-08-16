import sys
import copy
from pathlib import Path
import torch
import torch.nn as nn

# Thêm module ycbcr2rgb từ thư mục utils của DCVC-RT
from src.utils.transforms import ycbcr2rgb


def dcvc_to_yolo_rgb(x_dcvc: torch.Tensor) -> torch.Tensor:
    """
    Chuyển đổi Tensor từ không gian màu DCVC-RT (centered YCbCr [-0.5, 0.5])
    sang không gian màu chuẩn YOLOv5 (RGB [0.0, 1.0]).
    
    Quy trình:
        x_dcvc + 0.5  -> YCbCr trong dải [0, 1]
        ycbcr2rgb     -> RGB trong dải [0, 1]
        clamp(0, 1)   -> Khống chế nhiễu sai số điểm ảnh
    """
    x_ycbcr = x_dcvc + 0.5
    x_rgb = ycbcr2rgb(x_ycbcr)
    return torch.clamp(x_rgb, 0.0, 1.0)


class YOLOv5FrontEnd(nn.Module):
    """
    Front-end trích xuất đặc trưng gồm N layers đầu tiên của YOLOv5.
    
    Hỗ trợ chuyển đổi tự động từ YCbCr sang RGB nếu convert_dcvc_input=True.
    """
    def __init__(self, layers: nn.Sequential, convert_dcvc_input: bool = True):
        super().__init__()
        self.layers = layers
        self.convert_dcvc_input = convert_dcvc_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.convert_dcvc_input:
            x = dcvc_to_yolo_rgb(x)
        return self.layers(x)


def build_yolov5_frontends(
    weights_path: str,
    yolov5_root: str = "./third_party/yolov5",
    device: torch.device = torch.device("cpu"),
    num_layers: int = 5,
):
    """
    Khởi tạo 2 mô hình Front-end YOLOv5:
      1. Teacher Front-End: Khóa weights (frozen), eval mode, không tính Gradient.
      2. Student Front-End: Cho phép huấn luyện (trainable), train mode, tính Gradient.
    """
    yolov5_path = Path(yolov5_root).resolve()
    if str(yolov5_path) not in sys.path:
        sys.path.insert(0, str(yolov5_path))

    # Load weights bằng hàm torch.load từ ultralytics
    checkpoint = torch.load(
    	weights_path,
    	map_location=device,
    	weights_only=False,
    )
    full_model = checkpoint.get("model", checkpoint)
    if hasattr(full_model, "float"):
        full_model = full_model.float()
    
    full_model.eval()

    # Lấy danh sách các lớp trong mô hình YOLOv5
    if hasattr(full_model, "model") and hasattr(full_model.model, "children"):
        yolo_layers = list(full_model.model.children())
    else:
        yolo_layers = list(full_model.children())

    # Kiểm tra tính toàn vẹn của kết nối luồng sequential (f == -1)
    for idx in range(num_layers):
        layer = yolo_layers[idx]
        from_index = getattr(layer, "f", -1)

        if isinstance(from_index, int):
            valid = (from_index == -1)
        elif isinstance(from_index, (list, tuple)):
            valid = (len(from_index) == 1 and from_index[0] == -1)
        else:
            valid = False

        if not valid:
            raise RuntimeError(
                f"Selected YOLO layer {idx} (f={from_index}) requires multiple inputs "
                "and cannot be wrapped into a simple nn.Sequential."
            )

    # Đóng gói 5 layers đầu tiên thành nn.Sequential
    base_frontend = nn.Sequential(
        *[copy.deepcopy(yolo_layers[idx]) for idx in range(num_layers)]
    )

    # 1. Khởi tạo Teacher Front-End (Frozen)
    teacher_frontend = YOLOv5FrontEnd(
        layers=copy.deepcopy(base_frontend),
        convert_dcvc_input=True,
    ).to(device)
    
    teacher_frontend.eval()
    for param in teacher_frontend.parameters():
        param.requires_grad = False

    # 2. Khởi tạo Student Front-End (Trainable)
    student_frontend = YOLOv5FrontEnd(
        layers=copy.deepcopy(base_frontend),
        convert_dcvc_input=True,
    ).to(device)
    
    student_frontend.train()
    for param in student_frontend.parameters():
        param.requires_grad = True

    # Giải phóng bộ nhớ của full model
    del full_model
    del base_frontend

    return teacher_frontend, student_frontend