import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureMSELoss(nn.Module):
    """
    Hàm tính Loss tương đồng đặc trưng (Feature Matching Loss):
    
    D_feature = MSE(r_hat_t, r_t)
    
    Trong đó:
        r_t     = 特征 Target trích xuất từ Teacher (Frozen)
        r_hat_t = 特征 Prediction trích xuất từ Student (Trainable)
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction mode: {reduction}")
        self.reduction = reduction

    def forward(
        self,
        student_feature: torch.Tensor,
        teacher_feature: torch.Tensor,
    ) -> torch.Tensor:

        if student_feature.shape != teacher_feature.shape:
            raise ValueError(
                f"Feature shape mismatch: Student {tuple(student_feature.shape)} vs "
                f"Teacher {tuple(teacher_feature.shape)}"
            )

        # Ngắt Gradient khỏi Teacher để đảm bảo an toàn tuyệt đối
        teacher_feature = teacher_feature.detach()

        return F.mse_loss(student_feature, teacher_feature, reduction=self.reduction)