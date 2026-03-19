import torch

def si_snr_loss(estimate, target, eps=1e-8):
    """
    Tính toán hàm mất mát SI-SNR (Scale-Invariant Signal-to-Noise Ratio).
    Loss trả về số âm vì mục tiêu của tối ưu hóa là TỐI THIỂU HÓA giá trị mất mát 
    (SI-SNR càng lớn càng tốt, nên ta minimize -SI_SNR).
    
    Args:
        estimate (torch.Tensor): Waveform được mô hình dự đoán. Shape: (batch_size, num_samples)
        target (torch.Tensor): Waveform sạch (ground truth). Shape: (batch_size, num_samples)
        eps (float): Hằng số nhỏ để tránh lỗi chia cho 0.
        
    Returns:
        torch.Tensor: Giá trị loss vô hướng (scalar).
    """
    # 1. Trừ đi giá trị trung bình (zero-mean)
    estimate = estimate - torch.mean(estimate, dim=1, keepdim=True)
    target = target - torch.mean(target, dim=1, keepdim=True)
    
    # 2. Tính hệ số tỉ lệ (scaling factor) alpha
    # alpha = <estimate, target> / ||target||^2
    dot_product = torch.sum(estimate * target, dim=1, keepdim=True)
    target_energy = torch.sum(target ** 2, dim=1, keepdim=True)
    alpha = (dot_product + eps) / (target_energy + eps)
    
    # 3. Tính tín hiệu mục tiêu (s_target) và nhiễu (e_noise)
    s_target = alpha * target
    e_noise = estimate - s_target
    
    # 4. Tính tỉ số tín hiệu trên nhiễu SI-SNR
    s_target_energy = torch.sum(s_target ** 2, dim=1)
    e_noise_energy = torch.sum(e_noise ** 2, dim=1)
    
    si_snr = 10 * torch.log10((s_target_energy + eps) / (e_noise_energy + eps))
    
    # 5. Lấy trung bình toàn batch và đảo dấu
    return -torch.mean(si_snr)