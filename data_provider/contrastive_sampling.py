import numpy as np
from utils import new_augmentation
from utils.inject_anomalies import *

# positive sampler
def stochastic_positive_sampler(i, datacustom, jitter_range=(-2, 2)):
    """
    Bản tối ưu: Vectorized Bidirectional Sampling.
    Tìm tất cả các vị trí chu kỳ trong cả quá khứ và tương lai cực nhanh.
    """
    lags = datacustom.lags
    scores = datacustom.scores
    data = datacustom.data_x
    win_size = datacustom.win_size
    T_max = data.shape[0] - win_size
    
    # 1. Tính toán song song tất cả các vị trí ứng viên (Tiến & Lùi)
    fwd_indices = i + lags
    bwd_indices = i - lags
    
    # 2. Tạo Mask lọc nhanh các chỉ số hợp lệ bằng Numpy (Vectorized)
    mask_fwd = (fwd_indices >= 0) & (fwd_indices <= T_max)
    mask_bwd = (bwd_indices >= 0) & (bwd_indices <= T_max)
    
    # 3. Gộp các ứng viên hợp lệ
    valid_indices = np.concatenate([fwd_indices[mask_fwd], bwd_indices[mask_bwd]])
    valid_scores = np.concatenate([scores[mask_fwd], scores[mask_bwd]])
    
    # 4. Xử lý trường hợp không có lag nào khớp (Edge cases)
    if valid_indices.size == 0:
        # Fallback: Lấy neighbor ngẫu nhiên trong tầm win_size
        shift = np.random.randint(-win_size, win_size + 1)
        pos_idx = int(np.clip(i + shift, 0, T_max))
        return data[pos_idx : pos_idx + win_size].copy()

    # 5. Lấy mẫu theo xác suất (Weighted Random Choice)
    probs = valid_scores / (valid_scores.sum() + 1e-8)
    chosen_idx = np.random.choice(valid_indices, p=probs)
    
    # 6. Thêm Jitter để tăng tính Robust
    jitter = np.random.randint(jitter_range[0], jitter_range[1] + 1)
    final_idx = int(np.clip(chosen_idx + jitter, 0, T_max))
    
    return data[final_idx : final_idx + win_size].copy()

def neighbor_positive_sampler(datacustom, x_index_start):
    """
    Lấy cửa sổ lân cận (Adjacent). Tối ưu bằng cách tránh slice mảng trước khi chọn.
    """
    T_max = datacustom.data_x.shape[0] - datacustom.win_size
    neighbor_dist = 3  # Khoảng cách lân cận
    
    # Chọn index trước, slice sau
    possible_offsets = np.arange(-neighbor_dist, neighbor_dist + 1)
    possible_offsets = possible_offsets[possible_offsets != 0]
    
    offset = np.random.choice(possible_offsets)
    final_idx = int(np.clip(x_index_start + offset, 0, T_max))
    
    return datacustom.data_x[final_idx : final_idx + datacustom.win_size].copy()

def negative_sampler(datacustom, x_index_start, x_index_end):
    """
    Negative sampling (Injection). Tối ưu bằng cách ép kiểu mảng ngay lập tức.
    """
    raw_window = datacustom.df_for_contrastive.iloc[x_index_start: x_index_end].copy()
    num_anomaly = 1 if np.random.rand() < 0.7 else 2
    
    anomaly_ls = np.random.choice(datacustom.real_anomaly_ls, num_anomaly, replace=False)
    
    negative, label = inject_full(raw_window, anomaly_ls, datacustom.position_map, datacustom.name_id_map)
    
    # Trả về mảng numpy float32 để tiết kiệm bộ nhớ và tăng tốc độ truyền lên GPU
    return datacustom.scaler.transform(negative.values).astype('float32'), label

# augmentation_positive_sampler giữ nguyên logic cũ nhưng thêm .astype('float32') ở đầu ra

def augmentation_positive_sampler(datacustom, x_index_start, x_index_end):
    raw_window = datacustom.df_for_contrastive.iloc[x_index_start: x_index_end].values
    
    data_x = datacustom.data_x
    augmentation_config = datacustom.augmentation_config
    inverse_transform = datacustom.inverse_transform
    scaler = datacustom.scaler
    
    
    seq_x = data_x[x_index_start: x_index_end]
    positive = seq_x.copy()

    neighbors = []

    offsets = np.random.choice(
        [o for o in range(-24, 25) if o != 0],
        size=6,
        replace=False
    )
    
    for offset in offsets:
        start = x_index_start + offset
        end = x_index_end + offset
    
        if start >= 0 and end <= len(data_x):
            neighbors.append( data_x[start:end].copy())
    
    positive = inverse_transform(positive)
    neighbors = [inverse_transform(neighbor) for neighbor in neighbors]

    augmentation_config["random_guided_warp_5g"]["parameter"]["neighbors"] = neighbors

    r = np.random.rand()
    aug_ls = []
    if r<0.7: num_aug = 1
    elif r<0.95: num_aug = 2
    else: num_aug = 3

    if num_aug == 1:
        aug_ls = list(augmentation_config.keys())
        p = [augmentation_config[k]["ratio"] for k in aug_ls]
        aug_ls = np.random.choice(aug_ls, num_aug, p=p)
    else:
        temporal_augs = ["time_warp", "window_warp_5g", "random_guided_warp_5g", "window_slice"]
        temporal_p = np.array([augmentation_config[k]["ratio"] for k in temporal_augs])
        temporal_p = temporal_p / temporal_p.sum()
        magnitude_augs = ["jitter", "scaling", "magnitude_warp"]
        magnitude_p = np.array([augmentation_config[k]["ratio"] for k in magnitude_augs])
        magnitude_p = magnitude_p / magnitude_p.sum()
        temporal_aug = np.random.choice(temporal_augs, p=temporal_p)
        aug_ls.append(temporal_aug)
        aug_ls.extend(np.random.choice(magnitude_augs, num_aug - 1, p=magnitude_p))

    for aug in aug_ls:
        aug_func = getattr(new_augmentation, aug)
        parameters = augmentation_config[aug]["parameter"]
        positive = aug_func(positive, **parameters)
        
    window_min, window_max = raw_window.min(axis=0), raw_window.max(axis=0)
    margin = 0.15 * (window_max - window_min)
    positive = np.clip(positive, window_min - margin, window_max + margin)
    positive = scaler.transform(positive)

    return positive
