import numpy as np
from utils import new_augmentation
from utils.inject_anomalies import *

# positive sampler
def stochastic_positive_sampler(i, datacustom, jitter_range=(-2, 2)):
    """
    Lấy mẫu dương theo phương pháp Stochastic (Xác suất) kết hợp Jitter.
    """
    lags = datacustom.lags 
    scores = datacustom.scores
    data = datacustom.data_x 
    win_size = datacustom.args.win_size
    T = data.shape[0]
    
    # Lọc các lag không bị tràn mảng (Out of bounds)
    valid_indices = [idx for idx, lag in enumerate(lags) if i + lag <= T - win_size]
    
    if not valid_indices:
        # Fallback: Nếu không có lag nào khớp, lấy ngay cửa sổ tiếp theo kèm jitter
        # để tránh làm gãy luồng training.
        shift = win_size + np.random.randint(jitter_range[0], jitter_range[1] + 1)
        pos_idx = min(i + shift, T - win_size)
        return data[pos_idx : pos_idx + win_size]

    # VẤN ĐỀ 3 FIX: Sampling dựa trên trọng số điểm tương đồng
    p_lags = lags[valid_indices]
    p_scores = scores[valid_indices]
    
    # Chuyển scores thành xác suất (Softmax hoặc đơn giản là Normalize)
    # Dùng Softmax nếu bạn muốn nhấn mạnh vào các lag có score cực cao
    probs = p_scores / (np.sum(p_scores) + 1e-8)
    
    chosen_lag = np.random.choice(p_lags, p=probs)
    
    # Thêm Jitter để model không "học thuộc lòng" vị trí chính xác của chu kỳ
    jitter = np.random.randint(jitter_range[0], jitter_range[1] + 1)
    final_pos_idx = max(0, min(i + chosen_lag + jitter, T - win_size))
    
    return data[final_pos_idx : final_pos_idx + win_size]

def augmentation_positive_sampler(datacustom, x_index_start, x_index_end):
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
    window_min = seq_x.min(axis=0)
    window_max = seq_x.max(axis=0)
    
    margin = 0.15 * (window_max - window_min)
    positive = np.clip( positive, window_min - margin, window_max + margin)
    positive = scaler.transform(positive)

    return positive

def neighbor_positive_sampler(datacustom, x_index_start, x_index_end):
    data_x = datacustom.data_x
    win_size = datacustom.win_size
    
    neighbor_distance = 1
    neighbor_candidates = [data_x[j: j + win_size].copy() for j in range(max(0,x_index_start - neighbor_distance), 
                                                                                min(len(data_x),x_index_end + neighbor_distance)) if j !=x_index_start ]
    positive = np.random.choice(neighbor_candidates)
    return positive

# negative sampler
def negative_sampler(datacustom, x_index_start, x_index_end):
    raw_window = datacustom.df_for_contrastive.iloc[x_index_start: x_index_end].copy()
    num_anomaly = np.random.choice([1,2], p=[0.7,0.3])
    anomaly_ls = np.random.choice(datacustom.real_anomaly_ls, num_anomaly, replace = False)
    negative, label = inject_full(raw_window, anomaly_ls, datacustom.position_map, datacustom.name_id_map)
    negative = datacustom.scaler.transform(negative)
