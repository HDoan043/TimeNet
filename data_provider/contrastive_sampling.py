import numpy as np
from utils import new_augmentation
from utils.inject_anomalies_optimal import *

def random_positive_sampler(i, datacustom):
    """
    Lấy một mẫu bình thường bất kỳ trong tập train, né vùng lân cận của i.
    Tối ưu O(1), không tạo list gây chậm Dataloader.
    """
    T_max = datacustom.data_x.shape[0] - datacustom.win_size
    # Nên để range rộng hơn một chút (ví dụ 5-10) để mẫu thực sự "khác biệt"
    neighbor_range = 10 
    
    # Tổng số chỉ số khả thi sau khi trừ đi vùng cấm [i-range, i+range]
    forbidden_span = 2 * neighbor_range + 1
    total_valid = T_max - forbidden_span
    
    if total_valid <= 0: # Trường hợp mảng quá ngắn
        return datacustom.data_x[i : i + datacustom.win_size].copy()
    
    # Chọn ngẫu nhiên trong dải thu hẹp
    idx = np.random.randint(0, total_valid)
    
    # Nếu idx rơi vào hoặc vượt quá vùng cấm bên trái của i, ta đẩy nó sang bên phải
    if idx >= (i - neighbor_range):
        idx += forbidden_span
    
    # Đảm bảo index cuối cùng nằm trong biên an toàn
    final_idx = int(np.clip(idx, 0, T_max))
    
    return datacustom.data_x[final_idx : final_idx + datacustom.win_size].copy()
    
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
    Negative sampling (Injection). Trả về mảng đã scale và label.
    """
    # Lấy window thô từ DataFrame làm đầu vào cho hàm inject
    raw_window = datacustom.df_for_contrastive.iloc[x_index_start: x_index_end].copy()
    num_anomaly = 1 if np.random.rand() < 0.7 else 2
    anomaly_ls = np.random.choice(datacustom.real_anomaly_ls, num_anomaly, replace=False)
    
    # FIX: inject_full trả về Numpy (negative) và int/list (label)
    negative, label = inject_full(raw_window, anomaly_ls, datacustom.position_map, datacustom.name_id_map)
    
    # scaler.transform trực tiếp lên mảng Numpy negative
    negative_scaled = datacustom.scaler.transform(negative).astype('float32')
    
    return negative_scaled, label

def augmentation_positive_sampler(datacustom, x_index_start, x_index_end):
    """
    Bản fix tối ưu: Thực hiện Data Augmentation một cách an toàn và hiệu quả.
    """
    # 1. Thu thập tài nguyên từ datacustom (Truy xuất 1 lần)
    data_x = datacustom.data_x
    win_size = datacustom.win_size
    T_max = data_x.shape[0] - win_size
    aug_config = datacustom.augmentation_config
    
    # Lấy raw_window (chưa chuẩn hóa) để tính biên clipping
    raw_window = datacustom.df_for_contrastive.iloc[x_index_start: x_index_end].values
    
    # Lấy cửa sổ hiện tại (đã chuẩn hóa) và đưa về miền raw để augment
    seq_x_scaled = data_x[x_index_start: x_index_end].copy()
    positive = datacustom.inverse_transform(seq_x_scaled)

    # 2. Xử lý Neighbors cho Random Guided Warp (Tối ưu hóa tìm index)
    offsets = np.random.choice([o for o in range(-24, 25) if o != 0], size=6, replace=False)
    neighbors = []
    
    for offset in offsets:
        n_start = x_index_start + offset
        if 0 <= n_start <= T_max:
            # Lấy neighbor, transform ngược và lưu lại
            n_scaled = data_x[n_start : n_start + win_size]
            neighbors.append(datacustom.inverse_transform(n_scaled))
    
    # Cập nhật config cho hàm warp (nếu danh sách trống thì hàm warp sẽ tự fallback)
    aug_config["random_guided_warp_5g"]["parameter"]["neighbors"] = neighbors

    # 3. Logic chọn số lượng và loại Augmentation (Gọn hơn)
    r = np.random.rand()
    num_aug = 1 if r < 0.7 else (2 if r < 0.95 else 3)
    
    selected_augs = []
    if num_aug == 1:
        keys = list(aug_config.keys())
        probs = [aug_config[k]["ratio"] for k in keys]
        selected_augs = np.random.choice(keys, size=1, p=probs/np.sum(probs))
    else:
        # Chọn 1 phép biến đổi thời gian (Temporal)
        t_keys = ["time_warp", "window_warp_5g", "random_guided_warp_5g", "window_slice"]
        t_probs = np.array([aug_config[k]["ratio"] for k in t_keys])
        selected_augs.append(np.random.choice(t_keys, p=t_probs/t_probs.sum()))
        
        # Chọn các phép biến đổi biên độ (Magnitude) còn lại
        m_keys = ["jitter", "scaling", "magnitude_warp"]
        m_probs = np.array([aug_config[k]["ratio"] for k in m_keys])
        m_selected = np.random.choice(m_keys, size=num_aug - 1, p=m_probs/m_probs.sum(), replace=False)
        selected_augs.extend(m_selected)

    # 4. Thực thi Augmentation
    for aug_name in selected_augs:
        aug_func = getattr(new_augmentation, aug_name)
        params = aug_config[aug_name]["parameter"]
        positive = aug_func(positive, **params)
        
    # 5. Clipping & Re-normalization (Bước quan trọng để dữ liệu không bị hỏng)
    # Dùng raw_window (gốc) để làm mốc giới hạn
    w_min, w_max = raw_window.min(axis=0), raw_window.max(axis=0)
    margin = 0.15 * (w_max - w_min + 1e-8)
    
    positive = np.clip(positive, w_min - margin, w_max + margin)
    
    # Trả về kết quả đã chuẩn hóa lại và ép kiểu float32
    return datacustom.scaler.transform(positive).astype('float32')
