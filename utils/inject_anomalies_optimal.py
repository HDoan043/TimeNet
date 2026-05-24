import numpy as np
import pandas as pd

def apply_noise(x, scale=0.05):
    # x có thể là một số hoặc 1 numpy array
    return x * (1 + np.random.normal(0, scale, size=np.shape(x)))

def smooth_transition(start, end, length):
    return np.linspace(start, end, length)

def lag(df_clean, df, idx_array, col, col_multiplier_map):
    lag_val = np.random.randint(5, 10)
    # Lấy dữ liệu bị trễ bằng cách trừ index (vectorized)
    shifted_idx = idx_array - lag_val
    # Lọc những index hợp lệ (không bị âm và có trong df_clean)
    valid_mask = (shifted_idx >= 0) & np.isin(shifted_idx, df_clean.index)
    
    if valid_mask.any():
        v_idx = idx_array[valid_mask]
        v_shift = shifted_idx[valid_mask]
        base = df_clean.loc[v_shift, col].values * col_multiplier_map[col]
        df.loc[v_idx, col] = np.maximum(base, 0)
        
def trend(df_clean, df, idx_array, col, col_multiplier_map):
    # Tính trend bằng shift pandas cực nhanh
    base = df_clean.loc[idx_array, col].values * col_multiplier_map[col]
    
    # Để tránh lỗi out-of-bounds, dùng .shift() trên df_clean rồi loc ra
    val_minus_1 = df_clean[col].shift(1).loc[idx_array].fillna(df_clean.loc[idx_array, col])
    val_minus_5 = df_clean[col].shift(5).loc[idx_array].fillna(df_clean.loc[idx_array, col])
    
    trend_val = val_minus_1.values - val_minus_5.values
    df.loc[idx_array, col] = np.maximum(base - 0.7 * trend_val, 0)
    
def correlation(df_clean, df, idx_array, col, col_multiplier_map):
    cols = list(col_multiplier_map.keys())   
    if len(cols) <= 1:
        noise(df_clean, df, idx_array, col, col_multiplier_map) # Fallback
        return

    other = np.random.choice([c for c in cols if c != col])
    m1 = col_multiplier_map[col]
    m2 = col_multiplier_map.get(other, 1.0)
    
    base_clean = df_clean.loc[idx_array, col].values
    other_clean = df_clean.loc[idx_array, other].values
    
    base = base_clean * m1
    other_val = other_clean * m2

    delta_base = base - base_clean
    delta_other = other_val - other_clean

    sign = (m1 - 1) * (m2 - 1)
    
    # Tính gộp 3 trường hợp bằng np.where (Vectorized If-Else)
    new_vals = np.where(sign > 0, 
                        base_clean + 0.7 * delta_base + 0.3 * delta_other,
               np.where(sign < 0, 
                        base_clean + 0.7 * delta_base - 0.3 * delta_other,
                        base))
                        
    df.loc[idx_array, col] = np.maximum(new_vals, 0)

def noise(df_clean, df, idx_array, col, col_multiplier_map):
    base = df_clean.loc[idx_array, col].values * col_multiplier_map[col]
    df.loc[idx_array, col] = np.maximum(apply_noise(base, 0.03), 0)

# =========================
# Build timeline effects (Giữ nguyên)
# =========================
def build_timeline_effects(anomaly, position_map, name_id_map):
    timeline_effects = {}
    for counter in anomaly["counter"]:
        module, scenario, timeline = counter["module"], counter["scenario"], int(counter["timeline"])
        multiplier = counter["impact_multiplier"]
        try: names = position_map[module][scenario][str(timeline)]
        except: continue
        if not names: continue
        for name in names:
            if name not in name_id_map: continue
            cid = str(name_id_map[name])
            timeline_effects.setdefault(timeline, {})
            timeline_effects[timeline][cid] = multiplier

    return {k: v.copy() for k, v in timeline_effects.items() if v}

# =========================
# Core injection (ADVANCED & VECTORIZED)
# =========================
def apply_causal_injection(df_clean, df, overlap_indexes, anomaly_start, anomaly_length, timeline_effects):
    if not timeline_effects or len(overlap_indexes) == 0:
        return df

    sorted_timelines = sorted(list(timeline_effects.keys()))
    num_timelines = len(sorted_timelines)
    timeline_to_indexes = {t: [] for t in sorted_timelines}
    
    # Vector hóa việc ánh xạ tiến trình (progress) sang timeline
    idx_arr = np.array(overlap_indexes)
    progress = (idx_arr - anomaly_start) / max(1, anomaly_length)
    timeline_indices = np.clip((progress * num_timelines).astype(int), 0, num_timelines - 1)
    
    for i, t_idx in enumerate(timeline_indices):
        timeline_to_indexes[sorted_timelines[t_idx]].append(idx_arr[i])
        
    for timeline, step_idx in timeline_to_indexes.items():
        if not step_idx: continue
        
        # Bỏ qua các index < 30 bằng mask NumPy
        step_idx_arr = np.array(step_idx)
        step_idx_arr = step_idx_arr[step_idx_arr >= 30]
        if len(step_idx_arr) == 0: continue
            
        cols = list(timeline_effects[timeline].keys())
        col_multiplier_map = timeline_effects[timeline]
        
        # Sinh mảng ngẫu nhiên cho toàn bộ step_idx cùng lúc
        distort_type = np.random.rand()
        r_array = distort_type + np.random.normal(0, 0.05, size=len(step_idx_arr))

        # Phân chia index vào 4 rổ (buckets) dựa trên r_array
        mask_lag = r_array < 0.15
        mask_trend = (r_array >= 0.15) & (r_array < 0.3)
        mask_corr = (r_array >= 0.3) & (r_array < 0.5)
        mask_noise = r_array >= 0.5

        for col in cols:
            if col not in df.columns: continue
            
            # Thay vì loop i, ta quăng nguyên mảng index vào hàm distort tương ứng
            if mask_lag.any(): lag(df_clean, df, step_idx_arr[mask_lag], col, col_multiplier_map)
            if mask_trend.any(): trend(df_clean, df, step_idx_arr[mask_trend], col, col_multiplier_map)
            if mask_corr.any(): correlation(df_clean, df, step_idx_arr[mask_corr], col, col_multiplier_map)
            if mask_noise.any(): noise(df_clean, df, step_idx_arr[mask_noise], col, col_multiplier_map)

    return df

# =========================
# LONG-TERM DRIFT & SEASONAL DISTORTION
# =========================
def apply_long_term_drift(df_clean, df, indexes, exclude_cols=None):
    cols = [c for c in df.columns if c not in ["date", "label", "Unnamed: 0"]]
    if exclude_cols: cols = [c for c in cols if c not in exclude_cols]
    if not cols: return df

    np.random.seed(hash(str(indexes[0])) % 2**32)
    selected_cols = np.random.choice(cols, max(1, len(cols)//10), replace=False)

    for col in selected_cols:
        drift_strength = np.random.uniform(0.01, 0.05)
        values = df.loc[indexes, col].values
        trend_arr = smooth_transition(0, drift_strength * np.mean(values), len(values))
        df.loc[indexes, col] = values + trend_arr
    return df

def apply_seasonal_shift(df_clean, df, indexes, exclude_cols=None):
    cols = [c for c in df.columns if c not in ["date", "label", "Unnamed: 0"]]
    if exclude_cols: cols = [c for c in cols if c not in exclude_cols]
    if not cols: return df

    np.random.seed(hash(str(indexes[0])) % 2**32)
    selected_cols = np.random.choice(cols, max(1, len(cols)//10), replace=False)

    for col in selected_cols:
        shift = np.random.randint(1, 6)
        idx_arr = np.array(indexes)
        shifted_idx = idx_arr - shift
        
        valid_mask = (shifted_idx >= 0) & np.isin(shifted_idx, df.index)
        if valid_mask.any():
            v_idx = idx_arr[valid_mask]
            v_shift = shifted_idx[valid_mask]
            
            curr_vals = df.loc[v_idx, col].values
            past_vals = df.loc[v_shift, col].values
            df.loc[v_idx, col] = 0.8 * curr_vals + 0.2 * past_vals
    return df

# =========================
# Recovery (imperfect) - VECTORIZED
# =========================
def apply_tail_effect(df_clean, df, anomaly_index_end, anomaly_length):
    cols = [c for c in df.columns if c not in ["date", "label", "Unnamed: 0"]]
    tail_len = int(0.2 * anomaly_length)
    
    tail_start_global = anomaly_index_end + 1
    tail_end_global = tail_start_global + tail_len
    
    df_start, df_end = df.index[0], df.index[-1]
    overlap_start = max(df_start, tail_start_global)
    overlap_end = min(df_end, tail_end_global)
    
    if overlap_start > overlap_end: return df 
        
    target_idx = list(range(overlap_start, min(overlap_end + 1, df_end + 1)))
    
    anchor_idx = overlap_start - 1
    
    # Kịch bản 1: Cửa sổ có điểm tựa phía trước
    if anchor_idx in df.index:
        working_idx = [anchor_idx] + target_idx
        data_matrix = df.loc[working_idx, cols].values
        
        for i in range(1, len(data_matrix)):
            drift = np.random.uniform(0.95, 1.05, size=len(cols))
            data_matrix[i] = apply_noise(data_matrix[i-1] * drift, 0.02)
            
        df.loc[target_idx, cols] = data_matrix[1:]
        
    # Kịch bản 2: Cửa sổ bị cắt ngang ngay tại điểm bắt đầu của đuôi (không có anchor)
    elif len(target_idx) > 1:
        data_matrix = df.loc[target_idx, cols].values
        
        # Bắt đầu vòng lặp từ index 1, lấy index 0 làm gốc
        for i in range(1, len(data_matrix)):
            drift = np.random.uniform(0.95, 1.05, size=len(cols))
            data_matrix[i] = apply_noise(data_matrix[i-1] * drift, 0.02)
            
        df.loc[target_idx, cols] = data_matrix

    return df

# =========================
# Labels (VECTORIZED)
# =========================
def apply_labels(df, overlap_indexes, anomaly_index_start, anomaly_index_end, is_fake=False):
    if "label" not in df.columns: df["label"] = 0
    if is_fake or not overlap_indexes: return df

    idx_arr = np.array(overlap_indexes)
    anomaly_length = anomaly_index_end - anomaly_index_start
    fade = max(3, int(0.15 * max(1, anomaly_length)))

    # Tạo mảng labels mới kích thước bằng overlap_indexes
    new_labels = np.zeros(len(idx_arr))

    # Vùng lõi (Core): Nhãn = 1
    core_mask = (idx_arr >= (anomaly_index_start + fade)) & (idx_arr <= (anomaly_index_end - fade))
    new_labels[core_mask] = 1

    # Vùng biên trái (Khởi phát)
    left_mask = idx_arr < (anomaly_index_start + fade)
    if left_mask.any():
        probs_left = (idx_arr[left_mask] - anomaly_index_start + 1) / fade
        probs_left = np.clip(probs_left, 0.0, 1.0)
        new_labels[left_mask] = (np.random.rand(sum(left_mask)) < probs_left).astype(int)

    # Vùng biên phải (Kết thúc)
    right_mask = idx_arr > (anomaly_index_end - fade)
    if right_mask.any():
        probs_right = (anomaly_index_end - idx_arr[right_mask] + 1) / fade
        probs_right = np.clip(probs_right, 0.0, 1.0)
        new_labels[right_mask] = (np.random.rand(sum(right_mask)) < probs_right).astype(int)

    # Cập nhật nhãn bằng np.maximum (những điểm đã là 1 trước đó thì không bị đè về 0)
    df.loc[idx_arr, "label"] = np.maximum(df.loc[idx_arr, "label"].values, new_labels)

    return df

# =========================
# Inject one
# =========================
def inject_one(df_clean, df, anomaly, position_map, name_id_map, is_fake=False):
    start = anomaly["start"]
    end = anomaly["end"]
    anomaly_length = end - start
    
    indexes_start = df.index[0]
    indexes_end = df.index[-1]
    window_length = indexes_end - indexes_start + 1
    # Cho phép sự cố bắt đầu TRƯỚC cửa sổ (để bắt đoạn đuôi) 
    # Hoặc bắt đầu GẦN CUỐI cửa sổ (để bắt đoạn đầu)
    # ---------------------------------------------------------
    # NEW LOGIC: Ép buộc tỷ lệ giao thoa tối thiểu (ví dụ 20%)
    # ---------------------------------------------------------
    overlap_ratio = 0.2
    
    # Số điểm tối thiểu phải lọt vào cửa sổ
    min_overlap = max(1, int(anomaly_length * overlap_ratio))
    
    # Đảm bảo min_overlap không đòi hỏi nhiều hơn kích thước cửa sổ hiện tại
    min_overlap = min(min_overlap, window_length)
    
    # Tính toán lại khoảng kẹp Random
    min_start = indexes_start - anomaly_length + min_overlap
    max_start = indexes_end - min_overlap + 1
    
    # Fallback an toàn nếu cửa sổ quá kỳ dị
    if min_start > max_start:
        min_start = indexes_start
        max_start = indexes_end
    
    anomaly_index_start = np.random.randint(min_start, max_start + 1)
    anomaly_index_end = anomaly_index_start + anomaly_length
    
    # Tìm vùng giao thoa thực tế giữa sự cố và cửa sổ
    overlap_start = max(indexes_start, anomaly_index_start)
    overlap_end = min(indexes_end, anomaly_index_end)

    if overlap_start > overlap_end:
        return df # Sự cố vô tình trượt hẳn ra ngoài, trả về sạch

    indexes = list(range(overlap_start, min(overlap_end + 1, indexes_end + 1)))

    timeline_effects = build_timeline_effects(anomaly, position_map, name_id_map)
    anomaly_cols = set()
    for t in timeline_effects:
        anomaly_cols.update(timeline_effects[t].keys())
    
    if np.random.rand() < 0.3:
        df = apply_long_term_drift(df_clean, df, indexes, exclude_cols=anomaly_cols)
    if np.random.rand() < 0.3:
        df = apply_seasonal_shift(df_clean, df, indexes, exclude_cols=anomaly_cols)
    df = apply_causal_injection(df_clean, df, indexes, anomaly_index_start, anomaly_length, timeline_effects)
    df = apply_tail_effect(df_clean, df, anomaly_index_end, anomaly_length)
    df = apply_labels(df, indexes, anomaly_index_start, anomaly_index_end, is_fake)

    return df

# =========================
# Inject full
# =========================
def inject_full(df, anomaly_ls, position_map, name_id_map):
    df = df.copy()
    
    df.columns = df.columns.astype(str)

    cols = [c for c in df.columns if c not in ["date", "label", "Unnamed: 0"]]
    df[cols] = df[cols].astype(float)

    for anomaly in anomaly_ls:
        df_clean = df.copy()
        if "FALSE ANOMALY (LABEL 0)" in anomaly or "HARD NEGATIVE - MUST LABEL AS 0" in anomaly\
                or "FALSE ANOMALY (LABEL 0)" in anomaly or "FALSE ANOMALY (LABEL 0)" in anomaly\
                    or "LABEL 0" in anomaly or "LABEL AS 0" in anomaly:
                        df = inject_one(df_clean, df, anomaly, position_map, name_id_map, is_fake=True)
        else:
            df = inject_one(df_clean, df, anomaly, position_map, name_id_map, is_fake=False)

    label = df['label'].values
    feature = df[cols].values
    return feature, label
