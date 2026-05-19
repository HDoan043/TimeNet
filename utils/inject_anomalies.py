# =========================
# Utils
# =========================
import numpy as np
import pandas as pd


def apply_noise(x, scale=0.05):
    return x * (1 + np.random.normal(0, scale))

def smooth_transition(start, end, length):
    return np.linspace(start, end, length)

def lag(df_clean, df, i, col, col_multiplier_map):
    lag = np.random.randint(5, 10)
    if i - lag >= 0 and (i-lag) in df_clean.index:
        df.at[i, col] = df_clean.at[i - lag, col] * col_multiplier_map[col]
        df.at[i, col] = max(df.at[i, col], 0)
        
def trend(df_clean, df, i, col, col_multiplier_map):
    base = df_clean.at[i, col] * col_multiplier_map[col]
    if (i-1) in df_clean.index and (i-5) in df_clean.index:
        trend = df_clean.at[i - 1, col] - df_clean.at[i - 5, col]
        df.at[i, col] = base - 0.7 * trend
        df.at[i, col] = max(df.at[i, col], 0)
    
def correlation(df_clean, df, i, col, col_multiplier_map):
    cols = list(col_multiplier_map.keys())   
    if len(cols) <= 1:
        return

    other = np.random.choice([c for c in cols if c != col])

    m1 = col_multiplier_map[col]
    m2 = col_multiplier_map.get(other, 1.0)
    base = df_clean.at[i, col] * col_multiplier_map[col]
    other_val = df_clean.at[i, other] * m2

    base_clean = df_clean.at[i, col]
    other_clean = df_clean.at[i, other]

    delta_base = base - base_clean
    delta_other = other_val - other_clean

    # cùng chiều
    if (m1 - 1) * (m2 - 1) > 0:
        df.at[i, col] = base_clean + 0.7 * delta_base + 0.3 * delta_other

    # ngược chiều
    elif (m1 - 1) * (m2 - 1) < 0:
        df.at[i, col] = base_clean + 0.7 * delta_base - 0.3 * delta_other

    else:
        df.at[i, col] = base
        
    df.at[i, col] = max(df.at[i, col], 0)

def noise(df_clean, df, i, col, col_multiplier_map):
    base = df_clean.at[i, col] * col_multiplier_map[col]
    df.at[i, col] = apply_noise(base, 0.03)
    df.at[i, col] = max(df.at[i, col], 0)
# =========================
# Build timeline effects
# =========================
def build_timeline_effects(anomaly, position_map, name_id_map):
    '''
    Input: 
        _ anomaly: JSON {'anomaly':'...', 'start':..., 'end':..., 'counters': [{'module', 'scenario', 'timeline', 'impact_multiplier'}, {}, ...]}
        _ position_map: map position, scenario, timeline to the real counter name
        _ name_id_map: map the counter name to counter id
    Output: Dictionary:
        {
            'timeline_1': [<list of counter ids that are affected by anomaly in timeline 1 and their impact multiplier>],
            'timeline_2': [<list of counter ids that are affected by anomaly in timeline 2 and their impact multiplier>],
            ...
        }
    '''
    
    timeline_effects = {}

    for counter in anomaly["counter"]:
        module = counter["module"]
        scenario = counter["scenario"]
        timeline = int(counter["timeline"])
        multiplier = counter["impact_multiplier"]

        try:
            names = position_map[module][scenario][str(timeline)]
        except:
            continue

        if not names:
            continue

        # k = max(1, int(len(names) * np.random.uniform(0.3, 0.7)))
        # selected = np.random.choice(names, k, replace=False)
        selected = names

        for name in selected:
            if name not in name_id_map:
                continue

            cid = str(name_id_map[name])
            timeline_effects.setdefault(timeline, {})
            timeline_effects[timeline][cid] = multiplier

    # Skip the timeline which contains no counters effected
    new_timeline_effects = {}
    for key, counters_dict in timeline_effects.items():
        if len(list(counters_dict.keys())) > 0:
            new_timeline_effects[key] = counters_dict.copy()
    return new_timeline_effects
    # return dict(sorted(new_timeline_effects.items(), key=lambda x: x[0]))

# =========================
# Core injection (ADVANCED)
# =========================
def apply_causal_injection(df_clean, df, overlap_indexes, anomaly_start, anomaly_length, timeline_effects):
    if not timeline_effects:
        return df

    # 1.  
    sorted_timelines = sorted(list(timeline_effects.keys()))
    num_timelines = len(sorted_timelines)
    
    # Nhóm các index thực tế vào đúng timeline của nó dựa trên tiến trình %
    timeline_to_indexes = {t: [] for t in sorted_timelines}
    
    for i in overlap_indexes:
        # Tính xem điểm i này nằm ở bao nhiêu % của toàn bộ sự cố gốc
        progress = (i - anomaly_start) / max(1, anomaly_length)
        
        # Ánh xạ % đó sang đúng timeline phase
        timeline_idx = int(progress * num_timelines)
        timeline_idx = min(max(0, timeline_idx), num_timelines - 1) 
        
        current_timeline = sorted_timelines[timeline_idx]
        timeline_to_indexes[current_timeline].append(i)
        
    # num_steps = len(timeline_effects)
    # step_size = max(1, len(indexes) // num_steps)
    
    # used = set()
    # for step, timeline in enumerate(timeline_effects.keys()):
        # # 2. Modify to make steps contain different indexes --> more realistic
        # shift = np.random.randint(-step_size//4, step_size//4)
        # start = max(0, step * step_size + shift)
        # end = min(len(indexes), (step + 1) * step_size + shift)
        # raw_idx = indexes[start:end]
        
        # step_idx = []
        # for i in raw_idx:
        #     if i not in used or np.random.rand() < 0.1:  # cho overlap nhẹ
        #         step_idx.append(i)
        #         used.add(i)
                
        # step_idx = indexes[start:end]
    for timeline, step_idx in timeline_to_indexes.items():
        if not step_idx: 
            continue
            
        cols = list(timeline_effects[timeline].keys())

        col_multiplier_map = timeline_effects[timeline]
        distort_type = np.random.rand()
            
        for i in step_idx:
            r = distort_type + np.random.normal(0, 0.05)  # jitter nhẹ

            # ===== 1. TEMPORAL SHIFT (long delay)
            if r < 0.15:
                distort = lag
            # ===== 2. LOCAL TREND BREAK
            elif r < 0.3:
                distort = trend
            # ===== 3. CROSS-FEATURE DISTORTION (weak, realistic)
            elif r < 0.5:
                distort = correlation
            # ===== 4. MICRO NOISE ONLY
            else:
                distort = noise
                
            if i < 30:
                continue

            for col in cols:
                if col not in df.columns:
                    continue

                distort(df_clean, df, i, col, col_multiplier_map)

    return df

# =========================
# LONG-TERM DRIFT
# =========================
def apply_long_term_drift(df_clean, df, indexes, exclude_cols=None):
    cols = [c for c in df.columns if c not in ["date", "label", "Unnamed: 0"]]

    if exclude_cols:
        cols = [c for c in cols if c not in exclude_cols]

    np.random.seed(hash(str(indexes[0])) % 2**32)

    selected_cols = np.random.choice(cols, max(1, len(cols)//10), replace=False)


    for col in selected_cols:
        drift_strength = np.random.uniform(0.01, 0.05)

        values = df.loc[indexes, col].values
        trend = smooth_transition(0, drift_strength * np.mean(values), len(values))
        
        df.loc[indexes, col] = values + trend

    return df

# =========================
# SEASONAL DISTORTION
# =========================
def apply_seasonal_shift(df_clean, df, indexes, exclude_cols=None):
    cols = [c for c in df.columns if c not in ["date", "label", "Unnamed: 0"]]

    if exclude_cols:
        cols = [c for c in cols if c not in exclude_cols]

    np.random.seed(hash(str(indexes[0])) % 2**32)

    selected_cols = np.random.choice(cols, max(1, len(cols)//10), replace=False)

    for col in selected_cols:
        shift = np.random.randint(1, 6)

        for i in indexes:
            if i - shift >= 0 and (i-shift) in df.index:
                df.at[i, col] = 0.8 * df.at[i, col] + 0.2 * df.at[i - shift, col]

    return df
# =========================
# Recovery (imperfect)
# =========================
def apply_tail_effect(df_clean, df, anomaly_index_end, anomaly_length):
    cols = [c for c in df.columns if c not in ["date", "label", "Unnamed: 0"]]

    tail_len = int(0.2 * anomaly_length)
    
    # Giới hạn vùng đuôi thực tế trên trục thời gian
    tail_start_global = anomaly_index_end + 1
    tail_end_global = tail_start_global + tail_len
    
    # Tìm vùng giao thoa giữa đuôi sự cố và cửa sổ df hiện tại
    df_start = df.index[0]
    df_end = df.index[-1]
    
    overlap_start = max(df_start, tail_start_global)
    overlap_end = min(df_end, tail_end_global)
    
    if overlap_start > overlap_end:
        return df # Đuôi không lọt vào cửa sổ này
        
    for i in range(overlap_start, min(overlap_end + 1, df_end + 1)):
        # Đảm bảo không bị lỗi index out of bounds khi lấy prev
        if (i - 1) in df.index:
            prev = df.loc[i - 1]
            for col in cols:
                drift = np.random.uniform(0.95, 1.05)
                df.at[i, col] = apply_noise(prev[col] * drift, 0.02)

    return df

# =========================
# Labels (AMBIGUOUS BOUNDARY)
# =========================
def apply_labels(df, overlap_indexes, anomaly_index_start, anomaly_index_end, is_fake=False):
    if "label" not in df.columns:
        df["label"] = 0

    if not is_fake:
        anomaly_length = anomaly_index_end - anomaly_index_start
        # Tính độ mờ dựa trên chiều dài sự cố GỐC
        fade = max(3, int(0.15 * max(1, anomaly_length)))

        for idx in overlap_indexes:
            # Nếu điểm này nằm ở vùng lõi của sự cố (cách xa 2 biên)
            if (anomaly_index_start + fade) <= idx <= (anomaly_index_end - fade):
                df.at[idx, "label"] = 1
            else:
                # Nếu nằm ở vùng biên khởi phát
                if idx < (anomaly_index_start + fade):
                    prob = (idx - anomaly_index_start + 1) / fade
                # Nếu nằm ở vùng biên kết thúc
                else:
                    prob = (anomaly_index_end - idx + 1) / fade
                
                # Đảm bảo xác suất hợp lệ
                prob = max(0.0, min(1.0, prob))
                if np.random.rand() < prob:
                    df.at[idx, "label"] = 1

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
    
    # Cho phép sự cố bắt đầu TRƯỚC cửa sổ (để bắt đoạn đuôi) 
    # Hoặc bắt đầu GẦN CUỐI cửa sổ (để bắt đoạn đầu)
    min_start = indexes_start - anomaly_length + 1
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
