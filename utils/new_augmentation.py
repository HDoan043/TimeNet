import numpy as np
from tqdm import tqdm

def jitter(x, sigma=0.01, clip=0.03):
    """
    Correlated Jitter: Noise biến thiên theo thời gian nhưng tác động lên các channel 
    có sự tương quan (cùng tăng hoặc cùng giảm nhẹ).
    x: [T, C]
    """
    T, C = x.shape
    # Noise theo trục thời gian (cùng giá trị bộ đếm tại cùng nhãn thời gian sẽ chịu 1 noise như nhau)
    noise_t = np.random.normal(0, sigma, size=(T, 1))
    # Tỷ trọng nhiễu riêng cho mỗi channel để tạo sự khác biệt nhỏ (trong mỗi channel thuộc cùng nhãn thời gian, có sự khác biệt nhỏ về nhiễu)
    channel_scale = np.random.normal(1.0, 0.05, size=(1, C))
    
    noise = noise_t * channel_scale
    noise = np.clip(noise, -clip, clip)
    
    return x * (1 + noise)


def scaling(x, sigma=0.02):
    """
    Global Scaling: Toàn bộ hệ thống tăng/giảm tải đồng bộ.
    Bảo toàn invariant: Req >= Accept.
    """
    factor = np.random.normal(1.0, sigma)
    return x * factor

def magnitude_warp(x, sigma=0.05, knot=4):
    """
    Correlated Magnitude Warp: Toàn bộ hệ thống fluctuation đồng bộ theo thời gian.
    """
    T, C = x.shape
    orig_steps = np.arange(T)
    warp_steps = np.linspace(0, T - 1, num=knot + 2)
    
    # Một đường Spline duy nhất cho tất cả channels
    random_warp = np.random.normal(1.0, sigma, size=(knot + 2,))
    random_warp = np.clip( random_warp, 0.9, 1.1)
    warper = CubicSpline(warp_steps, random_warp)(orig_steps)
    
    return x * warper[:, np.newaxis]

def time_warp(x, sigma=0.1, knot=4):
    """
    Global Time Warp: Co giãn trục thời gian đồng bộ cho tất cả counter.
    Giữ nguyên causal chain (Req t=10, Accept t=12).
    """
    T, C = x.shape
    orig_steps = np.arange(T)
    warp_steps = np.linspace(0, T - 1, num=knot + 2)
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2,))
    
    # Tạo trục thời gian mới bị biến dạng
    tt_raw = CubicSpline(warp_steps, warp_steps * random_warps)(orig_steps)
    tt_raw = np.maximum.accumulate(tt_raw)
    
    # Chuẩn hóa để đảm bảo tt nằm trong [0, T-1]
    tt_raw = np.clip(tt_raw, 0, T - 1)
    scale = (T - 1) / tt_raw[-1]
    tt = np.clip(scale * tt_raw, 0, T - 1)
    
    ret = np.zeros_like(x)
    for c in range(C):
        ret[:, c] = np.interp(orig_steps, tt, x[:, c])
        
    return ret

def window_slice(x):
    """
    Phase-Preserving Slice: Cắt và resize nhưng giữ trọng tâm cửa sổ.
    Tránh việc cắt mất các event chính ở giữa window.
    """
    T, C = x.shape
    reduce_ratio = np.random.uniform(0.94, 0.98)
    target_len = int(np.ceil(reduce_ratio * T))
    if target_len >= T:
        return x
    
    center = T // 2
    # Giới hạn vùng bắt đầu để đảm bảo center vẫn nằm trong slice
    start_min = max(0, center - target_len + 1)
    start_max = min(center, T - target_len)
    
    start = np.random.randint(start_min, start_max + 1)
    end = start + target_len
    
    sliced = x[start:end]
    
    # Interpolate lại về size gốc
    ret = np.zeros_like(x)
    orig_steps = np.arange(T)
    slice_steps = np.linspace(0, target_len - 1, T)
    for c in range(C):
        ret[:, c] = np.interp(slice_steps, np.arange(target_len), sliced[:, c])
        
    return ret

def window_warp(x, window_ratio=0.1, scales=[0.5, 2.]):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)
        
    window_starts = np.random.randint(low=1, high=x.shape[1]-warp_size-1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)
            
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[:window_starts[i],dim]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i],dim])
            end_seg = pat[window_ends[i]:,dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))                
            ret[i,:,dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1]-1., num=warped.size), warped).T
    return ret

def spawner(x, labels, sigma=0.05, verbose=0):
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6983028/
    # use verbose=-1 to turn off warnings
    # use verbose=1 to print out figures
    
    import utils.dtw as dtw
    random_points = np.random.randint(low=1, high=x.shape[1]-1, size=x.shape[0])
    window = np.ceil(x.shape[1] / 10.).astype(int)
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    
    ret = np.zeros_like(x)
    # for i, pat in enumerate(tqdm(x)):
    for i, pat in enumerate(x):
        # guarentees that same one isnt selected
        choices = np.delete(np.arange(x.shape[0]), i)
        # remove ones of different classes
        choices = np.where(l[choices] == l[i])[0]
        if choices.size > 0:     
            random_sample = x[np.random.choice(choices)]
            # SPAWNER splits the path into two randomly
            path1 = dtw.dtw(pat[:random_points[i]], random_sample[:random_points[i]], dtw.RETURN_PATH, slope_constraint="symmetric", window=window)
            path2 = dtw.dtw(pat[random_points[i]:], random_sample[random_points[i]:], dtw.RETURN_PATH, slope_constraint="symmetric", window=window)
            combined = np.concatenate((np.vstack(path1), np.vstack(path2+random_points[i])), axis=1)
            if verbose:
                # print(random_points[i])
                dtw_value, cost, DTW_map, path = dtw.dtw(pat, random_sample, return_flag = dtw.RETURN_ALL, slope_constraint=slope_constraint, window=window)
                dtw.draw_graph1d(cost, DTW_map, path, pat, random_sample)
                dtw.draw_graph1d(cost, DTW_map, combined, pat, random_sample)
            mean = np.mean([pat[combined[0]], random_sample[combined[1]]], axis=0)
            for dim in range(x.shape[2]):
                ret[i,:,dim] = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=mean.shape[0]), mean[:,dim]).T
        else:
            # if verbose > -1:
            #     print("There is only one pattern of class {}, skipping pattern average".format(l[i]))
            ret[i,:] = pat
    return jitter(ret, sigma=sigma)

# Proposed

def random_guided_warp_5g(x, neighbors):
    """
    x: [432, 136] - Cửa sổ hiện tại
    neighbors: List các cửa sổ normal khác trong batch
    """
    import utils.dtw as dtw
    T, C = x.shape
    
    # 1. Chọn ngẫu nhiên một "người hướng dẫn" (Normal khác)
    target = neighbors[np.random.randint(len(neighbors))]
    
    # 2. Tính toán Path dựa trên toàn bộ cấu trúc đa biến
    # Điều này giữ cho 136 channels luôn đồng bộ với nhau
    path = dtw.dtw(x, target, dtw.RETURN_PATH)
    
    # 3. Sửa lỗi lặp chỉ số (anti-plateau)
    # path[1] là danh sách chỉ số thời gian của chuỗi target
    warped_idx = np.maximum.accumulate(np.array(path[1]))
    
    # 4. Lấy dữ liệu theo path đã warp
    warped_data = target[warped_idx] # Hình dáng mới, độ dài có thể != T
    
    # 5. Tái lấy mẫu (Resampling) để đưa về đúng độ dài T=432
    # Đây là bước GPT thường thiếu, dẫn đến lỗi kích thước mảng
    current_len = warped_data.shape[0]
    resample_steps = np.linspace(0, current_len - 1, T)
    
    ret = np.zeros_like(x)
    for c in range(C):
        # Nội suy tuyến tính để làm mượt các đoạn gấp khúc do DTW
        alpha = np.random.uniform(0.7, 0.9)
        ret[:, c] = (alpha * x[:, c] + (1 - alpha) * warped_interp)
        
    return ret

def window_warp_5g(x, window_ratio=0.1, scale_range=(0.92, 1.08)):
    """
    x: [T, C] - Mẫu 2D từ __getitem__
    window_ratio: Tỷ lệ độ dài đoạn bị warp (nên để 0.05 - 0.15)
    scale_range: Ngưỡng co giãn (0.8 - 1.2 là an toàn cho viễn thông)
    """
    T, C = x.shape
    
    # 1. Xác định độ dài đoạn warp (tối thiểu 4 bước để spline/interp có nghĩa)
    warp_size = max(4, int(window_ratio * T))
    
    # 2. Chọn điểm bắt đầu ngẫu nhiên (tránh sát biên)
    start = np.random.randint(1, T - warp_size - 1)
    end = start + warp_size
    
    # 3. Chọn scale ngẫu nhiên trong vùng an toàn
    scale = np.random.uniform(*scale_range)
    warped_len = max(2, int(warp_size * scale))
    
    # 4. Trích xuất đoạn window
    window_segment = x[start:end] # [warp_size, C]
    
    # 5. Co giãn đoạn window (Dùng chung index cho TẤT CẢ channel)
    old_idx = np.arange(warp_size)
    new_idx = np.linspace(0, warp_size - 1, warped_len)
    
    warped_window = np.zeros((warped_len, C))
    for c in range(C):
        warped_window[:, c] = np.interp(new_idx, old_idx, window_segment[:, c])
    
    # 6. Ghép nối lại (Stitching)
    # merged sẽ có độ dài mới = (T - warp_size + warped_len)
    merged = np.concatenate([
        x[:start],
        warped_window,
        x[end:]
    ], axis=0)
    
    # 7. Resample toàn bộ về độ dài gốc T (432)
    # Bước này cực kỳ quan trọng để giữ nguyên cấu trúc TimesNet 
    # và làm mượt các điểm nối (boundary smoothing)
    final = np.zeros_like(x)
    merged_T = merged.shape[0]
    final_idx = np.linspace(0, merged_T - 1, T)
    
    for c in range(C):
        final[:, c] = np.interp(final_idx, np.arange(merged_T), merged[:, c])
        
    return final
