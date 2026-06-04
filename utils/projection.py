import numpy as np
from matplotlib import pyplot as plt


def project_point_cloud(points, colors, view_point, width=2048, height=1024, max_depth=100.0):
    points = points.transpose(1, 0).cpu().numpy()
    colors = colors.transpose(1, 0).cpu().numpy()
    vp = view_point.cpu().numpy()
    points_centered = points - vp

    X = points_centered[:, 0]
    Y = points_centered[:, 1]
    Z = points_centered[:, 2]
    r = np.sqrt(X * X + Y * Y + Z * Z)
    valid = (r > 0) & (r <= float(max_depth))

    panorama = np.zeros((height, width, 3), dtype=np.float32)
    depth_map = np.full((height, width), float(max_depth), dtype=np.float32)
    mask = np.ones((height, width), dtype=np.uint8)

    if not np.any(valid):
        # plt.imsave("panorama.png", panorama)
        # plt.imsave("depth_map.png", depth_map, cmap="inferno")
        # plt.imsave("mask.png", mask, cmap="gray")
        return panorama, depth_map, mask

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]
    r = r[valid]
    colors = colors[valid]

    lon = np.arctan2(Z, X)
    yr = np.clip(-Y / r, -1.0, 1.0)
    lat = np.arcsin(yr)

    u = ((lon + np.pi) / (2.0 * np.pi) * width).astype(np.int32)
    v = ((lat / np.pi + 0.5) * height).astype(np.int32)
    u = np.clip(u, 0, width - 1)
    v = np.clip(v, 0, height - 1)

    lin_idx = v * width + u

    order = np.lexsort((r, lin_idx))
    lin_idx_sorted = lin_idx[order]
    r_sorted = r[order]
    colors_sorted = colors[order]

    uniq_lin_idx, first_pos = np.unique(lin_idx_sorted, return_index=True)

    depth_flat = depth_map.ravel()
    pano_flat = panorama.reshape(-1, 3)
    mask_flat = mask.ravel()

    depth_flat[uniq_lin_idx] = r_sorted[first_pos]
    pano_flat[uniq_lin_idx] = colors_sorted[first_pos]
    mask_flat[uniq_lin_idx] = 0

    # plt.imsave("panorama.png", panorama)
    # plt.imsave("depth_map.png", depth_map, cmap="inferno")
    # plt.imsave("mask.png", mask, cmap="gray")

    return panorama, depth_map, mask

def project_point_cloud_orig(points, colors, view_point, width=2048, height=1024, max_depth=100.0):
    # 获取点和颜色
    points = points.transpose(1, 0).cpu().numpy()
    colors = colors.transpose(1, 0).cpu().numpy()
    vp = view_point.cpu().numpy()
    points_centered = points - vp

    # 转换到球面坐标系
    x, y, z = points_centered[:, 0], points_centered[:, 1], points_centered[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arctan2(y, x)  # 水平角 [-π, π]
    phi = np.arcsin(z / r)  # 垂直角 [-π/2, π/2]

    # 像素映射
    u = (theta + np.pi) / (2 * np.pi) * width
    v = (np.pi / 2 - phi) / np.pi * height
    u = np.clip(u.astype(int), 0, width - 1)
    v = np.clip(v.astype(int), 0, height - 1)

    # 初始化全景图、深度图和掩码
    panorama = np.zeros((height, width, 3), dtype=np.float32)
    depth_map = np.full((height, width), max_depth, dtype=np.float32)
    mask = np.ones((height, width), dtype=np.uint8)

    # 填充全景图、深度图和掩码
    progress_bar = tqdm(range(1, points.shape[0] + 1), desc="Projecting to new pose")
    for i in range(points.shape[0]):
        # progress_bar = tqdm(range(1, points.shape[0] + 1), desc="Projecting to new pose")
        if r[i] > max_depth:  # 忽略超过最大深度的点
            progress_bar.update(1)
            continue
        # 仅在当前深度更小的情况下填充
        if depth_map[v[i], u[i]] > r[i]:
            depth_map[v[i], u[i]] = r[i]
            panorama[v[i], u[i]] = colors[i]
            mask[v[i], u[i]] = 0
        progress_bar.update(1)
    progress_bar.close()

    # debug
    plt.imsave("panorama.png", panorama)
    plt.imsave("depth_map.png", depth_map, cmap="inferno")
    plt.imsave("mask.png", mask, cmap="gray")

    return panorama, depth_map, mask


def project_point_cloud_2(points, colors, view_point, width=2048, height=1024, max_depth=100.0, radius=1):
    points = points.transpose(1, 0).cpu().numpy()
    colors = colors.transpose(1, 0).cpu().numpy()
    vp = view_point.cpu().numpy()
    points_centered = points - vp
    X = points_centered[:, 0]
    Y = points_centered[:, 1]
    Z = points_centered[:, 2]
    r = np.sqrt(X * X + Y * Y + Z * Z)
    valid = (r > 0) & (r <= float(max_depth))
    panorama = np.zeros((height, width, 3), dtype=np.float32)
    depth_map = np.full((height, width), float(max_depth), dtype=np.float32)
    mask = np.ones((height, width), dtype=np.uint8)
    if not np.any(valid):
        return panorama, depth_map, mask
    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]
    r = r[valid]
    colors = colors[valid]
    lon = np.arctan2(Z, X)
    yr = np.clip(-Y / r, -1.0, 1.0)
    lat = np.arcsin(yr)
    u0 = ((lon + np.pi) / (2.0 * np.pi) * width).astype(np.int32)
    v0 = ((lat / np.pi + 0.5) * height).astype(np.int32)
    u0 = np.clip(u0, 0, width - 1)
    v0 = np.clip(v0, 0, height - 1)
    offs = []
    R = int(radius)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            if dx * dx + dy * dy <= R * R:
                offs.append((dx, dy))
    if len(offs) == 0:
        offs = [(0, 0)]
    dx = np.array([o[0] for o in offs], dtype=np.int32)
    dy = np.array([o[1] for o in offs], dtype=np.int32)
    u_rep = (u0[:, None] + dx[None, :]).clip(0, width - 1)
    v_rep = (v0[:, None] + dy[None, :]).clip(0, height - 1)
    lin_idx = (v_rep * width + u_rep).ravel()
    r_rep = np.repeat(r, len(offs))
    colors_rep = np.repeat(colors, len(offs), axis=0)
    order = np.lexsort((r_rep, lin_idx))
    lin_sorted = lin_idx[order]
    r_sorted = r_rep[order]
    colors_sorted = colors_rep[order]
    uniq_lin_idx, first_pos = np.unique(lin_sorted, return_index=True)
    depth_flat = depth_map.ravel()
    pano_flat = panorama.reshape(-1, 3)
    mask_flat = mask.ravel()
    depth_flat[uniq_lin_idx] = r_sorted[first_pos]
    pano_flat[uniq_lin_idx] = colors_sorted[first_pos]
    mask_flat[uniq_lin_idx] = 0
    return panorama, depth_map, mask


def project_point_cloud_3(points, colors, view_point, width=2048, height=1024, max_depth=100.0,
                           radius_alpha=2.0, radius_max=4, polar_gamma=1.5, wrap_u=True,
                           super_sample=1.5, weight_sigma=0.2, do_morph=True,
                           kernel_close=5, kernel_open=3, depth_gate_thr=0.3):
    import cv2
    pts = points.transpose(1, 0).cpu().numpy()
    cols = colors.transpose(1, 0).cpu().numpy()
    vp = view_point.cpu().numpy()
    pc = pts - vp
    X = pc[:, 0]
    Y = pc[:, 1]
    Z = pc[:, 2]
    r = np.sqrt(X * X + Y * Y + Z * Z)
    valid = (r > 0) & (r <= float(max_depth))
    Hs = int(height * super_sample)
    Ws = int(width * super_sample)
    if not np.any(valid):
        pano = np.zeros((height, width, 3), dtype=np.float32)
        dep = np.full((height, width), float(max_depth), dtype=np.float32)
        m = np.ones((height, width), dtype=np.uint8)
        return pano, dep, m
    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]
    r = r[valid]
    cols = cols[valid]
    lon = np.arctan2(Z, X)
    yr = np.clip(-Y / r, -1.0, 1.0)
    lat = np.arcsin(yr)
    u0 = ((lon + np.pi) / (2.0 * np.pi) * Ws).astype(np.int32)
    v0 = ((lat / np.pi + 0.5) * Hs).astype(np.int32)
    u0 = np.clip(u0, 0, Ws - 1)
    v0 = np.clip(v0, 0, Hs - 1)
    r_med = np.median(r) + 1e-6
    R = np.clip(np.round(radius_alpha * (1.0 + r / r_med)), 1, radius_max).astype(np.int32)
    polar = (v0 < Hs * 0.1) | (v0 > Hs * 0.9)
    R[polar] = np.clip(np.round(R[polar] * polar_gamma), 1, radius_max).astype(np.int32)
    offs = []
    Rm = int(radius_max)
    for dy in range(-Rm, Rm + 1):
        for dx in range(-Rm, Rm + 1):
            if dx * dx + dy * dy <= Rm * Rm:
                offs.append((dx, dy))
    dx = np.array([o[0] for o in offs], dtype=np.int32)
    dy = np.array([o[1] for o in offs], dtype=np.int32)
    if wrap_u:
        u_rep = (u0[:, None] + dx[None, :]) % Ws
    else:
        u_rep = (u0[:, None] + dx[None, :]).clip(0, Ws - 1)
    v_rep = (v0[:, None] + dy[None, :]).clip(0, Hs - 1)
    active = (dx[None, :] ** 2 + dy[None, :] ** 2) <= (R[:, None] ** 2)
    lin_idx_all = (v_rep * Ws + u_rep).ravel()
    active_flat = active.ravel()
    if not np.any(active_flat):
        pano = np.zeros((height, width, 3), dtype=np.float32)
        dep = np.full((height, width), float(max_depth), dtype=np.float32)
        m = np.ones((height, width), dtype=np.uint8)
        return pano, dep, m
    lin_idx = lin_idx_all[active_flat]
    r_rep = np.repeat(r, len(offs))
    r_flat = r_rep[active_flat]
    cols_rep = np.repeat(cols, len(offs), axis=0)
    cols_flat = cols_rep[active_flat]
    depth_flat = np.full(Hs * Ws, float(max_depth), dtype=np.float32)
    np.minimum.at(depth_flat, lin_idx, r_flat)
    sigma = max(1e-6, weight_sigma * r_med)
    dr = r_flat - depth_flat[lin_idx]
    w = np.exp(-(dr * dr) / (sigma * sigma)) + 1e-6
    sum_w = np.zeros(Hs * Ws, dtype=np.float32)
    np.add.at(sum_w, lin_idx, w)
    sum0 = np.zeros(Hs * Ws, dtype=np.float32)
    sum1 = np.zeros(Hs * Ws, dtype=np.float32)
    sum2 = np.zeros(Hs * Ws, dtype=np.float32)
    np.add.at(sum0, lin_idx, w * cols_flat[:, 0])
    np.add.at(sum1, lin_idx, w * cols_flat[:, 1])
    np.add.at(sum2, lin_idx, w * cols_flat[:, 2])
    pano_flat = np.zeros((Hs * Ws, 3), dtype=np.float32)
    valid_pix = sum_w > 0
    pano_flat[valid_pix, 0] = sum0[valid_pix] / sum_w[valid_pix]
    pano_flat[valid_pix, 1] = sum1[valid_pix] / sum_w[valid_pix]
    pano_flat[valid_pix, 2] = sum2[valid_pix] / sum_w[valid_pix]
    mask_flat = np.ones(Hs * Ws, dtype=np.uint8)
    mask_flat[valid_pix] = 0
    panorama_ss = pano_flat.reshape(Hs, Ws, 3)
    depth_ss = depth_flat.reshape(Hs, Ws)
    mask_ss = mask_flat.reshape(Hs, Ws)
    if super_sample != 1:
        panorama = cv2.resize(panorama_ss, (width, height), interpolation=cv2.INTER_AREA)
        depth_map = cv2.resize(depth_ss, (width, height), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask_ss.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)
        mask = (mask > 0.5).astype(np.uint8)
    else:
        panorama = panorama_ss
        depth_map = depth_ss
        mask = mask_ss
    if do_morph:
        kC = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_close, kernel_close))
        kO = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_open, kernel_open))
        mtmp = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kC)
        mtmp = cv2.morphologyEx(mtmp, cv2.MORPH_OPEN, kO)
        if depth_gate_thr > 0:
            mean = cv2.blur(depth_map, (5, 5))
            var = cv2.blur((depth_map - mean) ** 2, (5, 5))
            gate = (var < (depth_gate_thr * depth_gate_thr)).astype(np.uint8)
            mask = np.where(gate > 0, mtmp, mask).astype(np.uint8)
        else:
            mask = mtmp
    return panorama, depth_map, mask
    