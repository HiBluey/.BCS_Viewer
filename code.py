import sys
import traceback
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, TextBox
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D, proj3d
import tkinter as tk
from tkinter import filedialog

# ================= 1. 防闪退封装 =================
def main_wrapper():
    try:
        main_application()
    except Exception:
        print("\n" + "="*60)
        print("!!! 程序发生严重错误 / CRITICAL ERROR !!!")
        print("="*60)
        traceback.print_exc()
        print("="*60)
        input("\n按回车键退出 (Press Enter to exit)...")

# ================= 2. 核心数学算法 =================
M_709 = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
M_2020 = np.array([[0.6370, 0.1446, 0.1689], [0.2627, 0.6780, 0.0593], [0.0000, 0.0281, 1.0610]])
M_P3 = np.array([[0.4866, 0.2657, 0.1982], [0.2290, 0.6917, 0.0793], [0.0000, 0.0451, 1.0439]])
M_XYZ_LMS = np.array([[0.3592, 0.6976, -0.0359], [-0.1922, 1.1095, 0.0755], [0.0071, 0.0748, 0.8438]])
M_LMS_ICTCP = np.array([[0.5, 0.5, 0], [1.6137, -3.3234, 1.7097], [4.378, -4.2455, -0.1325]])

def get_primaries_and_levels(stimuli, measured):
    tol = 0.001
    idx_r = np.where(np.all(np.abs(stimuli - [1,0,0]) < tol, axis=1))[0]
    idx_g = np.where(np.all(np.abs(stimuli - [0,1,0]) < tol, axis=1))[0]
    idx_b = np.where(np.all(np.abs(stimuli - [0,0,1]) < tol, axis=1))[0]
    idx_w = np.where(np.all(np.abs(stimuli - [1,1,1]) < tol, axis=1))[0]
    idx_k = np.where(np.all(np.abs(stimuli - [0,0,0]) < tol, axis=1))[0]
    res = {}
    if len(idx_r): res['R'] = measured[idx_r[0]]
    if len(idx_g): res['G'] = measured[idx_g[0]]
    if len(idx_b): res['B'] = measured[idx_b[0]]
    if len(idx_w): res['W'] = measured[idx_w[0]]
    if len(idx_k): res['K'] = measured[idx_k[0]]
    else: res['K'] = np.array([0.0, 0.0, 0.0])
    
    is_gray = np.all(np.isclose(stimuli[:, 0:1], stimuli[:, 1:3]), axis=1)
    res['gray_s'] = stimuli[is_gray, 0]
    res['gray_xyz'] = measured[is_gray]
    
    sorter = np.argsort(res['gray_s'])
    res['gray_s'] = res['gray_s'][sorter]
    res['gray_xyz'] = res['gray_xyz'][sorter]
    res['gray_y'] = res['gray_xyz'][:, 1]
    
    return res

def calc_matrix_for_target(gamut_mode, target_wp_xyz, native_primaries):
    if gamut_mode == 'Native' and 'R' in native_primaries:
        r, g, b = native_primaries['R'], native_primaries['G'], native_primaries['B']
        M_base = np.stack([r, g, b], axis=1)
        native_w = r + g + b
        with np.errstate(divide='ignore', invalid='ignore'):
            gain = target_wp_xyz / native_w
            gain = np.where(np.isfinite(gain), gain, 1.0)
        M_final = M_base * gain[:, None]
    else:
        if gamut_mode == 'Rec.2020': M_std = M_2020
        elif gamut_mode == 'P3': M_std = M_P3
        else: M_std = M_709
        D65_XYZ = np.array([0.95047, 1.00000, 1.08883])
        wp_norm = target_wp_xyz / target_wp_xyz[1] if target_wp_xyz[1] > 0 else D65_XYZ
        gain = wp_norm / D65_XYZ
        M_final = M_std * gain[:, None]
    return M_final

def xyz_to_ictcp(xyz):
    lms = (M_XYZ_LMS @ xyz.T).T
    lms_norm = np.maximum(0, lms / 10000.0)
    m1 = 2610.0 / 16384.0; m2 = 2523.0 / 32.0 
    c1 = 3424.0 / 4096.0; c2 = 2413.0 / 128.0; c3 = 2392.0 / 128.0
    val_pow = np.power(lms_norm, m1)
    num = c1 + c2 * val_pow
    den = 1.0 + c3 * val_pow
    lms_pq = np.power(num / den, m2)
    return (M_LMS_ICTCP @ lms_pq.T).T

def calc_de_itp(ictcp1, ictcp2):
    diff = ictcp1 - ictcp2
    return 720.0 * np.sqrt(diff[:,0]**2 + 0.5*diff[:,1]**2 + diff[:,2]**2)

def xyz_to_lab(xyz, white_xyz):
    xyz_n = xyz / white_xyz
    mask = xyz_n > 0.008856
    f = np.zeros_like(xyz_n)
    f[mask] = np.cbrt(xyz_n[mask])
    f[~mask] = (7.787 * xyz_n[~mask]) + 16/116
    L = 116 * f[:,1] - 16
    a = 500 * (f[:,0] - f[:,1])
    b = 200 * (f[:,1] - f[:,2])
    return np.stack([L, a, b], axis=1)

def calc_de_2000(lab1, lab2):
    L1, a1, b1 = lab1.T; L2, a2, b2 = lab2.T
    avg_L = (L1 + L2) / 2.0
    C1 = np.sqrt(a1**2 + b1**2); C2 = np.sqrt(a2**2 + b2**2); avg_C = (C1 + C2) / 2.0
    G = 0.5 * (1 - np.sqrt(avg_C**7 / (avg_C**7 + 25**7)))
    a1p = (1 + G) * a1; a2p = (1 + G) * a2
    C1p = np.sqrt(a1p**2 + b1**2); C2p = np.sqrt(a2p**2 + b2**2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360; h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    avg_Cp = (C1p + C2p) / 2.0
    diff_h = h2p - h1p
    delta_h_p = np.where(np.abs(diff_h) <= 180, diff_h, np.where(diff_h > 180, diff_h - 360, diff_h + 360))
    delta_H_p = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(delta_h_p) / 2.0)
    dLp = L2 - L1; dCp = C2p - C1p
    Sl = 1 + (0.015 * (avg_L - 50)**2) / np.sqrt(20 + (avg_L - 50)**2)
    Sc = 1 + 0.045 * avg_Cp
    sum_hp = h1p + h2p; abs_diff = np.abs(diff_h)
    avg_hp = sum_hp / 2.0
    avg_hp = np.where((abs_diff > 180) & (sum_hp < 360), (sum_hp + 360) / 2.0, avg_hp)
    avg_hp = np.where((abs_diff > 180) & (sum_hp >= 360), (sum_hp - 360) / 2.0, avg_hp)
    T = 1 - 0.17 * np.cos(np.radians(avg_hp - 30)) + 0.24 * np.cos(np.radians(2 * avg_hp)) + \
        0.32 * np.cos(np.radians(3 * avg_hp + 6)) - 0.20 * np.cos(np.radians(4 * avg_hp - 63))
    Sh = 1 + 0.015 * avg_Cp * T
    dTheta = 30 * np.exp(-((avg_hp - 275) / 25)**2)
    Rc = 2 * np.sqrt(avg_Cp**7 / (avg_Cp**7 + 25**7))
    Rt = -np.sin(np.radians(2 * dTheta)) * Rc
    return np.sqrt((dLp / Sl)**2 + (dCp / Sc)**2 + (delta_H_p / Sh)**2 + Rt * (dCp / Sc) * (delta_H_p / Sh))

def xyz_to_xyY(xyz):
    s = np.sum(xyz, axis=1)
    s[s==0] = 1.0
    x = xyz[:,0] / s
    y = xyz[:,1] / s
    Y = xyz[:,1]
    return np.stack([x, y, Y], axis=1)

# ================= 3. 主界面类 =================
class BCSAnalyzer:
    def __init__(self, stimuli, measured, filename):
        self.stimuli = stimuli
        self.measured = measured
        self.raw_data = get_primaries_and_levels(stimuli, measured)
        
        # 参数
        self.p_gamut = 'Rec.709'
        self.p_eotf = 'Gamma 2.4'
        self.p_wp = 'D65'
        self.p_metric = 'dE 2000'
        self.p_luma_max_mode = 'Native'
        self.p_luma_min_mode = 'Native'
        self.man_wx = 0.3127; self.man_wy = 0.3290
        self.man_max_y = 100.0; self.man_min_y = 0.0
        self.filter_min = 0.0; self.filter_max = 100.0
        
        self.visible_indices = []
        self.visible_coords_3d = []
        self.current_errors = []
        
        # 窗口
        self.fig = plt.figure(figsize=(16, 9), facecolor='white')
        self.fig.canvas.manager.set_window_title(f"BCS Analysis: {filename}")
        self.fig.add_artist(plt.Rectangle((0.01, 0.02), 0.22, 0.96, color='#f5f5f5', zorder=-1))
        
        # --- 布局重构 (EOTF 正方形, 右下角) ---
        
        # 1. 顶部 3D/2D (保持不变)
        # y=0.55 ~ 0.95
        self.ax_3d = self.fig.add_axes([0.25, 0.55, 0.35, 0.40], projection='3d')
        self.ax_cbar = self.fig.add_axes([0.61, 0.58, 0.015, 0.35])
        self.ax_2d = self.fig.add_axes([0.68, 0.55, 0.28, 0.40])
        
        # 2. 底部区域 (y=0.05 ~ 0.50, Height=0.45)
        
        # 右下: EOTF 正方形
        # 16:9 屏幕下, 宽度 0.26 对应高度 0.45 左右差不多是正方形
        self.ax_eotf = self.fig.add_axes([0.70, 0.05, 0.26, 0.45])
        
        # 左下: RGB Balance (中左) & Histogram (左底)
        # 宽度 0.40 (0.26 ~ 0.66)
        
        # RGB Balance: 顶部对齐 y=0.50
        # Height 0.22 -> y=0.28
        self.ax_rgb = self.fig.add_axes([0.26, 0.28, 0.40, 0.22])
        
        # Histogram: 底部对齐 y=0.05
        # Height 0.18 -> Top y=0.23. (Gap 0.05 between graphs)
        self.ax_hist = self.fig.add_axes([0.26, 0.05, 0.40, 0.18])
        
        colors = [(0.0, "#00FF00"), (0.33, "#FFA500"), (0.66, "#FF0000"), (1.0, "#800080")]
        self.cmap = LinearSegmentedColormap.from_list("grad", colors, N=256)
        
        # 初始化占位 (将在 update_plot 中创建)
        self.annot = None
        self.annot_2d = None
        
        self.create_controls()
        self.update_plot()
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_hover)
        
    def create_controls(self):
        bg = '#f5f5f5'; sz = 9
        self.fig.text(0.02, 0.96, "Color Space Settings", fontsize=11, fontweight='bold', color='#333333')
        
        ax = plt.axes([0.03, 0.81, 0.18, 0.13], facecolor=bg); ax.set_title("Target Gamut", fontsize=sz, loc='left'); ax.axis('off')
        self.rb_gamut = RadioButtons(ax, ('Rec.709', 'Rec.2020', 'P3', 'Native'))
        self.rb_gamut.on_clicked(self.set_gamut)
        
        ax = plt.axes([0.03, 0.66, 0.18, 0.13], facecolor=bg); ax.set_title("Target EOTF", fontsize=sz, loc='left'); ax.axis('off')
        self.rb_eotf = RadioButtons(ax, ('Gamma 2.2', 'Gamma 2.4', 'PQ', 'Native'))
        self.rb_eotf.on_clicked(self.set_eotf)
        
        ax = plt.axes([0.03, 0.53, 0.18, 0.11], facecolor=bg); ax.set_title("Target White Point", fontsize=sz, loc='left'); ax.axis('off')
        self.rb_wp = RadioButtons(ax, ('D65', 'Native', 'Manual'))
        self.rb_wp.on_clicked(self.set_wp)
        
        self.fig.text(0.04, 0.50, "x:", fontsize=9, backgroundcolor=bg)
        self.box_wx = TextBox(plt.axes([0.06, 0.495, 0.05, 0.03]), '', initial="0.3127")
        self.box_wx.on_submit(self.update_manual_vals)
        self.fig.text(0.12, 0.50, "y:", fontsize=9, backgroundcolor=bg)
        self.box_wy = TextBox(plt.axes([0.14, 0.495, 0.05, 0.03]), '', initial="0.3290")
        self.box_wy.on_submit(self.update_manual_vals)

        self.fig.text(0.02, 0.44, "Luma & Filter", fontsize=11, fontweight='bold', color='#333333')
        self.fig.add_artist(plt.Line2D([0.02, 0.22], [0.47, 0.47], color='#cccccc'))
        
        ax = plt.axes([0.03, 0.34, 0.18, 0.08], facecolor=bg); ax.set_title("Max Luma", fontsize=sz, loc='left'); ax.axis('off')
        self.rb_lmax = RadioButtons(ax, ('Native', 'Manual'))
        self.rb_lmax.on_clicked(self.set_lmax_mode)
        self.box_lmax = TextBox(plt.axes([0.12, 0.36, 0.07, 0.03]), '', initial="100")
        self.box_lmax.on_submit(self.update_manual_vals)
        
        ax = plt.axes([0.03, 0.24, 0.18, 0.08], facecolor=bg); ax.set_title("Min Luma", fontsize=sz, loc='left'); ax.axis('off')
        self.rb_lmin = RadioButtons(ax, ('Native', 'Manual'))
        self.rb_lmin.on_clicked(self.set_lmin_mode)
        self.box_lmin = TextBox(plt.axes([0.12, 0.26, 0.07, 0.03]), '', initial="0.0")
        self.box_lmin.on_submit(self.update_manual_vals)
        
        ax = plt.axes([0.03, 0.13, 0.18, 0.08], facecolor=bg); ax.set_title("Metric", fontsize=sz, loc='left'); ax.axis('off')
        self.rb_metric = RadioButtons(ax, ('dE 2000', 'dE ITP'))
        self.rb_metric.on_clicked(self.set_metric)
        
        self.fig.text(0.02, 0.10, "Filter dE:", fontsize=sz, fontweight='bold', backgroundcolor=bg)
        self.fig.text(0.04, 0.07, "Min:", fontsize=sz, backgroundcolor=bg)
        self.box_fmin = TextBox(plt.axes([0.07, 0.065, 0.05, 0.03]), '', initial="0")
        self.box_fmin.on_submit(self.update_filter)
        self.fig.text(0.13, 0.07, "Max:", fontsize=sz, backgroundcolor=bg)
        self.box_fmax = TextBox(plt.axes([0.16, 0.065, 0.05, 0.03]), '', initial="100")
        self.box_fmax.on_submit(self.update_filter)

    def set_gamut(self, v): self.p_gamut = v; self.update_plot()
    def set_eotf(self, v): self.p_eotf = v; self.update_plot()
    def set_wp(self, v):
        self.p_wp = v
        if v == 'D65': self.box_wx.set_val("0.3127"); self.box_wy.set_val("0.3290")
        elif v == 'Native' and 'W' in self.raw_data:
            W = self.raw_data['W']; s = np.sum(W)
            if s > 0: self.box_wx.set_val(f"{W[0]/s:.4f}"); self.box_wy.set_val(f"{W[1]/s:.4f}")
        self.update_plot()
    def set_lmax_mode(self, v):
        self.p_luma_max_mode = v
        if v == 'Native' and 'W' in self.raw_data: self.box_lmax.set_val(f"{self.raw_data['W'][1]:.2f}")
        self.update_plot()
    def set_lmin_mode(self, v):
        self.p_luma_min_mode = v
        if v == 'Native' and 'K' in self.raw_data: self.box_lmin.set_val(f"{self.raw_data['K'][1]:.4f}")
        self.update_plot()
    def set_metric(self, v): self.p_metric = v; self.update_plot()
    def update_filter(self, t):
        try: self.filter_min = float(self.box_fmin.text); self.filter_max = float(self.box_fmax.text); self.update_plot()
        except: pass
    def update_manual_vals(self, t):
        try:
            self.man_wx = float(self.box_wx.text); self.man_wy = float(self.box_wy.text)
            self.man_max_y = float(self.box_lmax.text); self.man_min_y = float(self.box_lmin.text)
            self.update_plot()
        except: pass

    def on_hover(self, event):
        if len(self.visible_indices) == 0: return
        # 检查是否在 3D 或 2D 坐标轴内
        is_3d = (event.inaxes == self.ax_3d)
        is_2d = (event.inaxes == self.ax_2d)
        
        if not (is_3d or is_2d):
            if self.annot: self.annot.set_visible(False)
            if self.annot_2d: self.annot_2d.set_visible(False)
            self.fig.canvas.draw_idle()
            return

        filtered_x = self.visible_coords_3d[:,1]
        filtered_y = self.visible_coords_3d[:,2]
        filtered_z = self.visible_coords_3d[:,0]
        
        if is_3d and self.annot:
            proj_matrix = self.ax_3d.get_proj()
            x2, y2, _ = proj3d.proj_transform(filtered_x, filtered_y, filtered_z, proj_matrix)
            x_pix, y_pix = self.ax_3d.transData.transform(np.column_stack([x2, y2])).T
            dist = np.sqrt((x_pix - event.x)**2 + (y_pix - event.y)**2)
            min_idx = np.argmin(dist)
            if dist[min_idx] < 15: self.show_annotation(min_idx, x2[min_idx], y2[min_idx], self.annot)
            else: self.annot.set_visible(False); self.fig.canvas.draw_idle()
            
        elif is_2d and self.annot_2d:
            pts_pix = self.ax_2d.transData.transform(np.column_stack([filtered_x, filtered_y]))
            dist = np.sqrt((pts_pix[:,0] - event.x)**2 + (pts_pix[:,1] - event.y)**2)
            min_idx = np.argmin(dist)
            if dist[min_idx] < 15: self.show_annotation(min_idx, filtered_x[min_idx], filtered_y[min_idx], self.annot_2d)
            else: self.annot_2d.set_visible(False); self.fig.canvas.draw_idle()

    def show_annotation(self, idx_in_filter, x_pos, y_pos, annot_obj):
        real_idx = self.visible_indices[idx_in_filter]
        rgb = self.stimuli[real_idx]
        xyz = self.measured[real_idx]
        err = self.current_errors[real_idx]
        xyY = xyz_to_xyY(xyz.reshape(1,3))[0]
        text = f"RGB: {rgb[0]:.2f}, {rgb[1]:.2f}, {rgb[2]:.2f}\nxyY: {xyY[0]:.4f}, {xyY[1]:.4f}, {xyY[2]:.1f}\n{self.p_metric}: {err:.2f}"
        annot_obj.xy = (x_pos, y_pos)
        annot_obj.set_text(text)
        annot_obj.set_visible(True)
        self.fig.canvas.draw_idle()

    def get_targets(self):
        target_max = self.raw_data['W'][1] if (self.p_luma_max_mode == 'Native' and 'W' in self.raw_data) else self.man_max_y
        target_min = self.raw_data['K'][1] if (self.p_luma_min_mode == 'Native' and 'K' in self.raw_data) else self.man_min_y
        if self.p_wp == 'D65': wx, wy = 0.3127, 0.3290
        elif self.p_wp == 'Native' and 'W' in self.raw_data:
            W = self.raw_data['W']; s = np.sum(W)
            wx, wy = (W[0]/s, W[1]/s) if s>0 else (0.3127, 0.329)
        else: wx, wy = self.man_wx, self.man_wy
        wz = 1 - wx - wy; factor = 1.0 / wy if wy > 0 else 1.0
        wp_xyz_norm = np.array([wx*factor, 1.0, wz*factor])
        matrix = calc_matrix_for_target(self.p_gamut, wp_xyz_norm, self.raw_data)
        return target_min, target_max, wp_xyz_norm, matrix

    def update_plot(self):
        l_min, l_max, wp_xyz_norm, matrix = self.get_targets()
        
        # EOTF logic
        if self.p_eotf == 'Native':
            gs, gy = self.raw_data.get('gray_s'), self.raw_data.get('gray_y')
            if gs is not None and len(gs) > 1:
                gy_norm = (gy - np.min(gy)) / (np.max(gy) - np.min(gy))
                linear_rgb = np.interp(self.stimuli, gs, gy_norm)
            else: linear_rgb = np.power(self.stimuli, 2.2)
        elif self.p_eotf == 'Gamma 2.2': linear_rgb = np.power(self.stimuli, 2.2)
        elif self.p_eotf == 'Gamma 2.4': linear_rgb = np.power(self.stimuli, 2.4)
        elif self.p_eotf == 'PQ':
            m1 = 2610/16384; m2 = 2523/32; c1 = 3424/4096; c2 = 2413/128; c3 = 2392/128
            Np = np.power(self.stimuli, 1.0/m2); num = np.maximum(0, Np - c1); den = c2 - c3 * Np
            linear_rgb = np.power(np.maximum(0, num/den), 1.0/m1)
            
        xyz_rel = (matrix @ linear_rgb.T).T
        if self.p_eotf == 'PQ': 
            targ_xyz = xyz_rel * 10000.0
            Y_t = targ_xyz[:, 1]; mask_clip = Y_t > l_max
            if np.any(mask_clip):
                scale = l_max / Y_t[mask_clip]
                targ_xyz[mask_clip] *= scale[:, np.newaxis]
        else:
            xyz_black = wp_xyz_norm * l_min
            targ_xyz = xyz_black + (xyz_rel * (l_max - l_min))

        ictcp_m = xyz_to_ictcp(self.measured)
        if 'ITP' in self.p_metric:
            limit = 20.0; ictcp_t = xyz_to_ictcp(targ_xyz); errors = calc_de_itp(ictcp_m, ictcp_t)
            xl, yl, zl = 'Ct', 'Cp', 'I'
        else:
            limit = 6.0
            wp_abs = wp_xyz_norm * l_max if self.p_eotf != 'PQ' else wp_xyz_norm * 10000.0
            if wp_abs[1] == 0: wp_abs = np.array([95.04, 100.0, 108.88])
            lab_m = xyz_to_lab(self.measured, wp_abs); lab_t = xyz_to_lab(targ_xyz, wp_abs)
            errors = calc_de_2000(lab_m, lab_t)
            xl, yl, zl = 'Ct', 'Cp', 'I'
            
        self.current_errors = errors
        mask = (errors >= self.filter_min) & (errors <= self.filter_max)
        self.visible_indices = np.where(mask)[0]; self.visible_coords_3d = ictcp_m[mask]
        
        filtered_errors = errors[mask]
        fx = ictcp_m[mask, 1]; fy = ictcp_m[mask, 2]; fz = ictcp_m[mask, 0]
        c_vals = np.clip(filtered_errors / limit, 0, 1)

        # 1. 3D Plot
        self.ax_3d.clear()
        self.ax_3d.scatter(fx, fy, fz, c=c_vals, cmap=self.cmap, s=20, alpha=0.8, vmin=0, vmax=1)
        self.ax_3d.set_xlabel(xl); self.ax_3d.set_ylabel(yl); self.ax_3d.set_zlabel(zl)
        self.ax_3d.set_title(f"3D Analysis ({len(filtered_errors)} pts)")
        
        # 重新创建 Annotation (解决 clear 后丢失问题)
        self.annot = self.ax_3d.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                                       bbox=dict(boxstyle="round", fc="w", ec="k", alpha=0.9), arrowprops=dict(arrowstyle="->"), zorder=999)
        self.annot.set_visible(False)
        
        # 2. Colorbar
        self.ax_cbar.clear()
        norm = plt.Normalize(0, limit)
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=self.cmap), cax=self.ax_cbar)
        cb.set_label(f"{self.p_metric} ΔE")

        # 3. 2D Plot
        self.ax_2d.clear()
        self.ax_2d.scatter(fx, fy, c=c_vals, cmap=self.cmap, s=20, alpha=0.7, vmin=0, vmax=1)
        self.ax_2d.set_xlabel("$C_T$"); self.ax_2d.set_ylabel("$C_P$"); self.ax_2d.set_title("Chroma Plane")
        self.ax_2d.grid(True, linestyle='--', alpha=0.5)
        self.ax_2d.axhline(0, color='k', lw=0.5); self.ax_2d.axvline(0, color='k', lw=0.5)
        
        self.annot_2d = self.ax_2d.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                                            bbox=dict(boxstyle="round", fc="w", ec="k", alpha=0.9), arrowprops=dict(arrowstyle="->"), zorder=999)
        self.annot_2d.set_visible(False)

        # 4. RGB Balance & EOTF
        self.ax_rgb.clear(); self.ax_eotf.clear()
        gray_s = self.raw_data.get('gray_s', []); gray_xyz = self.raw_data.get('gray_xyz', [])
        
        if len(gray_s) > 1:
            try:
                inv_mat = np.linalg.inv(matrix)
                meas_rgb_lin = (inv_mat @ gray_xyz.T).T
                g_ch = meas_rgb_lin[:, 1]
                g_ch = np.where(g_ch==0, 1e-9, g_ch)
                r_bal = (meas_rgb_lin[:, 0] / g_ch) * 100.0
                g_bal = (meas_rgb_lin[:, 1] / g_ch) * 100.0
                b_bal = (meas_rgb_lin[:, 2] / g_ch) * 100.0
                
                self.ax_rgb.plot(gray_s, r_bal, 'r', label='R', lw=1.5)
                self.ax_rgb.plot(gray_s, g_bal, 'g', label='G', lw=1.5)
                self.ax_rgb.plot(gray_s, b_bal, 'b', label='B', lw=1.5)
                self.ax_rgb.axhline(100, color='gray', linestyle='--')
                self.ax_rgb.set_title("RGB Balance (%)")
                self.ax_rgb.set_ylim(80, 120); self.ax_rgb.grid(True, alpha=0.3)
            except: self.ax_rgb.text(0.5, 0.5, "Matrix Err", ha='center')

            # EOTF Square Plot (Signal Space)
            meas_Y = gray_xyz[:, 1]
            if self.p_eotf == 'PQ':
                m1=2610/16384; m2=2523/32; c1=3424/4096; c2=2413/128; c3=2392/128
                Y_norm = np.clip(meas_Y / 10000.0, 0, 1)
                val_pow = np.power(Y_norm, m1)
                num = c1 + c2 * val_pow; den = 1 + c3 * val_pow
                signal_meas = np.power(num / den, m2)
            else:
                y_peak = np.max(meas_Y) if np.max(meas_Y) > 0 else l_max
                gamma = 2.4 if self.p_eotf == 'Gamma 2.4' else 2.2
                with np.errstate(divide='ignore', invalid='ignore'):
                    signal_meas = np.power(meas_Y / y_peak, 1.0/gamma)

            self.ax_eotf.scatter(gray_s, signal_meas, c='k', s=10)
            self.ax_eotf.plot([0,1], [0,1], color='gray', linestyle='--')
            self.ax_eotf.set_title("EOTF Tracking")
            self.ax_eotf.set_xlabel("Input Signal"); self.ax_eotf.set_ylabel("Measured Signal")
            self.ax_eotf.set_xlim(0, 1); self.ax_eotf.set_ylim(0, 1)
            self.ax_eotf.grid(True, alpha=0.3)
            # 强制正方形 (Square aspect)
            self.ax_eotf.set_aspect('equal', adjustable='box')
            
        else:
            self.ax_rgb.axis('off'); self.ax_eotf.axis('off')

        self.ax_hist.clear()
        self.ax_hist.hist(filtered_errors, bins=60, color='#666666', edgecolor='white')
        avg = np.mean(filtered_errors) if len(filtered_errors)>0 else 0
        mx = np.max(filtered_errors) if len(filtered_errors)>0 else 0
        self.ax_hist.set_title(f"Error Dist: Avg {avg:.2f} | Max {mx:.2f}")
        self.ax_hist.set_xlabel("Delta E")
        self.ax_hist.axvline(avg, color='orange', linestyle='--', label='Average')
        
        self.fig.canvas.draw_idle()

def main_application():
    print("Select BCS File...")
    root = tk.Tk(); root.withdraw()
    path = filedialog.askopenfilename(filetypes=[("BCS", "*.bcs")])
    if not path:
        print("No file selected."); input("Press Enter to exit..."); return
        
    print(f"Loading {path}...")
    try:
        tree = ET.parse(path)
        data = tree.getroot().find('data')
        if data is None: raise ValueError("No <data> tag found in XML")
        
        s_list, m_list = [], []
        for p in data:
            s = p.find('stimuli'); r = p.find('results').find('XYZ')
            if s is not None and r is not None:
                s_list.append([float(s.find('red').text), float(s.find('green').text), float(s.find('blue').text)])
                m_list.append([float(r.find('X').text), float(r.find('Y').text), float(r.find('Z').text)])
        
        if len(s_list) == 0: raise ValueError("Data is empty")
        
        app = BCSAnalyzer(np.array(s_list), np.array(m_list), path.split('/')[-1])
        plt.show()
        
    except Exception as e:
        print(f"Error parsing file: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")

if __name__ == "__main__":
    main_wrapper()
