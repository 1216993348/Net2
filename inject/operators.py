"""
6个高区分度算子 - 效果明显版
"""

import numpy as np


# ==================== 1. SPIKE ====================
class Spike:
    """宽峰异常 - 明显凸起"""
    @staticmethod
    def apply(x, d, intensity, s, e):
        L = e - s
        if L < 10:
            return x
        r = np.std(x[s:e, d]) + 1e-6
        left_val = x[s-1, d] if s > 0 else x[s, d]
        right_val = x[e, d] if e < len(x) else x[e-1, d]

        # 线性基线
        t = np.arange(L) / L
        baseline = left_val * (1 - t) + right_val * t

        # 1-2个宽峰
        n_spikes = np.random.choice([1, 2], p=[0.7, 0.3])

        for _ in range(n_spikes):
            # 峰宽度：占异常段的 20%-40%
            spike_width = max(5, int(L * np.random.uniform(0.2, 0.4)))
            spike_center = np.random.randint(L//4, 3*L//4)
            spike_start = max(0, spike_center - spike_width // 2)
            spike_end = min(L, spike_center + spike_width // 2)

            actual_width = spike_end - spike_start
            if actual_width > 0:
                spike_t = np.arange(actual_width)
                spike_shape = np.exp(-((spike_t - actual_width/2) / (actual_width/3))**2)
                spike_height = intensity * r * np.random.uniform(3, 6)
                baseline[spike_start:spike_end] += spike_height * spike_shape

        x[s:e, d] = baseline
        return x


# ==================== 2. DRIFT ====================
class Drift:
    @staticmethod
    def apply(x, d, intensity, s, e):
        L = e - s
        r = np.std(x[s:e, d]) + 1e-6
        left_val = x[s - 1, d] if s > 0 else x[s, d]

        t = np.arange(L) / L
        # 非线性曲线：指数或对数
        if np.random.rand() < 0.5:
            curve = np.exp(t * 3) - 1  # 先缓后急
        else:
            curve = 1 - np.exp(-t * 3)  # 先急后缓

        curve = curve / curve[-1]  # 归一化
        direction = 1 if np.random.rand() < 0.5 else -1
        trend_amp = intensity * r * np.random.uniform(1.5, 3)

        trend = left_val + direction * trend_amp * curve
        x[s:e, d] = trend
        return x


# ==================== 3. SHIFT ====================
class Shift:
    """整体平移 - 明显抬高或降低"""
    @staticmethod
    def apply(x, d, intensity, s, e):
        L = e - s
        if L < 2:
            return x
        r = np.std(x[s:e, d]) + 1e-6
        original = x[s:e, d].copy()

        # 大平移幅度
        shift = np.random.uniform(2.0, 4.0) * r
        shift = shift * (1 if np.random.rand() < 0.5 else -1)

        shifted = original + shift
        x[s:e, d] = shifted
        return x


# ==================== 4. PERIOD ====================
class Period:
    """周期破坏 - 所有模式强制生效"""

    @staticmethod
    def apply(x, d, intensity, s, e):
        L = e - s
        if L < 10:
            return x
        r = np.std(x[s:e, d]) + 1e-6
        left_val = x[s - 1, d] if s > 0 else x[s, d]
        right_val = x[e, d] if e < len(x) else x[e - 1, d]

        t = np.arange(L) / L
        baseline = left_val * (1 - t) + right_val * t

        # 随机选择模式
        mode = np.random.choice(['fft', 'sine', 'shake', 'chirp'], p=[0.40, 0.20, 0.20, 0.20])

        if mode == 'sine':
            # 正弦波
            osc_freq = np.random.uniform(3, 12)
            osc_amp = intensity * r * np.random.uniform(2, 5)
            injected = osc_amp * np.sin(2 * np.pi * osc_freq * t)
            alpha = np.random.uniform(0.1, 0.3)
            original = x[s:e, d].copy()
            original_centered = original - baseline
            result = baseline + alpha * original_centered + (1 - alpha) * injected

        elif mode == 'fft':
            # FFT 花活：摧毁主频 + 随机化相位 + 增强其他频率
            col = x[s:e, d].copy()
            col_centered = col - baseline
            fft = np.fft.fft(col_centered)
            half = L // 2

            # 找到主频
            mag = np.abs(fft)
            mag[0] = 0
            peak = np.argmax(mag[:half]) if half > 1 else 1

            # 1. 摧毁主频（完全移除）
            fft[peak] = 0
            if -peak < len(fft):
                fft[-peak] = 0

            # 2. 随机化其他频率的幅度和相位
            for i in range(1, half):
                if i != peak:
                    # 随机幅度 (0.5x ~ 3x)
                    fft[i] = fft[i] * np.random.uniform(0.5, 3) * (1 + intensity)
                    # 随机相位
                    fft[i] = fft[i] * np.exp(1j * np.random.uniform(-np.pi, np.pi))
                    if -i < len(fft):
                        fft[-i] = np.conj(fft[i])

            # 3. 增强高频（让波形更乱）
            high_start = half // 2
            for i in range(high_start, half):
                fft[i] = fft[i] * (1 + intensity * 2)
                if -i < len(fft):
                    fft[-i] = np.conj(fft[i])

            # IFFT 还原
            injected_centered = np.fft.ifft(fft).real
            injected = baseline + injected_centered

            # 混合：注入信号主导
            alpha = np.random.uniform(0.1, 0.3)
            original = x[s:e, d].copy()
            result = alpha * original + (1 - alpha) * injected

        elif mode == 'shake':
            # 剧烈抖动
            noise_amp = intensity * r * np.random.uniform(2, 5)
            noise = np.random.randn(L) * noise_amp
            # 添加低频包络让抖动更有节奏
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)
            noise = noise * envelope
            alpha = np.random.uniform(0.1, 0.3)
            original = x[s:e, d].copy()
            original_centered = original - baseline
            result = baseline + alpha * original_centered + (1 - alpha) * noise

        else:  # chirp
            # 频率扫描 + 幅度调制
            freq_start = np.random.uniform(0.5, 2)
            freq_end = np.random.uniform(10, 20)
            if np.random.rand() < 0.5:
                freq_start, freq_end = freq_end, freq_start
            instant_freq = freq_start + (freq_end - freq_start) * t
            phase = 2 * np.pi * np.cumsum(instant_freq) / L
            chirp = np.sin(phase)
            # 幅度调制
            amp_mod = 0.5 + 0.5 * np.sin(2 * np.pi * 1 * t)
            chirp_amp = intensity * r * np.random.uniform(2, 5)
            injected = chirp_amp * chirp * amp_mod
            alpha = np.random.uniform(0.1, 0.3)
            original = x[s:e, d].copy()
            original_centered = original - baseline
            result = baseline + alpha * original_centered + (1 - alpha) * injected

        x[s:e, d] = result
        return x

# ==================== 5. CASCADE ====================
class Cascade:
    """级联异常 - 触发点 → 振荡衰减 → 恢复"""
    @staticmethod
    def apply(x, d, intensity, s, e):
        L = e - s
        if L < 20:
            return x
        r = np.std(x[s:e, d]) + 1e-6
        left_val = x[s-1, d] if s > 0 else x[s, d]
        right_val = x[e, d] if e < len(x) else x[e-1, d]

        t = np.arange(L) / L

        # 基线
        baseline = left_val * (1 - t) + right_val * t

        # 触发点（20%位置）
        trigger_pos = int(L * 0.2)
        trigger = np.zeros(L)
        trigger[trigger_pos] = intensity * r * 6

        # 振荡衰减（20%-70%）
        decay_start = trigger_pos
        decay_end = int(L * 0.7)
        decay_len = decay_end - decay_start
        cascade_effect = np.zeros(L)

        if decay_len > 0:
            decay_t = np.arange(decay_len) / decay_len
            decay_env = np.exp(-decay_t * 2.5)
            osc = np.sin(2 * np.pi * 3 * decay_t) * decay_env
            cascade_effect[decay_start:decay_end] = intensity * r * 3 * osc

        # 恢复期（70%之后）
        recovery_start = decay_end
        if recovery_start < L:
            recovery_t = np.arange(L - recovery_start) / (L - recovery_start)
            recovery_factor = 1 - recovery_t
            cascade_effect[recovery_start:] *= recovery_factor

        x[s:e, d] = baseline + trigger + cascade_effect
        return x


# ==================== 6. MISSING ====================
class Missing:
    """置零异常 - 整个异常段全部置零"""
    @staticmethod
    def apply(x, d, intensity, s, e):
        x[s:e, d] = 0
        return x


# ==================== 算子映射 ====================
OPERATOR_MAP = {
    1: Spike,
    2: Drift,
    3: Shift,
    4: Period,
    5: Cascade,
    6: Missing,
}