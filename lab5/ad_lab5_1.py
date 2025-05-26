import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from scipy.signal import butter, filtfilt

# ---------------------------
# Функція для генерації гармоніки з шумом
# ---------------------------
def harmonic_with_noise(t, amplitude, frequency, phase, noise_mean, noise_covariance, noise=None):
    y_clean = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    if noise is None:
        noise = np.random.normal(noise_mean, np.sqrt(noise_covariance), size=len(t))
    y_noisy = y_clean + noise
    return y_clean, y_noisy, noise

# ---------------------------
# Фільтр Butterworth
# ---------------------------
def apply_filter(data, cutoff, fs=100, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# ---------------------------
# Початкові параметри
# ---------------------------
init_amplitude = 1.0
init_frequency = 1.0
init_phase = 0.0
init_noise_mean = 0.0
init_noise_covariance = 0.1
init_cutoff = 2.0

# Масив часу
t = np.linspace(0, 10, 1000)
fs = len(t) / (t[-1] - t[0])

# Зберігаємо фіксований шум
current_noise = np.random.normal(init_noise_mean, np.sqrt(init_noise_covariance), size=len(t))

# Початкові сигнали
y_clean, y_noisy, _ = harmonic_with_noise(t, init_amplitude, init_frequency, init_phase, init_noise_mean, init_noise_covariance, noise=current_noise)
y_filtered = apply_filter(y_noisy, cutoff=init_cutoff, fs=fs)

# ---------------------------
# Графік
# ---------------------------
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.5)

line_clean, = ax.plot(t, y_clean, label='Clean Signal', color='blue')
line_noisy, = ax.plot(t, y_noisy, label='Noisy Signal', color='orange')
line_filtered, = ax.plot(t, y_filtered, label='Filtered Signal', color='green')

ax.legend(loc='upper right')
ax.set_xlabel('Time')
ax.set_ylabel('Amplitude')

# ---------------------------
# Слайдери
# ---------------------------
axcolor = 'lightgoldenrodyellow'
slider_positions = [0.4, 0.35, 0.3, 0.25, 0.2, 0.15]
amp_slider = Slider(plt.axes([0.1, slider_positions[0], 0.8, 0.03], facecolor=axcolor), 'Amplitude', 0.1, 5.0, valinit=init_amplitude)
freq_slider = Slider(plt.axes([0.1, slider_positions[1], 0.8, 0.03], facecolor=axcolor), 'Frequency', 0.1, 5.0, valinit=init_frequency)
phase_slider = Slider(plt.axes([0.1, slider_positions[2], 0.8, 0.03], facecolor=axcolor), 'Phase', 0.0, 2*np.pi, valinit=init_phase)
mean_slider = Slider(plt.axes([0.1, slider_positions[3], 0.8, 0.03], facecolor=axcolor), 'Noise Mean', -1.0, 1.0, valinit=init_noise_mean)
cov_slider = Slider(plt.axes([0.1, slider_positions[4], 0.8, 0.03], facecolor=axcolor), 'Noise Covariance', 0.001, 1.0, valinit=init_noise_covariance)
cutoff_slider = Slider(plt.axes([0.1, slider_positions[5], 0.8, 0.03], facecolor=axcolor), 'Filter Cutoff Freq', 0.1, 10.0, valinit=init_cutoff)

# ---------------------------
# Чекбокси
# ---------------------------
check_ax = plt.axes([0.8, 0.91, 0.15, 0.1])
check = CheckButtons(check_ax, ['Show Noise', 'Apply Filter'], [True, True])

# ---------------------------
# Кнопка Reset
# ---------------------------
reset_ax = plt.axes([0.65, 0.92, 0.1, 0.04])
button = Button(reset_ax, 'Reset')

# ---------------------------
# Зберігаємо поточні значення шуму
# ---------------------------
prev_noise_mean = init_noise_mean
prev_noise_cov = init_noise_covariance

# ---------------------------
# Функція оновлення
# ---------------------------
def update(val):
    global current_noise, prev_noise_mean, prev_noise_cov

    amplitude = amp_slider.val
    frequency = freq_slider.val
    phase = phase_slider.val
    noise_mean = mean_slider.val
    noise_cov = cov_slider.val
    cutoff = cutoff_slider.val
    show_noise = check.get_status()[0]
    apply_filter_flag = check.get_status()[1]

    # Генеруємо новий шум, якщо змінилися параметри шуму
    if noise_mean != prev_noise_mean or noise_cov != prev_noise_cov:
        current_noise = np.random.normal(noise_mean, np.sqrt(noise_cov), size=len(t))
        prev_noise_mean = noise_mean
        prev_noise_cov = noise_cov

    y_clean = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    y_noisy = y_clean + current_noise
    y_filtered = apply_filter(y_noisy, cutoff=cutoff, fs=fs) if apply_filter_flag else y_noisy

    line_clean.set_ydata(y_clean)
    line_noisy.set_ydata(y_noisy)
    line_noisy.set_visible(show_noise)

    line_filtered.set_ydata(y_filtered)
    line_filtered.set_visible(apply_filter_flag)

    fig.canvas.draw_idle()

# ---------------------------
# Reset
# ---------------------------
def reset(event=None):
    global current_noise, prev_noise_mean, prev_noise_cov
    amp_slider.reset()
    freq_slider.reset()
    phase_slider.reset()
    mean_slider.reset()
    cov_slider.reset()
    cutoff_slider.reset()
    # Увімкнення обох чекбоксів
    if not check.get_status()[0]: check.set_active(0)
    if not check.get_status()[1]: check.set_active(1)
    # Скидання шуму
    prev_noise_mean = init_noise_mean
    prev_noise_cov = init_noise_covariance
    current_noise = np.random.normal(prev_noise_mean, np.sqrt(prev_noise_cov), size=len(t))
    update(None)

# ---------------------------
# Події
# ---------------------------
for s in [amp_slider, freq_slider, phase_slider, mean_slider, cov_slider, cutoff_slider]:
    s.on_changed(update)
check.on_clicked(update)
button.on_clicked(reset)

# ---------------------------
# Інструкції
# ---------------------------
print("Програма для побудови гармоніки з шумом:")
print("- Слайдери: керують параметрами сигналу та шуму.")
print("- Чекбокс 'Show Noise' – показ/приховування шуму.")
print("- Чекбокс 'Apply Filter' – вмикає/вимикає фільтрацію.")
print("- Слайдер 'Filter Cutoff Freq' – керує частотою зрізу фільтру.")
print("- Кнопка 'Reset' – скидає всі налаштування.")

plt.show()
