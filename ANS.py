import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
from collections import deque

# --- Paramètres ---
FS = 48000
BLOCK_SIZE = 512
N_TAPS = 64
PLOT_SAMPLES = 2048
THRESHOLD = 0.001
FFT_SIZE = 2048

# --- État LMS ---
mu_current = 0.0001
w = np.zeros(N_TAPS, dtype=np.float64)
x_buffer = np.zeros(N_TAPS, dtype=np.float64)

# --- Buffers partagés ---
buf_ref  = deque(np.zeros(PLOT_SAMPLES), maxlen=PLOT_SAMPLES)
buf_anti = deque(np.zeros(PLOT_SAMPLES), maxlen=PLOT_SAMPLES)
buf_err  = deque(np.zeros(PLOT_SAMPLES), maxlen=PLOT_SAMPLES)
buf_sum  = deque(np.zeros(PLOT_SAMPLES), maxlen=PLOT_SAMPLES)

block_count = 0
paused = False

def callback(indata, outdata, frames, time, status):
    global w, x_buffer, block_count, mu_current

    if status:
        print(f"[STATUS] {status}")

    block_count += 1

    ref = indata[:, 0].astype(np.float64) / 2147483647.0
    err = indata[:, 1].astype(np.float64) / 2147483647.0
    ref_rms = np.sqrt(np.mean(ref**2))

    output = np.zeros(frames, dtype=np.float64)

    for i in range(frames):
        x_buffer[1:] = x_buffer[:-1]
        x_buffer[0] = ref[i]

        y = np.dot(w, x_buffer)
        output[i] = -y

        if ref_rms > THRESHOLD:
            w += 2.0 * mu_current * err[i] * x_buffer

    # Somme ref + anti-bruit = signal résiduel théorique
    signal_sum = ref + output[:len(ref)]

    buf_ref.extend(ref)
    buf_anti.extend(output)
    buf_err.extend(err)
    buf_sum.extend(signal_sum)

    if block_count % 200 == 0:
        attenuation = 0
        if ref_rms > THRESHOLD:
            err_rms = np.sqrt(np.mean(err**2))
            if err_rms > 0:
                attenuation = 20 * np.log10(ref_rms / err_rms)
        print(f"[Block {block_count}] ref_rms={ref_rms:.5f}  "
              f"err_rms={np.sqrt(np.mean(err**2)):.5f}  "
              f"||w||={np.linalg.norm(w):.5f}  "
              f"µ={mu_current:.6f}  "
              f"Atténuation={attenuation:.1f} dB  "
              f"{'[ACTIF]' if ref_rms > THRESHOLD else '[SILENCE]'}")

    out_clipped = np.clip(output * 0.8, -1.0, 1.0)
    outdata[:, 0] = (out_clipped * 2147483647.0).astype(np.int32)
    outdata[:, 1] = (out_clipped * 2147483647.0).astype(np.int32)

# ================================================================
# FIGURE 1 — Signaux temporels + somme
# ================================================================
fig1, axes1 = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
plt.subplots_adjust(bottom=0.12, hspace=0.45)
fig1.suptitle("ANC — Domaine temporel", fontsize=13, fontweight='bold')

t_ms = np.linspace(0, PLOT_SAMPLES / FS * 1000, PLOT_SAMPLES)

line_ref,  = axes1[0].plot(t_ms, np.zeros(PLOT_SAMPLES), color='royalblue', lw=1)
line_anti, = axes1[1].plot(t_ms, np.zeros(PLOT_SAMPLES), color='tomato',    lw=1)
line_err,  = axes1[2].plot(t_ms, np.zeros(PLOT_SAMPLES), color='seagreen',  lw=1)
line_sum,  = axes1[3].plot(t_ms, np.zeros(PLOT_SAMPLES), color='darkorchid', lw=1)

# Superposition ref + anti sur graphique 2
line_sup_ref,  = axes1[1].plot(t_ms, np.zeros(PLOT_SAMPLES),
                                color='royalblue', lw=0.8, alpha=0.4, linestyle='--')

axes1[0].set_ylabel("Référence (micro 1)")
axes1[1].set_ylabel("Anti-bruit généré")
axes1[2].set_ylabel("Erreur résiduelle\n(micro 2)")
axes1[3].set_ylabel("Somme ref + anti\n(résidu théorique)")
axes1[3].set_xlabel("Temps (ms)")

# Annotation déphasage
annot_phase = axes1[1].annotate("", xy=(0,0),
                                 xytext=(0.01, 0.85),
                                 textcoords='axes fraction',
                                 fontsize=8, color='navy')

for ax in axes1:
    ax.axhline(0, color='gray', lw=0.5, linestyle='--')
    ax.grid(True, alpha=0.3)

# ================================================================
# FIGURE 2 — FFT spectrale
# ================================================================
fig2, axes2 = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
plt.subplots_adjust(bottom=0.18, hspace=0.45)
fig2.suptitle("ANC — Domaine fréquentiel (FFT)", fontsize=13, fontweight='bold')

freqs = np.fft.rfftfreq(FFT_SIZE, d=1/FS)
# On ne garde que 0-1000 Hz (zone utile pour ton TIPE)
freq_mask = freqs <= 1000

line_fft_ref,  = axes2[0].plot(freqs[freq_mask], np.zeros(np.sum(freq_mask)),
                                color='royalblue', lw=1.2)
line_fft_anti, = axes2[1].plot(freqs[freq_mask], np.zeros(np.sum(freq_mask)),
                                color='tomato',    lw=1.2)
line_fft_err,  = axes2[2].plot(freqs[freq_mask], np.zeros(np.sum(freq_mask)),
                                color='seagreen',  lw=1.2)

axes2[0].set_ylabel("Ref (dB)")
axes2[1].set_ylabel("Anti-bruit (dB)")
axes2[2].set_ylabel("Erreur (dB)")
axes2[2].set_xlabel("Fréquence (Hz)")

# Lignes verticales pour repérer 300/400/500 Hz
for ax in axes2:
    for f, c in zip([300, 400, 500], ['gold', 'orange', 'red']):
        ax.axvline(f, color=c, lw=0.8, linestyle=':', alpha=0.7)
    ax.set_ylim(-80, 10)
    ax.grid(True, alpha=0.3)

# Légende fréquences cibles
axes2[0].legend(['Signal', '300 Hz', '400 Hz', '500 Hz'],
                 loc='upper right', fontsize=7)

# Annotation fréquence dominante
annot_freq = axes2[0].annotate("", xy=(0, 0),
                                xytext=(0.6, 0.85),
                                textcoords='axes fraction',
                                fontsize=9, color='darkred',
                                fontweight='bold')

# ================================================================
# Bouton Pause (commun aux deux figures via fig1)
# ================================================================
ax_btn = fig1.add_axes([0.4, 0.02, 0.2, 0.05])
btn = Button(ax_btn, 'Pause', color='lightgray', hovercolor='silver')

# Slider µ sur fig2
ax_slider = fig2.add_axes([0.15, 0.06, 0.7, 0.03])
slider_mu = Slider(ax_slider, 'µ', 0.00001, 0.01,
                   valinit=mu_current, valfmt='%.5f')
slider_mu.label.set_fontsize(9)

def update_mu(val):
    global mu_current, w
    mu_current = slider_mu.val
    # Reset des poids quand µ change (nouvelle convergence)
    w[:] = 0
    print(f"[µ changé] Nouveau µ = {mu_current:.6f} — poids remis à zéro")

slider_mu.on_changed(update_mu)

def toggle_pause(event):
    global paused
    paused = not paused
    btn.label.set_text('Reprendre' if paused else 'Pause')
    fig1.canvas.draw_idle()

btn.on_clicked(toggle_pause)

# ================================================================
# Fonctions utilitaires
# ================================================================
def compute_fft_db(signal):
    """FFT en dB, fenêtrée (Hann) pour réduire les lobes secondaires."""
    window = np.hanning(FFT_SIZE)
    if len(signal) < FFT_SIZE:
        signal = np.pad(signal, (0, FFT_SIZE - len(signal)))
    windowed = signal[:FFT_SIZE] * window
    spectrum = np.abs(np.fft.rfft(windowed)) / FFT_SIZE
    spectrum = np.maximum(spectrum, 1e-10)
    return 20 * np.log10(spectrum)

def estimate_dominant_freq(fft_db, freqs, mask):
    """Trouve la fréquence dominante sous 1000 Hz."""
    idx = np.argmax(fft_db[mask])
    return freqs[mask][idx]

def estimate_phase_shift(ref, anti):
    """Déphasage via corrélation croisée."""
    if np.max(np.abs(ref)) < THRESHOLD:
        return None
    corr = np.correlate(ref - np.mean(ref), anti - np.mean(anti), mode='full')
    lag = np.argmax(np.abs(corr)) - (len(ref) - 1)
    return lag

# ================================================================
# Fonctions d'animation
# ================================================================
def update_temporal(_):
    if paused:
        return line_ref, line_anti, line_err, line_sum, line_sup_ref

    r = np.array(buf_ref)
    a = np.array(buf_anti)
    e = np.array(buf_err)
    s = np.array(buf_sum)

    line_ref.set_ydata(r)
    line_anti.set_ydata(a)
    line_err.set_ydata(e)
    line_sum.set_ydata(s)
    line_sup_ref.set_ydata(r)

    for ax, data in zip(axes1, [r, a, e, s]):
        peak = max(np.max(np.abs(data)), 0.01)
        ax.set_ylim(-peak * 1.2, peak * 1.2)

    lag = estimate_phase_shift(r, a)
    if lag is not None:
        annot_phase.set_text(f"Décalage : {lag} éch. ({lag/FS*1000:.2f} ms)")
    else:
        annot_phase.set_text("Signal trop faible")

    return line_ref, line_anti, line_err, line_sum, line_sup_ref

def update_fft(_):
    if paused:
        return line_fft_ref, line_fft_anti, line_fft_err

    r = np.array(buf_ref)
    a = np.array(buf_anti)
    e = np.array(buf_err)

    fft_r = compute_fft_db(r)
    fft_a = compute_fft_db(a)
    fft_e = compute_fft_db(e)

    line_fft_ref.set_ydata(fft_r[freq_mask])
    line_fft_anti.set_ydata(fft_a[freq_mask])
    line_fft_err.set_ydata(fft_e[freq_mask])

    # Fréquence dominante détectée
    f_dom = estimate_dominant_freq(fft_r, freqs, freq_mask)
    annot_freq.set_text(f"Fréq. dominante : {f_dom:.0f} Hz")

    return line_fft_ref, line_fft_anti, line_fft_err

# ================================================================
# Lancement
# ================================================================
try:
    stream = sd.Stream(
        device=(0, 0),
        channels=2,
        callback=callback,
        samplerate=FS,
        blocksize=BLOCK_SIZE,
        dtype='int32'
    )
    stream.start()
    print("=" * 55)
    print("  ANC Temps Réel — TIPE Alexandre Lhommedet")
    print("=" * 55)
    print(f"  FS={FS} Hz | BLOCK={BLOCK_SIZE} | N_TAPS={N_TAPS}")
    print(f"  µ initial = {mu_current} (modifiable via slider)")
    print(f"  Seuil activation = {THRESHOLD}")
    print("=" * 55)
    print("  Fenêtre 1 : signaux temporels")
    print("  Fenêtre 2 : spectres FFT + slider µ")
    print("=" * 55 + "\n")

    ani1 = animation.FuncAnimation(
        fig1, update_temporal, interval=50,
        blit=True, cache_frame_data=False
    )
    ani2 = animation.FuncAnimation(
        fig2, update_fft, interval=100,
        blit=True, cache_frame_data=False
    )

    plt.show()

finally:
    stream.stop()
    stream.close()
    print(f"\nArrêt propre. ||w||={np.linalg.norm(w):.5f}")
