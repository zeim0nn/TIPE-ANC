"""import numpy as np
import sounddevice as sd

FS = 48000
BLOCK_SIZE = 512
MU = 0.001
N_TAPS = 64

w = np.zeros(N_TAPS, dtype=np.float64)
x_buffer = np.zeros(N_TAPS, dtype=np.float64)
block_count = 0

def callback(indata, outdata, frames, time, status):
    global w, x_buffer, block_count

    if status:
        print(f"[STATUS] {status}")

    # --- Debug : affiche les niveaux toutes les 50 trames ---
    block_count += 1
    if block_count % 50 == 0:
        ref_max = np.max(np.abs(indata[:, 0]))
        err_max = np.max(np.abs(indata[:, 1]))
        w_norm  = np.linalg.norm(w)
        print(f"[Block {block_count}] ref_max={ref_max:.0f}  err_max={err_max:.0f}  ||w||={w_norm:.6f}")

    # --- Détection du format réel (int32 ou float) ---
    if indata.dtype == np.int32:
        scale = 2147483647.0
    else:
        scale = 1.0  # sounddevice a déjà normalisé

    ref = indata[:, 0].astype(np.float64) / scale
    err = indata[:, 1].astype(np.float64) / scale

    output = np.zeros(frames, dtype=np.float64)

    for i in range(frames):
        x_buffer[1:] = x_buffer[:-1]
        x_buffer[0] = ref[i]

        y = np.dot(w, x_buffer)
        output[i] = -y

        w += 2.0 * MU * err[i] * x_buffer

    out_clipped = np.clip(output * 0.8, -1.0, 1.0)

    if indata.dtype == np.int32:
        out_final = (out_clipped * 2147483647.0).astype(np.int32)
    else:
        out_final = out_clipped.astype(np.float32)

    outdata[:, 0] = out_final
    outdata[:, 1] = out_final


try:
    with sd.Stream(
    device=(1, 1),  # index 1 pour input ET output
    channels=2,
    callback=callback,
    samplerate=FS,
    blocksize=BLOCK_SIZE,
    dtype='int32'
    ):
        print("ANC activé — surveillance des niveaux toutes les ~5s")
        print("Parle près du micro pour tester...\n")
        while True:
            sd.sleep(1000)

except KeyboardInterrupt:
    print("\nArrêt.")
except Exception as e:
    print(f"Erreur : {e}")

import numpy as np
import sounddevice as sd

FS = 48000
BLOCK_SIZE = 512
MU = 0.0005
N_TAPS = 64

w = np.zeros(N_TAPS, dtype=np.float64)
x_buffer = np.zeros(N_TAPS, dtype=np.float64)
block_count = 0

def callback(indata, outdata, frames, time, status):
    global w, x_buffer, block_count

    if status:
        print(f"[STATUS] {status}")

    block_count += 1

    # Un seul micro : canal 0 = signal d'erreur
    err = indata[:, 0].astype(np.float64) / 2147483647.0

    output = np.zeros(frames, dtype=np.float64)

    for i in range(frames):
        x_buffer[1:] = x_buffer[:-1]
        x_buffer[0] = err[i]  # Le signal d'erreur sert aussi de référence

        y = np.dot(w, x_buffer)
        output[i] = -y

        # LMS : minimise le signal résiduel capté
        w += 2.0 * MU * err[i] * x_buffer

    if block_count % 100 == 0:
        print(f"[Block {block_count}] err_rms={np.sqrt(np.mean(err**2)):.6f}  ||w||={np.linalg.norm(w):.4f}")

    out_clipped = np.clip(output * 0.8, -1.0, 1.0)
    out_int32 = (out_clipped * 2147483647.0).astype(np.int32)

    outdata[:, 0] = out_int32
    outdata[:, 1] = out_int32

try:
    with sd.Stream(
        device=(1, 1),
        channels=2,
        callback=callback,
        samplerate=FS,
        blocksize=BLOCK_SIZE,
        dtype='int32'
    ):
        print("ANC Feedback activé (1 micro)")
        print("Logs toutes les ~1s...\n")
        while True:
            sd.sleep(1000)

except KeyboardInterrupt:
    print("\nArrêt.")
except Exception as e:
    print(f"Erreur : {e}")"""

import numpy as np
import sounddevice as sd

FS = 48000
BLOCK_SIZE = 512
MU = 0.001
N_TAPS = 64

w = np.zeros(N_TAPS, dtype=np.float64)
x_buffer = np.zeros(N_TAPS, dtype=np.float64)
block_count = 0

def callback(indata, outdata, frames, time, status):
    global w, x_buffer, block_count

    if status:
        print(f"[STATUS] {status}")

    block_count += 1

    ref = indata[:, 0].astype(np.float64) / 2147483647.0
    err = indata[:, 1].astype(np.float64) / 2147483647.0

    output = np.zeros(frames, dtype=np.float64)

    for i in range(frames):
        x_buffer[1:] = x_buffer[:-1]
        x_buffer[0] = ref[i]

        y = np.dot(w, x_buffer)
        output[i] = -y

        w += 2.0 * MU * err[i] * x_buffer

    # Log toutes les ~2.5s
    if block_count % 200 == 0:
        ref_rms = np.sqrt(np.mean(ref**2))
        err_rms = np.sqrt(np.mean(err**2))
        print(f"[Block {block_count:5d}] ref_rms={ref_rms:.5f}  err_rms={err_rms:.5f}  ||w||={np.linalg.norm(w):.5f}")

    out_clipped = np.clip(output * 0.8, -1.0, 1.0)
    out_int32 = (out_clipped * 2147483647.0).astype(np.int32)

    outdata[:, 0] = out_int32
    outdata[:, 1] = out_int32

try:
    with sd.Stream(
        device=(1, 1),
        channels=2,
        callback=callback,
        samplerate=FS,
        blocksize=BLOCK_SIZE,
        dtype='int32'
    ):
        print("ANC Temps Réel activé — device=(1,1)")
        print("Logs toutes les ~2.5s\n")
        while True:
            sd.sleep(1000)

except KeyboardInterrupt:
    print(f"\nArrêt. Poids finaux : ||w||={np.linalg.norm(w):.5f}")
except Exception as e:
    print(f"Erreur : {e}")
