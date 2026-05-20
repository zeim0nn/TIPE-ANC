import numpy as np
import sounddevice as sd

print(sd.query_devices())
print("\nDevice 1 détaillé:")
print(sd.query_devices(1))

print("\n=== Test enregistrement 3 secondes (parle près du micro) ===")
rec = sd.rec(48000 * 3, samplerate=48000, channels=2, dtype='int32', device=1)
sd.wait()

print(f"Max canal 0 : {np.max(np.abs(rec[:, 0]))}")
print(f"Max canal 1 : {np.max(np.abs(rec[:, 1]))}")
print(f"dtype       : {rec.dtype}")                                                                                
