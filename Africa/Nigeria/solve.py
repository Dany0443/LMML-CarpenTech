import sys
import wave

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

wav_path = sys.argv[1] if len(sys.argv) > 1 else "challenge.wav"
png_path = sys.argv[2] if len(sys.argv) > 2 else "spectrogram.png"

with wave.open(wav_path) as wav:
    rate = wav.getframerate()
    samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(float)

freqs, times, power = signal.spectrogram(samples, rate, nperseg=1024, noverlap=896)
db = 10 * np.log10(power + 1e-9)

# the buzz is flat noise everywhere except one band
mean_db = db.mean(axis=1)
hot = freqs[mean_db > np.median(mean_db) + 3]
band_lo, band_hi = hot.min(), hot.max()
pad = 400
keep = (freqs > band_lo - pad) & (freqs < band_hi + pad)

plt.figure(figsize=(16, 4))
plt.pcolormesh(times, freqs[keep], db[keep], shading="auto", cmap="magma")
plt.xlabel("seconds")
plt.ylabel("Hz")
plt.tight_layout()
plt.savefig(png_path, dpi=110)

print(f"hot band: {band_lo:.0f}-{band_hi:.0f} Hz")
print(f"band is {mean_db[keep].mean() - np.median(mean_db):.1f} dB above the noise floor")
print(f"spectrogram saved to {png_path}, read the letters off it")
