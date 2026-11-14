# The Gángan Never Lies

**Category:** Audio forensics / steganography  **Difficulty:** Easy  **Flag format:** `SIGMOID_<WORD>`

**Flag: `SIGMOID_TALKINGDRUM`**

## Problem statement

SIGMOID's archive team is digitizing old field recordings. One file, `challenge.wav`, is labelled "test tone - ignore" and was left behind by a retired sound engineer who never wrote anything down. It sounds like a flat, buzzing hiss. The job is to work out what is hidden in it and submit it as the flag.

Input: `challenge.wav`, 16-bit PCM, mono, 22050 Hz, about 10.17 s (224,256 samples).

## Approach

The file is noise to the ear but the energy is not spread evenly across frequency. Plotting it as a spectrogram (time on x, frequency on y, brightness as power) shows a block of text written into the spectrum.

1. Read the samples and compute a spectrogram (1024-sample window, 87.5% overlap).
2. Average each frequency bin over time. Everything sits at roughly the same level except one narrow band, which stands out.
3. Crop the plot to that band (about 3.1-3.5 kHz) and look at it. The letters read `TALKINGDRUM`.
4. Wrap it in the required format: `SIGMOID_TALKINGDRUM`.

## How it works

A spectrogram is just an image whose pixels are power at a given time and frequency. If you build that image first (draw the text, then work backwards to audio), you get a signal that sounds like noise but looks like writing when analysed. Each pixel becomes a short burst of sinusoids at that row's frequency, so the text survives any viewer that shows a spectrogram. It is a good fit for the story: the drum doesn't play a rhythm, it speaks one, and here the "tone" is speaking in the frequency domain instead of the time domain.

The text sits at 3-3.5 kHz, well inside the audible range, so it is hidden by being drowned in noise rather than by going ultrasonic.

## Hiding places that were checked

The story hints the engineer hid it somewhere nobody would look, so I ruled out the usual spots before settling on the spectrogram:

| Place | Result |
|---|---|
| WAV header / trailing bytes | Clean. File size matches the declared `data` chunk exactly, no extra chunks, no readable strings. |
| Sample LSBs | The first bytes of the LSB stream look random, no readable text. |
| Broadband spectrum | One band is clearly hotter than the rest (about 3.1-3.5 kHz). |
| Spectrogram image | Text `TALKINGDRUM`, readable on a plain render. |

## Installation and execution

Python 3.9+.

```bash
pip install -r requirements.txt
python solve.py challenge.wav spectrogram.png
```

`solve.py` prints the detected band and writes `spectrogram.png`, cropped to it. Open the image and read the letters. Both arguments are optional and default to the names above.

Any audio editor with a spectrogram view (Audacity, Sonic Visualiser) reproduces the same result with no code: set the view to spectrogram and the window to about 1024.

## Evidence

Running `solve.py` on the provided file:

```
hot band: 3101-3531 Hz
band is 4.2 dB above the noise floor
spectrogram saved to spectrogram.png, read the letters off it
```

`spectrogram.png` shows `TALKINGDRUM` spanning roughly 0.4 s to 9.7 s in the 3.1-3.5 kHz range. The text is a continuous band of bright pixels against flat noise and the letters are unambiguous, including the `G`, `D`, `R` and `M`.

## Known limitations

- The final reading step is visual. The script finds and crops the band but does not do OCR, so a human (or an OCR tool) has to read the letters.
- Band detection is a simple threshold (median + 3 dB). It works here because the noise is flat. A file with coloured noise or a quieter message would need a different threshold.
- The LSB check only looked at the start of the stream. It is a sanity check, not an exhaustive steganalysis.
- Spectrogram settings matter. With a very short window the letters smear horizontally, with a very long one they smear vertically. About 1024 samples at 22050 Hz works well.
