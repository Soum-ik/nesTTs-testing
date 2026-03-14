from neutts import NeuTTS
import soundfile as sf

# Load the model
tts = NeuTTS(
    backbone_repo="neuphonic/neutts-nano",   # Full precision
    backbone_device="cpu",
    codec_repo="neuphonic/neucodec",
    codec_device="cpu",
)

# Your text to synthesize
input_text = "Hello! This is NeuTTS-Nano speaking in your application."

# Reference audio for voice cloning (3+ seconds of any voice)
ref_text_path = "samples/jo.txt"
ref_audio_path = "samples/jo.wav"

ref_text = open(ref_text_path, "r").read().strip()
ref_codes = tts.encode_reference(ref_audio_path)

# Generate speech
wav = tts.infer(input_text, ref_codes, ref_text)

# Save to file
sf.write("output.wav", wav, 24000)
