# VideoLLM_suite
VideoLLM_suite is a collection of inference-only open-source video LLMs.

# Usage
Excepted.
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VideoLLaMA(model_path)
model.to(device)

input_path = "videos/input.mp4"
prompt = "Describe the video for details."

model.encode_video(input_path)
sentence = model.generate(prompt)
```

# Models
- [ ] : video LLaMA
- [ ] : video LLaMA 2
- [ ] : video Chat
- [ ] : video Chat 2
- [ ] : LITA
- [ ] : Video LLaVA
- [ ] : Chat-UniVi
- [ ] : LaViLa
- [ ] : MA-LMM
- [ ] : MovieChat
- [ ] : MovieChat+
- [ ] : Koala
- [ ] : LongVLM
- [ ] : MiniGPT-video
- [ ] : PLLaVA
- [ ] : ST-LLM
- [ ] : InternVideo2
- [ ] : ViLA
- [ ] : Video-ChatGPT
- [ ] : TimeChat

# Test
```
pytest tests
```

# Mypy + Ruff
```
mypy firefly
ruff check firefly
```