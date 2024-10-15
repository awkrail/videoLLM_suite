# firefly
Yet another multimodal video feature extractor.

# Features
- unimodal: audio-only, visual-only
- multimodal: audio, visual, text
- multi GPU: multiple GPU supports
- multilingual: english, japanese VLM backbones
- synchronization: same-dimensional audio-visual feature (sequence length should be same)

# Models
### Vision-only
- [ ] : TIMM models (e.g., ViT, EVA, Swin, DINO, etc)

Action
- [ ] : I3D
- [ ] : Slowfast
- [ ] : VideoMAE

Optical flow
- [ ] : RAFT

Audio-only
- [ ] : PANNs
- [ ] : VGGish

Image-text
- [ ] : CLIP
- [ ] : Japanese CLIP

Video-text
- [ ] : CLIP4Clip
- [ ] : InternVideo

Audio-text
- [ ] : CLAP

# Test
```
pytest tests
```

# Mypy + Ruff
```
mypy firefly
ruff check firefly
```