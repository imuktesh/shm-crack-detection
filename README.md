---
title: Structural Health Monitoring
emoji: 🏗️
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# 🏗️ AI Structural Health Monitoring

Two-stage crack analysis pipeline:
1. Crack / no-crack classification (ResNet18, SDNET2018)
2. Crack segmentation (U-Net ResNet34, AfriCrack) at 512×512 with TTA
3. Structural vs non-structural heuristic (width, orientation, branching, length)

Trained models: `best_resnet18_sdnet.pth`, `best_unet_v2.pth`
