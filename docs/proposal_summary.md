# Proposal Summary

## Project Aim

Build a hybrid multimodal framework that allows a social robot, eventually Furhat, to predict appropriate turn-entry moments in multi-party conversations.

## Core Research Problem

Reactive silence-based turn-taking causes unnatural delays and fails during overlapping speech. The proposed solution replaces fixed silence thresholds with multimodal prediction of Transition Relevance Places.

## Planned Modalities

- Audio Direction of Arrival using GCC-PHAT.
- Visual attention and gaze vectors.
- Visual Voice Activity Detection using Mouth Aspect Ratio from face landmarks.
- Fused time-series features for TRP prediction.

## Initial Engineering Roadmap

1. Build a repeatable Python project framework.
2. Implement and test audio and visual feature extraction primitives.
3. Create a schema for fused frame-level features and annotations.
4. Validate the full pipeline on simulated or recorded sample data.
5. Train supervised models, starting from classical baselines and then LSTM/VAP-style models.
6. Integrate the predictive decision layer with Furhat or ROS2 once hardware paths are fixed.
