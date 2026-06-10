"""VAP-style multiparty turn-taking prediction.

Implements the project's core modelling decision: predict each participant's
future voice activity over discretised horizon bins (Ekstedt & Skantze VAP,
extended to 3-4 party settings), instead of a hand-labelled TRP classifier.
Labels are derived automatically from per-participant gold VAD streams.
"""
