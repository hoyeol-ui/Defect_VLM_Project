# Defect VLM Active Learning Project

This repository contains experimental code for VLM-based active learning sample selection for industrial defect detection.

## Research Goal

The goal is to select informative industrial defect images for annotation under a GT-free acquisition setting.

The pipeline uses:
- VLM-generated defect explanations
- SBERT-based semantic consistency
- OWL-ViT pseudo bounding boxes
- pseudo groundedness
- active learning sample selection
- YOLOv8 detector-level validation

## Main Strategies

- Random
- ConsistencyOnly
- GroundednessOnlySoft
- CombinedSoftPenalty
- LowPrioritySoft

## Important Note

Ground-truth bounding boxes are not used during acquisition scoring.
GT annotations are only used after sample selection to simulate annotation and train the YOLO detector.

## Current Status

The GT-free acquisition and detector-level validation pipeline has been implemented.
The current issue is the acquisition score direction, especially why LowPrioritySoft performs strongly in some experiments.