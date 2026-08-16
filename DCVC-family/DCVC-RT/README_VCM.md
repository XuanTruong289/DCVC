# DCVC-RT for Video Coding for Machines

This project adapts DCVC-RT for machine-oriented video coding.

## Backbone

DCVC-RT:
- Image codec: frozen
- Video codec: trainable

## Machine task

YOLOv5 object detection.

## Training objective

L = R_DCVC-RT + lambda_machine * D_feature

where:

- R_DCVC-RT is the estimated bitrate from the DCVC-RT entropy model.
- D_feature is the MSE between original YOLO features and features extracted from reconstructed frames.

## Training

- Dataset: Vimeo-90K Septuplet
- QP: random in [0, 63]
- lambda_machine: {2, 4, 8, 16}
- Enhancement Layer: removed