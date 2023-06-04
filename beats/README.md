## Using BEATs

Overview: https://paperswithcode.com/paper/beats-audio-pre-training-with-acoustic
Paper: https://arxiv.org/abs/2212.09058v1
Implementation reference: https://github.com/microsoft/unilm/tree/master/beats

I looked for a more state-of-the-art (SOTA) model for the Audio Classification task and arrived at this model in particular because it was the top-scoring _audio-only_ model for the [AudioSet](https://paperswithcode.com/dataset/audioset)/[ESC-50](https://paperswithcode.com/dataset/esc-50) datasets.

Because it is a pytorch model, I will try 2 approaches

1. Running inference on the model as it is

2. Converting it to ONNX -> Tensorflow -> tflite for a less energy-intensive monitoring background application, without sacrificing accuracy for my task.
