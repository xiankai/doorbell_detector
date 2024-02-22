## Using BEATs

Overview: https://paperswithcode.com/paper/beats-audio-pre-training-with-acoustic
Paper: https://arxiv.org/abs/2212.09058v1
Implementation reference: https://github.com/microsoft/unilm/tree/master/beats

I looked for a more state-of-the-art (SOTA) model for the Audio Classification task and arrived at this model in particular because it was the top-scoring _audio-only_ model for the [AudioSet](https://paperswithcode.com/dataset/audioset)/[ESC-50](https://paperswithcode.com/dataset/esc-50) datasets.

Because it is a pytorch model, I will try 2 approaches

1. Running inference on the model as it is

2. Converting it to ONNX -> Tensorflow -> tflite for a less energy-intensive monitoring background application, without sacrificing accuracy for my task.

## Setup - I use an Orange Pi 3 (OPi3)

### 1. Prerequisites
```
sudo apt-get install libportaudio2
git clone https://github.com/xiankai/doorbell_detector
```

### 2. Setup venv (optional)
```
sudo apt install python3.10-venv
python3 -m venv venv
source venv/bin/activate
```

### File setup (on OPi3)
```
pip3 install -r requirements.txt
cp .env.example .env
```

### File setup (on host machine)
Label.csv file was created from the output of [the tokenization process](https://github.com/microsoft/unilm/blob/master/beats/README.md#load-tokenizers)
Checkpoint file was downloaded from the [BEATS public repository](https://github.com/microsoft/unilm/blob/master/beats/README.md#pre-trained-and-fine-tuned-tokenizers-and-models)
Replace <ORANGEPI_IP> with device IP.
```
scp beats/class_labels_indices.csv orangepi@<ORANGEPI_IP>:~/doorbell_detector/beats
scp beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt orangepi@<ORANGEPI_IP>:~/doorbell_detector/beats
```

## Running in the background

I just use `nohup`.

`nohup python3 monitor_with_beats.py &`
