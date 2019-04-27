# BirdWatchin' 🐦

Bird Species Classifier built with Fast.ai

- Uses the NABirds Dataset with over 350+ species of birds found all over North America and Europe
- Uses FastAI that works on the PyTorch Deep Learning framework
- Accuracy of about 70%
- Refer to [birdwatchin.ipynb](./birdwatchin.ipynb)

## Gathering Dataset

```bash
curl -L -O narbirds.tar.gz https://www.dropbox.com/s/nf78cbxq6bxpcfc/nabirds.tar.gz?oref=e&n=13142758&submissionGuid=6bf6ac38-47ea-4e21-bcc7-91a0037d6027

mkdir data

tar -xzf nabirds.tar.gz -C data/
```
