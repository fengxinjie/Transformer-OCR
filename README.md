# Scene Text Recognition via Transformer
Implementation of Transformer OCR  as described at [Scene Text Recognition via Transformer](https://arxiv.org/abs/2003.08077).

## model architecture:

<img src="art.png" width = 60%  div align=center />

## Results across a number of methods and datasets:

<img src="result.png" width = 60%  div align=center />

## Heat map of the source attention (encoder memory) score of the first layer of decoder:

<img src="heatmap.png" width = 70%  div align=center />

## Pretrained model for IC15 dataset
We upload a pretrained model for IC15 dataset. Due to the file size limitation of the github, we split the model into two files. One need to merge these two files into one file for reuse.

use cmd: cd checkpoints && cat IC1500 IC1501 > IC15.zip && unzip IC15.zip

## The transformer source code from:http://nlp.seas.harvard.edu/2018/04/03/attention.html
