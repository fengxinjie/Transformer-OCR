Due to my negligence, I did not carefully check the Harvard source code before, and I was not aware of the gradient leakage problem in the training process. Therefore, the test accuracy in my arXiv paper is not reliable. I have withdrawn the paper and will modify the code for retraining later. The bug has been fixed and uploaded.
## model architecture:

<img src="art.png" width = 60%  div align=center />

## Heat map of the source attention (encoder memory) score of the first layer of decoder:

<img src="heatmap.png" width = 70%  div align=center />

## The transformer source code from:http://nlp.seas.harvard.edu/2018/04/03/attention.html
