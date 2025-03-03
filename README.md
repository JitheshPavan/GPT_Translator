# GPT Translator 
Implementing transformer Architecture from scratch in PyTorch as given in the paper- ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)

I have published a blog on transformers explaining things I learned from this project. [Medium Blog](https://medium.com/@jitheshdsouza19/decoding-transformers-how-attention-makes-them-work-4a91a3fe2371)

Check it out to understand how attention works and to know about technical details of simple transformer training.

# Script
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/JitheshPavan/GPT_Translator/blob/main/Pre_LN_Transformer.ipynb)

If you want the model code with explanation check out files in model folder. 

Pre_LN_Transformer is the complete script with training. 

## Training
In the original paper, the authors used warmup steps during training. The output gradients of the original transformer are huge. So, we use warmup to gradually build up to the learning rate so that that model gradients can stabilize. But, if we shift the position of the Layernorm layer to the behind of multi-head attention layer rather than after the residual layer, gradients remain stable. So you don't need a warmup for the latter. Note: both types of transformers can achieve the same accuracies. One is significantly faster. 

<p align="center">
  <img src="https://github.com/JitheshPavan/GPT_Translator/blob/main/data/loss.png" width="400"/>
  <img src="https://github.com/JitheshPavan/GPT_Translator/blob/main/data/bleu.png" width="400"/>
</p>
number of parameters in my model is 3,27,90,680.

### Inference
An example of successful translation is 'How are you,' translated to [wie,geht, es, dir]. 

A failed case: [what are you doing] translates to [was geht es gut], which, when translated back, means "what is going well".
