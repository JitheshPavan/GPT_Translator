# GPT Translator 
Implementing transformer Architecture from scratch in PyTorch as given in the paper- ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)

# Script
In the model folder, the model is explained with detailed comments.

Pre_LN_Transformer is a complete script with training 

- [Training](#Training)
  - [Inference](#Inference)
- [Theory](#Theory)
  - [Attention](#Attention)
  - [Why Transformer Works](#Why-Transformer-works)
  - [Masking](#Masking)  
  - [MultiHead Attention](#MultiHead-Attention)  <!-- Match with the section header's ID -->
  - [Transformer](#Transformer)  <!-- Lowercase 't' to match section ID -->
    - [Encoder](#Encoder)  <!-- Space added, lowercase 'e' -->
    - [Decoder](#Decoder)  <!-- Space added, lowercase 'd' -->
  - [Learning Rate](#Learning-rate)  <!-- Lowercase, hyphen instead of underscore -->
  - [Input](#Input)  <!-- Lowercase 'i' -->
  - [BLEU Score](#BLEU-Score)  <!-- Lowercase, hyphen instead of underscore -->
  - [Positional Encoding](#Positional-Encoding)  <!-- Lowercase, hyphen instead of underscore -->

## Training
In the original paper, the authors used warmup steps during training. The output gradients of the original transformer are huge. So, we use warmup to gradually build up to the learning rate so that that model can stabilize. But, if we shift the position of the Layernorm layer to behind the multi-head attention rather than after the residual layer, gradients remain stable. So you don't need a warmup for the latter. Note: both types of transformers can achieve the same accuracies. One is significantly faster.

<p align="center">
  <img src="https://github.com/JitheshPavan/GPT_Translator/blob/main/data/loss.png" width="400"/>
  <img src="https://github.com/JitheshPavan/GPT_Translator/blob/main/data/bleu.png" width="400"/>
</p>
number of parameters in my model is 32790680.

### Inference
An example of successful translation is 'How are you,' translated to [wie,geht, es, dir]. 

A failed case: [what are you doing] translates to [was geht es gut], which, when translated back, means "what is going well".

## Theory
### Attention
<p align="center">
  <img src="https://github.com/JitheshPavan/GPT_Translator/blob/main/data/attention.png" />
</p>

The attention mechanism captures the relations between different tokens or words in sequence. Input for attention looks like- [Batch, Tokens, Embeddings]. Tokens are generated from the sentence and embedded into a dimensional space. So, each word/token will have a representative learnable vector. Each token generates three vectors in the first part: Keys, Query, and Value. The Key vector encodes the information contained in a token, while the Query vector represents what information the token seeks. The Value vector maps the information resulting from Key-Query interactions to the relevant tasks. 

First, the information is generated using a dot product between queries and keys. In the output matrix, each row represents the interaction between a query and all the keys. Next, the softmax function is applied to each row of the dot product output to normalize the values into probabilities that sum to 1. Finally, multiply the resulting attention weights with the Value matrix to compute the output.

**Note:**   
- The first matmul has the dimension ==> [T, Q] * [K, T] ==> [T, T]. So, the dimensions of the query and keys should match. Q=K
- These multiplication works irrespective of the sequence length (T). Thus, the attention mechanism can have an infinite context.
- Which input matrix works as the key and query is determined by which side the value token is multiplied from.
- Matrix dimension for the final computation goes as [T, T] * [T, Embedding_dim] ==> [T, embedding_dim]. We have the original input dimensions back ( As long as the dimension size of the value matrix equals the dimension of the input). This multiplication is also a convex linear combination of individual scalars of values vector.
- When sequence lengths differ (e.g., Query has 𝑇1 tokens and Key has T2 tokens), the dot product produces a [𝑇1,T2] matrix. The attention mechanism remains valid if the Value matrix is sized [T2, E] as the final computation [T1, T2] * [T2, E] results in a [T1, E] output.

### Why Transformer works

Unique problems faced in Natural Language Processing (NLP) are complexities of language, such as grammar and varying sentence length. Transformers were designed to address this problem.

1)Every operation except attention is applied to the tokens independently of each other. These operations work on the embedding dimension of the token. The embedding dimension size is constant; therefore, the usual neural network layers can be used here. 

2)The attention mechanism is a form of inner product. This operation is defined because the number of columns in the first matrix and the number of rows in the second matrix are equal. No constraint is placed on the number of tokens. Therefore, the attention mechanism is independent of the number of tokens. 

## Masking
1) Although the transformer can handle different sequence lengths, it is advisable to use the same sequence length in a batch; this makes it easier to train it using GPUs. Such a padded sequence should not be considered, so we introduce padding_mask. This is after the Q.dot(K) mechanism sets all the padded values in the sequence to -infinity. The exponential of -infinity is zero, thus removing it from softmax calculations. 

2) "Look-Ahead Mask"-->This preserves auto-regressive property. The transformer generates a new token for every turn during inference or generation. The transformer( decoder in particular) can only access the tokens it has generated before. This property must be mimicked during training, so we added a look-ahead mask. During the output calculations, queries should not access any keys of the future token. So, we add a look-ahead mask. The upper triangular mask with infinity for every non-zero value acts as the mask. The look-ahead mask remains an upper triangular matrix, but the size increases with every new token generation. The token generated is added to the input during the next generation cycle.

3) The encoder only requires a padding mask. In a Decoderblock, the first Attention block requires a look_ahead and padded mask. The second block requires only a padded mask. A look-ahead mask should not be used because keys come from the encoder. This can be understood in the context of Machine Translation, as the transformer will always have access to the complete sentence that needs to be translated. This sentence is fed into the encoder.

What does a look_ahead mask + padding mask look like?
Ex: sequence=[1,2,0,0,0] , then mask will look like=[[T1,T2,0 0,0],[T3,T2,0,0,0],-----]. This mask is used during the first Attention block in the Decoder

## MultiHead Attention:

MultiHead Attention, as the name suggests, multiple attention is done in parallel. Keys, queries, and values received from the input are indexed into distinct matrices in the embedding dimensions. Attention is carried out independently for every set of matrices. 
Thus, with the same computational complexity, we can capture context many times. At the end, the output matrices are combined to achieve the same output. The main point is that K, Q, and V are indexed in the embedding dim, not the context/time/token dimension. This preserves the ability to access every context.

## Transformer
<p align="center">
  <img src="https://github.com/JitheshPavan/GPT_Translator/blob/main/data/transformer.png" width="400" />
</p>
<p align="center"><i>Transformer architecture</i></p>

### Encoder
The encoder consists of attention and FFNN with residual and layer norm connections. We start with the attention block, which acts as the residual input to the input. Layernorm is applied to the output(x+attention(x)). Then, FFNN is applied as another residual connection with another layer of the norm layer at the end. The FFNN consists of two linear layers. One maps emb_dim to 4* emb_dim, and the layer maps it back to the original dimension. The encoder part is used for the input if we have eng to ger translation. Eng sentence is the input to Encoder. The encoder thus does not require a look_ahead mask as the whole eng sentence should be accessed for translation in every German token generation. Dropout is added after every sub-layer, after the attention and FFNN layer.

### Decoder 

The decoder works similarly except for an MHA layer in between, which receives the keys and values from the encoder. If we implement a text generation model without any input, then we would not need an encoder. Thus, the transformer/decoder block would more resemble an encoder. 


## Learning Rate

![](https://github.com/JitheshPavan/GPT_Translator/blob/main/data/lr_formula.PNG)

The learning rate increases with warm-up steps and then gradually decreases. The bigger the warmup rate, the smaller the slope and peak in the lr. 
With increased dimension of the model, the learning rate decreases. It decreases by 10 points for every 100 increase in dimension. 
<img src="https://github.com/JitheshPavan/GPT_Translator/blob/main/data/learning_rate.png" width="800" height="500">

### Input
We add two tokens to each input: A beginning-of-sequence token and an end-of-sequence token. Doing so gives model markers/ boundaries to avoid stopping too early or blabbering endless tokens. It helps inference, where the model outputs these tokens, which means the generation is over.

The transformer predicts one token at a time. Suppose we have decoder input as "`<START>` How are you? `<END`". Decoder input and output follow this pattern. 

First iteration="`<START>`" --> "How".

Second Iteration=" `<START>` How" --> "are" 

Third Iteration ="`<START>` How are" --> "you?"

Fourth Iteration= "`<START>` How are you" -->" "`<END>`"

So, we need to shift our decoder output expectation to the right. Thus, we get for a token of 5, 4 examples. This is called Shifted Inputs.

### BLEU Score

Machine translation can have multiple correct answers. The output may not be similar to the input with respect to the words and their positions, but that does not make it a bad translation. So, we calculate the precision by taking single, bigram, trigram, and tetragram pairs from the outputs. We then compare how many times each pair appears in our target. We have the number of times a pair appears in the target (clipped so that it does not exceed the denominator) divided by the number of times the pair appears in the prediction. The geometric mean is taken over the numbers received from all the pairs of different sizes. We also add a penalty if the length of the prediction is smaller than the length of the target.

### Positional Encoding

The attention mechanism is independent of the length of tokens; it is also independent of order. It cannot know the order of the tokens. Why is it so?

1) The attention mechanism works on the embedding space, not the tokens. That is, throughout the ordeal, the last dimension is worked on. So, it works on every token independently, just like the batch size. Even at the end, the last neural network layer maps the embedding to vocabulary size done on the embedding dimension. That is why it is independent of the order. So, if you input two tokens to the decoder, it will output two tokens after working on both tokens independently. Now, you may think that since it works on both inputs independently, context is non-existent. However, tokens interact only during the attention mechanism and during that mechanism. This fuels its infinite attention span. This interaction captures the relation between the tokens, not the order of the tokens.
2) We use a positional encoding vector to capture the order of tokens. The positional encoding vector displaces the embedding vector in a specific direction. Thus, positional encoding can be considered an attempt to group the first tokens together. Thus, we have two opposing forces during encoding: embedding vector, which attempts to put similar vectors in a similar cluster, and positional encoding vector, which attempts to group tokens together based on their position. The positional embedding vector has to follow two properties.
- it should not get large in magnitude as the number of tokens increases. This will cause the embedding vector to grow small in influence. 
- The positional vector should not get smaller in magnitude as the token size increases; the embedding vector will overshadow the positional vector.
3) The trigonometric functions sin follows the first property. But it is periodic. That is why we decrease its frequency so that it does not repeat itself. But if you do that, it will slowly decrease on one end. That is why we alternate between cosine and sine functions. When the sin value is small, the cosine value is large, and vice-versa. So, the second option is satisfied.
  
- **Positional Encoding Equation:**
  
PE(pos, 2i) = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
Where:
- `pos` is the position of the token in the sequence.
- `i` is the dimension of the embedding.
- `d_model` is the total dimensionality of the embeddings.

**Note**
- The frequency remains the same with changes in position. That is, we get a constant frequency wave in the direction of position.
- The frequency decreases for the direction in dimension, leading to slower oscillations. Thus, we get a wave that is getting longer as we go deeper into the embedding dimension.
- The larger the d_model value, the faster the change in frequency.

Thus, we have a plot like this

<img src="https://github.com/JitheshPavan/GPT_Translator/blob/main/data/scaling.png" width="800" height="500">

How do we infer from this image? The usual sin wave is depicted in this image through colors. The light color represents the peak, and the dark color represents the negative peak. Now, if you look at the columns, spacing between the same colors does not change indicating no change in freq. Rowwise, the spacing increases the deeper you go, indicating increased frequency. Now, this change can be inferred as a decrease in magnitude for the sin wave, but since we are using the cos wave, alternatively, the effect is counteracted. This is why freq must change with dimension; the sin wave follows the cosine wave. If we do not change the frequency, the end and beginning values come out the same.
