r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=256,
        seq_len=128,
        h_dim=1024,
        n_layers=2,
        dropout=0.2,
        learn_rate=5e-4,
        lr_sched_factor=0.05,
        lr_sched_patience=0.5,
    )
    return hypers


def part1_generation_params():
    start_seq = "ACT I. SCENE I."
    temperature = 0.5
    return start_seq, temperature


part1_q1 = r"""
**Answer:**

The entire text is too large for the model to handle at once due to memory limits.
Working with sequences helps the model focus on the connections between nearby sentences
and makes it easier to pick up on relevant patterns without getting overwhelmed by irrelevant details.
It also prevents technical issues like vanishing or exploding gradients, which makes the training process smoother and more efficient.
"""

part1_q2 = r"""
**Answer:**

This happens because the model keeps a hidden state that carries information from earlier batches.
This lets it access context beyond the current sequence and incorporate it into the generated text.
The hidden state is updated at each step based on the current input and the previous hidden state,
which allows the model to maintain a coherent narrative throughout the generation process.
"""

part1_q3 = r"""
**Answer:**

We don't suffle the order of batches becuase the model relies on the sequential nature of the text to learn patterns and generate coherent text.
And because the we are given a text in English, which is a language that relies heavily on word order to convey meaning.
Thus the model needs to see the text in the correct order to learn how to generate text that makes sense.
"""

part1_q4 = r"""
**Answer:**

1.  The temperature parameter controls the randomness of the generated text.
    - A higher temperature increases the randomness, which can lead to more diverse but less coherent text.
    - A lower temperature makes the text more predictable and coherent but less creative.
    
    So lowering the temperature can help the model generate more coherent text that follows the patterns in the training data.
    
2.  When the temperature is very high, the probability distribution becomes more uniform, which means that all words are equally likely to be chosen, regardless of their scores.
    This can lead to more random and less coherent text because the model is less constrained by the patterns in the training data.
    
3.  When the temperature is very low, the probability distribution becomes more peaked, which means that the words with the highest scores are more likely to be chosen.
    This can lead to more predictable and coherent text because the model is more likely to choose the most probable words based on the training data.
    But it can also lead to repetitive output.

"""

# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = "https://github.com/AviaAvraham1/TempDatasets/raw/refs/heads/main/George_W_Bush2.zip"


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    hypers['batch_size'] = 16
    hypers['h_dim'] = 256
    hypers['z_dim'] = 32
    hypers['learn_rate'] = 1e-3
    hypers['betas'] = (0.9, 0.999)
    hypers['x_sigma2'] = 0.1
    return hypers


part2_q1 = r"""
**Answer:**
The $\sigma^2$ hyperparameter (`x_sigma2` in the code) controls the variance of the Gaussian distribution used in the decoder's likelihood function. 

- Low values of $\sigma^2$ make the model focus more on reconstructing the input data accurately, leading to sharper images but potentially overfitting.
- High values of $\sigma^2$ allow for more variability in the generated images, which can help in capturing the underlying data distribution better but may result in blurrier images.
"""

part2_q2 = r"""
**Answer:**
1. The VAE loss term consists of two parts: the reconstruction loss and the KL divergence loss.
   - The reconstruction loss measures how well the decoder can reconstruct the input data from the latent representation. It ensures that the generated data is similar to the input data.
   - The KL divergence loss measures the difference between the learned latent space distribution and the prior distribution (usually a standard normal distribution). It ensures that the latent space follows the desired distribution, allowing for meaningful sampling.

2. The KL loss term regularizes the latent space distribution to be close to the prior distribution. This prevents the latent space from becoming too irregular and ensures that it can be sampled from effectively.

3. The benefit of this effect is that it allows the VAE to generate new, meaningful samples by sampling from the latent space. The regularized latent space ensures that the generated samples are similar to the training data and follow the learned distribution.
"""

part2_q3 = r"""
**Answer:**
Maximizing the evidence distribution, $p(X)$, ensures that the model learns to generate data that is similar to the training data. By doing so, we ensure that the model captures the underlying data distribution and can generate new samples that are representative of the training data. This is crucial for the VAE to function as a generative model.
"""

part2_q4 = r"""
**Answer:**
In the VAE encoder, we model the **log** of the latent-space variance, $\sigma^2_{\alpha}$, instead of directly modeling the variance to ensure numerical stability and to avoid negative values. The log transformation ensures that the variance is always positive, which is a requirement for the Gaussian distribution. Additionally, it allows for a more stable and efficient optimization process.
"""

# Part 3 answers
def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0, discriminator_optimizer=dict(type="", lr=0.0, betas=(0.0, 0.0)),
        generator_optimizer=dict(type="", lr=0.0, betas=(0.0, 0.0)), data_label=0,label_noise=0.0
    )
    hypers["batch_size"] = 8
    hypers["z_dim"] = 128
    hypers["discriminator_optimizer"] = {
        "type": "Adam",
        "lr": 0.0002 ,
        "betas": (0.5, 0.999),
        "weight_decay": 1e-5,
    }

    hypers["generator_optimizer"] = {
        "type": "Adam",
        "lr": 0.0002,
        "betas": (0.5, 0.999),
        "weight_decay": 1e-5,
    }

    hypers["data_label"] = 0
    hypers["label_noise"] = 0.4
    return hypers

part3_q1 = r"""
**Answer:**

During GAN training, gradients are maintained when updating the parameters of either the Generator or the Discriminator.
When training the Generator, gradients are crucial because they indicate how changes in the Generator's
output affect the Discriminator's ability to classify fakes, allowing the Generator to improve.  Similarly,
when training the Discriminator, gradients are essential for the Discriminator to learn how to better distinguish real data from the Generator's
fakes.  However, once training is complete and the goal is simply to use the trained Generator to sample and create new data, 
gradients are discarded. In this phase, we're no longer adjusting the network's parameters, 
we're just leveraging the learned mapping from noise to data, making gradient calculation unnecessary and computationally wasteful.

"""

part3_q2 = r"""
**Answer:**

**1.** Stopping GAN training solely because the Generator loss is low is a bad idea. 
A low Generator loss can be misleading; it could mean the Generator is creating fantastic images,
but it could also mean the Discriminator isn't doing its job well, or that the Generator is only making a small variety of good images.
The goal is to find a balance where both the Generator and Discriminator are performing well and neither can easily improve without the 
other also improving. This balance is tricky to find directly.  Therefore, it's important to look at more than just the Generator loss.
Monitor both the Generator and Discriminator losses, and stop when they level off and aren't changing much. 

**2.** if the discriminator loss remains at a constant value while the generator loss decreases,
it suggests that the generator is improving and producing more realistic samples, 
but the discriminator is not adapting effectively to these changes. This could happen because the discriminator has reached a point where 
it can no longer distinguish between real and generated samples as effectively, or it may be stuck in a suboptimal state. 
The constant discriminator loss indicates that its performance is not improving, while the decreasing generator loss shows 
that the generator is becoming better at fooling the discriminator. This imbalance could lead to mode collapse or unstable training,
as the discriminator fails to provide meaningful feedback to the generator. 
"""

part3_q3 = r"""
**Answer:**


The main difference between the VAE and GAN-generated images is that VAE outputs appear blurry and smooth, 
while GAN outputs are sharper and more detailed. This difference is caused by how each model generates images:
VAEs optimize for a structured and continuous latent space by minimizing a reconstruction loss combined with a regularization term, 
which results in smooth but less detailed images. In contrast, GANs use an adversarial process where a generator competes against a discriminator, 
pushing the generator to produce more realistic and sharper images. However, while GANs generate high-quality images, 
they can introduce artifacts or suffer from mode collapse, producing less diverse outputs.

"""



PART3_CUSTOM_DATA_URL = "https://github.com/AviaAvraham1/TempDatasets/raw/refs/heads/main/George_W_Bush2.zip"


def part4_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )
    
    hypers['embed_dim'] = 128
    hypers['num_heads'] = 4
    hypers['num_layers'] = 4
    hypers['hidden_dim'] = 256
    hypers['window_size'] = 128
    hypers['dropout'] = 0.2
    hypers['lr'] = 1e-4
    return hypers



part4_q1 = r"""
**Answer:**

Stacking encoder layers with sliding-window attention broadens the context in the final layer by progressively integrating information,
much like CNNs expand receptive fields. Each layer captures local dependencies within its window, and by stacking layers,
subsequent layers attend to outputs that already incorporate contextual information from previous layers. 
This hierarchical processing effectively expands the receptive field of each layer, 
allowing the final layer to encompass a significantly broader context and capture longer-range dependencies, 
leading to a more comprehensive understanding of the input sequence.

"""

part4_q2 = r"""
**Answer:**

One variation to achieve a more global context with similar computational complexity to sliding-window attention (O(nw)) 
is Dilated Sliding-Window Attention.  Instead of attending to consecutive positions within the window, 
dilated attention introduces "gaps" by attending to positions with a certain "dilation rate."  For example,
with a dilation rate of 2, the attention window might look at positions i-2, i, and i+2 around the current position i, 
instead of i-1, i, and i+1 in standard sliding-window attention.  
This effectively increases the receptive field and allows each layer to capture broader context and longer-range dependencies 
without significantly increasing the computational cost, as the number of attended positions per query remains similar (window size w).
Stacking these dilated sliding-window attention layers further expands the global context captured in deeper layers, 
enabling the model to understand relationships across larger segments of the input sequence while maintaining efficiency.

"""

part5_q1 = r"""
**Answer:**

Fine-tuning DistilBERT worked better for sentiment analysis than starting from zero.
This is because DistilBERT already learned a lot about language from huge amounts of text before we used it.
This pre-learning gave it a strong base for understanding words and their context, 
which is better than a new model trained only for sentiment. Fine-tuning, especially training all parts of DistilBERT, 
let it adjust to the specific feelings in text. Both ways of fine-tuning DistilBERT got much better results (almost 80% accuracy) 
compared to training from scratch (around 69.5% accuracy). However, fine-tuning isn't always better. It depends on the task. 
Pre-trained models are great when the new task is similar to what they already learned. But for very different tasks,
training from scratch might be needed. This DistilBERT example shows how helpful pre-trained models can be for tasks like understanding sentiment.

"""

part5_q2 = r"""
**Answer:**

Fine-tuning only the middle layers while freezing both the initial and final layers is generally less effective and
often yields worse results compared to fine-tuning the last layers or the entire model.  
This is because pre-trained models are structured hierarchically: early and middle layers learn general, 
reusable features, while the final layers are specifically tailored for the original pre-training task's output. 
By freezing the last layers, you prevent the model from adapting its task-specific decision-making process to the new task. 
While fine-tuning middle layers can adjust the learned representations, the frozen last layers, designed for the pre-training task, 
might not optimally utilize these updated representations, leading to a mismatch and hindering performance on the new task.  
Although in rare cases of significant data distribution shifts this approach might offer some benefit, 
it's generally less robust and less likely to be successful for typical fine-tuning scenarios compared to focusing on the task-specific final layers.
"""


part5_q3= r"""
**Answer:**

The standard BERT architecture, in its original form, is not directly equipped for machine translation.
This limitation stems from BERT's inherent design as an encoder-only model, optimized for understanding and representing input sequences, 
not for generating new ones.  BERT's pre-training objectives, such as Masked Language Modeling and Next Sentence Prediction, 
are geared towards learning rich contextual representations within a single sequence.  For machine translation, however, 
the task is fundamentally different: it requires generating a target sequence in a different language conditioned on a source sequence.
BERT lacks the decoder component necessary for this sequential generation process, making it unsuitable for direct application to machine translation tasks.

To adapt BERT for machine translation, a significant architectural modification is essential: the addition of a decoder.  
This transformation results in an encoder-decoder architecture where BERT functions as the encoder, 
processing the source language tokens into contextualized embeddings.  A separate decoder component, typically a Transformer decoder,
is then introduced to generate the target language sequence token by token.  This decoder attends to the encoded source representations
from BERT and utilizes previously generated target tokens to predict the next token in the translated sequence.  Furthermore,
to optimize for translation performance, adjustments to pre-training are highly beneficial.  While BERT's original pre-training provides
valuable general language understanding, pre-training specifically on parallel corpora or with sequence-to-sequence objectives would better 
equip the model to learn cross-lingual mappings and generation patterns crucial for effective machine translation.


"""

part5_q4 = r"""
**Answer:**

 RNNs might be chosen over Transformers due to their inherent sequential processing which can be advantageous for certain tasks and data types. 
 RNNs naturally process data in a step-by-step manner, maintaining a hidden state that evolves sequentially, 
 making them conceptually simpler and potentially more intuitive for tasks where the temporal order of data is crucial and must be explicitly
 modeled at each step.  This sequential inductive bias can be beneficial in scenarios like real-time streaming data processing or tasks where
 the immediate past context is paramount, offering a more direct and potentially more interpretable approach compared to the parallel-by-design 
 Transformer architecture, even in its lightweight form.

"""

part5_q5 = r"""
**Answer:**

BERT's Next Sentence Prediction (NSP) task pre-trains the model to determine if two sentences follow each other in the original text. 
During training, BERT receives sentence pairs, half of which are consecutive ("IsNext"), and half are random pairs ("NotNext"). 
The model predicts "IsNext" or "NotNext" using the [CLS] token's output and optimizes this binary classification with cross-entropy loss.
Initially, NSP was included to foster sentence relationship understanding, believed necessary for tasks requiring cross-sentence reasoning.

However, current evidence suggests NSP is not a crucial pre-training component. 
While intended to enhance inter-sentence understanding, studies and model variants like RoBERTa and DistilBERT show that removing or omitting NSP
doesn't harm, and sometimes even improves, downstream task performance. This indicates Masked Language Modeling (MLM) is the more dominant task, 
implicitly capturing sufficient sentence-level context. Furthermore, NSP might be too simplistic for BERT, allowing it to learn superficial cues 
like topical consistency without truly grasping deeper inter-sentence relationships, thus limiting its added value to pre-training.
"""


# ==============
