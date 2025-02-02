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
    hypers['batch_size'] = 32
    hypers['h_dim'] = 512
    hypers['z_dim'] = 64
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
    hypers["batch_size"] = 16
    hypers["z_dim"] = 128
    hypers["discriminator_optimizer"] = {
        "type": "Adam",
        "lr": 0.0003,
        "betas": (0.5, 0.999),
    }

    hypers["generator_optimizer"] = {
        "type": "Adam",
        "lr": 0.0003,
        "betas": (0.5, 0.999),
    }

    hypers["data_label"] = 0
    hypers["label_noise"] = 0.4
    return hypers

part3_q1 = r"""
**Your answer:**


"""

part3_q2 = r"""
**Your answer:**


"""

part3_q3 = r"""
**Your answer:**



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

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    
    # ========================
    return hypers




part3_q1 = r"""
**Your answer:**

"""

part3_q2 = r"""
**Your answer:**


"""


part4_q1 = r"""
**Your answer:**


"""

part4_q2 = r"""
**Your answer:**


"""


part4_q3= r"""
**Your answer:**


"""

part4_q4 = r"""
**Your answer:**


"""

part4_q5 = r"""
**Your answer:**


"""


# ==============
