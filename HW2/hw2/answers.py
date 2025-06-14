r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**


**1.A:**

The output $\mathbf{Y}$ has a shape of $(N, \text{out\_features}) = (64, 512)$, 
and the input $\mathbf{X}$ has a shape of $(N, \text{in\_features}) = (64, 1024)$. 
For the Jacobian tensor $\mathbf{\frac{\partial \mathbf{Y}}{\partial \mathbf{X}}}$, we need to consider the partial derivatives 
of each element in $\mathbf{Y}$ with respect to each element in $\mathbf{X}$. so the shape is:

$$(N, \text{out\_features}, N, \text{in\_features}) = (64, 512, 64, 1024)$$

**1.B:**

Yes, the Jacobian $\mathbf{\frac{\partial \mathbf{Y}}{\partial \mathbf{X}}}$ is sparse. 
Each output vector $\mathbf{Y}[n]$ depends only on the corresponding input vector $\mathbf{X}[n]$. 
Thus, for $n \neq m$, the partial derivatives $\frac{\partial \mathbf{Y}[n]}{\partial \mathbf{X}[m]} = 0$. 
Only the diagonal terms $\frac{\partial \mathbf{Y}[n]}{\partial \mathbf{X}[n]}$ are non-zero.


**1.C:**

No, we do not need to materialize the Jacobian. Instead, we can compute $\delta \mathbf{X}$ 
(the gradient of the loss with respect to $\mathbf{X}$) using matrix multiplication. 
Given $\delta \mathbf{Y} = \frac{\partial L}{\partial \mathbf{Y}}$, the gradient is:

$$\delta \mathbf{X} = \frac{\partial L}{\partial \mathbf{X}} = \frac{\partial L}{\partial \mathbf{Y}} \cdot \mathbf{W}^\top = \delta \mathbf{Y} \cdot \mathbf{W}^\top$$

This avoids explicitly constructing the Jacobian.


**2.A:**

The output $\mathbf{Y}$ has a shape of $(N, \text{out\_features}) = (64, 512)$, 
and the weight $\mathbf{W}$ has a shape of $(\text{out\_features}, \text{in\_features}) = (512, 1024)$. 
For The Jacobian tensor $\frac{\partial \mathbf{Y}}{\partial \mathbf{W}}$ we need to consider the partial derivatives 
of each element in $\mathbf{Y}$ with respect to each element in $\mathbf{W}$ . so the shape is:


$$(N, \text{out\_features}, \text{out\_features}, \text{in\_features}) = (64, 512, 512, 1024)$$


**2.B:**

No, the Jacobian $\frac{\partial \mathbf{Y}}{\partial \mathbf{W}}$ is not sparse. 
Each element of $\mathbf{Y}$ depends on multiple elements of $\mathbf{W}$, 
as the output of a fully connected layer is computed as $\mathbf{Y} = \mathbf{X} \cdot \mathbf{W}^\top$.


**2.C:**

No, we do not need to materialize the Jacobian. Instead, we can compute $\delta \mathbf{W}$ 
(the gradient of the loss with respect to $\mathbf{W}$) using matrix multiplication. 
Given $\delta \mathbf{Y} = \frac{\partial L}{\partial \mathbf{Y}}$, the gradient is:

$$\delta \mathbf{W}= \frac{\partial L}{\partial \mathbf{W}} =
\frac{\partial L}{\partial \mathbf{Y}} \cdot \frac{\partial \mathbf{Y}}{\partial \mathbf{W}} = 
\frac{\partial L}{\partial \mathbf{Y}} \cdot \mathbf{X}^T =
\delta \mathbf{Y} \cdot \mathbf{X}^\top$$



This avoids explicitly constructing the Jacobian.



"""

part1_q2 = r"""
**Your answer:**

Back-propagation is not required for training neural networks using gradient-based optimization,
as alternatives like derivative-free optimization (e.g., Nelder-Mead) or finite difference methods can be used. However, 
these approaches are typically inefficient and impractical for large-scale networks. Back-propagation leverages the chain rule,
computational graphs, and automatic differentiation to compute gradients efficiently, making it crucial for modern deep learning. 
While alternatives exist, back-propagation remains the most practical and widely used method due to its scalability, precision, 
and computational effectiveness.

"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0.1, 0.05, 0

    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0.051,
        0.019,
        0.0041,
        0.00019,
        0.00151
    )
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0.15,
        0.001,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Answer:**
1.  Yes, the graph results we got match what we expected.
    The point of dropout is to prevent overfitting to the training data.
    We can see that when the model has no dropout layers, it has the best accuracy out of all the other models.
    This is because the model is overfitting to the training data.
    When we add dropout layers, the model's accuracy decreases. which is what we expected aswell. 
    Because the model is not overfitting to the training data anymore.
    But at the same time, the model with no dropout layers has a worse accuracy on the validation set.
    This is because when the model is overfitting to the training data, it is not generalizing well to the validation set,
    and it's learming the noise in the training data.
    When we apply too much dropout, the model's accuracy decreases even more, because the model is not learning enough from the training data,
    and it's underfitting to the training data.
    
2.  So the best model in our results is when we apply dropout of 0.4, because it has the best accuracy on the validation set,
    but if we look at the the one with dropout of 0.8, it's underfitting to the training data, and it's not learning enough from it,
    and then it has a bad accuracy on the validation set.
"""

part2_q2 = r"""
**Answer:**
    Yes, this is possible. Because the accuracy measures the number of correct predictions made by the model,
    but the cross-entropy loss measures the difference between the predicted probability distribution and the actual probability distribution,
    which reflects the model's confidence in its predictions.
    So there might be a situation where the model is making more correct predictions, but it's not confident in its predictions, 
    in other words, the difference between the predicted probability distribution and the actual probability distribution is increasing, 
    which means the cross-entropy loss is increasing.
    For example, let's say we a binary classification problem with probabilities of \[0.999, 0.001\] at the start, 
    and after a few epochs, the model's predictions are \[0.7, 0.3\], the model is making a correct prediction in both epochs,
    and thus the accuracy is increasing.
    But the difference between the predicted probability distribution and the actual probability distribution is increasing,
    and thus the cross-entropy loss is increasing.
"""

part2_q3 = r"""
**Answer:**
1.  
    - Gradient descent is the optimization algorithm that uses the gradients to update the hyper parameters and biases of the model, reducing the loss over time.
    - Backpropagation is the algorithm used to calculate the gradient of the loss function w.r.t. hyper parameters.


2.  
    - They are both optimization algorithms we use while traning models.
    - While copmuting the gradients, GD considers the entire dataset, while SGD samples a subset of the dataset 
      and considers only this subset in comuting the gradiants.
    - GD is more stable and has less noisy updates while SGD has faster updates and requires less memory.
    
    
3.
    - GD is slower, memory-intensive, and prone to numerical errors when working with large datasets.
    - SGD solves these issues by using mini-batches for faster training, reduced memory usage, and fewer numerical errors.
    
    And because in deep learning we have huge datasets, thus SGD is preferred over GD.
    
4. 
A. The total loss is:

$$
L = \frac{1}{N} \sum_{i=1}^N \ell(f(x_i, \theta), y_i),
$$

where $\ell$ is the loss, $f(x_i, \theta)$ is the model's output, and $\theta$ represents parameters. The gradient of $ L $ is:

$$
\nabla_\theta L = \frac{1}{N} \sum_{i=1}^N \nabla_\theta \ell(f(x_i, \theta), y_i).
$$

If the dataset is split into $ M $ batches, with each batch loss $ L_j $, and the total loss is:

$$
L = \frac{1}{M} \sum_{j=1}^M L_j,
$$

Then the gradient $ \nabla_\theta L $ from summing batch losses is mathematically identical to GD. 


B. Accumulating losses over all batches for a single backward pass requires storing computational graphs for all batches.
Memory usage grows linearly with the number of batches, leading to an out-of-memory error.

"""

part2_q4 = r"""
**Answer:**
1. 

- We can use a technique called checkpointing, the algorithm initializes two variables currentGradient and currentResult and proceeds with a forward pass through the computational graph. At each step, it computes the derivative of the current function, updating both the gradient and the result accordingly. By only keeping track of the gradient and the result at each point, the memory complexity is reduced to $O(1)$, effectively minimizing the memory footprint. However, in cases where the intermediate results are not pre-stored or readily available, the memory complexity increases to $O(n)$, as it becomes necessary to store all intermediate values to ensure accurate gradient computation.
- We can also use checkpointing here, the algorithm initializestwo variables backwardGradient and backwardResult are initialized, and a forward pass through the computational graph is performed, during which the intermediate results of the functions are saved. Then, in the backward pass, the algorithm iterates from the output to the input, calculating the gradients based on the previously saved function results. By only retaining the essential function results required for gradient computation, the memory complexity is reduced to $O(1)$, minimizing memory usage while still maintaining the accuracy of the gradient calculation.

2. Initially, for a computational graph with multiple input nodes, we would need two variables per input node, leading to $O(n)$ memory complexity. However, by traversing the graph from input to output for each gradient element, we can reduce memory complexity to $O(1)$. The drawback is that this approach isn't suitable for parallel computation, as it requires storing multiple sets of intermediate results, returning the memory complexity to $O(n)$.

3. In deep architectures like VGGs and ResNets, which have numerous parameters and layers, memory optimization techniques provide significant benefits. These architectures typically demand high memory, but by reducing the memory complexity of gradient computation to $O(1)$, we can ease the memory load during training. This is particularly advantageous for training on hardware with limited memory. Moreover, the reduced memory complexity enhances training speed and scalability, leading to faster overall training for deep architectures.

"""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**

**1:**

High Optimization Error occurs when The model isn't learning the training data well enough ,
i.e not minimizing the training loss effectively.
This could be due to factors such as suboptimal optimization algorithms,
low number of training epochs,or inadequate model complexity.
To improve optimization,we should consider using advanced optimization algorithms,
adjusting learning rates, increasing the number of training epochs, 
or Enhance the network's capacity by increasing layers or the number of units per layer, 
ensuring the model has a larger receptive field.


**2:**

This happens when the model performs well on the training data but fails to generalize effectively to unseen test data,
which is a clear sign of overfitting. To improve this issue, 
we can use regularization methods like L1/L2 regularization or dropout can be utilized to minimize overfitting. 
Additionally, the model's capacity can be adjusted by decreasing the number of layers or neurons, 
especially if the receptive field is excessively large for the problem at hand. 
Increasing the variety within the training dataset is another effective way to enhance the model's ability to generalize.


**3:**

This occurs when the model is too simple to capture the data's complexity, leading to underfitting. 
To improve this,we can use a more powerful hypothesis class like deep neural networks (DNNs) with additional parameters to model intricate patterns. 
Introducing inductive bias, such as using convolutional neural networks (CNNs) for image tasks, tailors the model to the domain,
improving its ability to learn relevant features and reducing approximation error.



"""

part3_q2 = r"""
**Your answer:**

case expecting false positive rate (FPR) to be higher:

**Tumor classifier**: This might occur in a scenario where the classifier is designed to err on the side of caution,
labeling benign tumors as malignant to ensure no possible cancer case is missed. 
This would lead to more unnecessary follow-ups or biopsies but minimizes the risk of missing a potentially life-threatening diagnosis.


case expecting false negative rate (FNR) to be higher:

**Spam email classifier:**
here false negative means A spam email is incorrectly identified as not spam and delivered to the inbox,
and false positive means A legitimate email is incorrectly identified as spam and sent to the spam folder.
Users are generally more tolerant of receiving a few extra spam emails in their inbox than missing important messages, 
that means they expect that the false negative rate (FNR) to be higher and not the false positive rate.


"""

part3_q3 = r"""
**Your answer:**

**1:** 

In this scenario, we should prioritize minimizing false positives. Since a false negative only means a delay in treatment (not a missed opportunity),
we can tolerate a higher false negative rate. We would choose a point on the ROC curve that has a low FPR, even if it means a lower TPR.
This means setting a higher threshold for our initial test, making it more stringent. This reduces the number of people sent for expensive testing.

**2:**

In this scenario, we must prioritize minimizing false negatives. Missing a case could be fatal. 
We are willing to accept a higher false positive rate to ensure we catch as many true positives as possible. 
We would choose a point on the ROC curve with a high TPR, even if it means a higher FPR.
This means setting a lower threshold for our initial test, making it more sensitive.
This will send more people for confirmatory testing, but it's a trade-off we must make to save lives.


"""


part3_q4 = r"""
**Your answer:**

MLPs are bad at understanding sequntial data (like words in a sentence or events in time) because they treat each item separately.
They don't remember what came before, so they miss the important connections between things in a sequence. 
For example, an MLP wouldn't understand that "cat chases mouse" is different from "mouse chases cat" because it sees
"cat," "chases," and "mouse" as separate, unrelated things. This makes them unsuitable for tasks where order matters, 
like understanding text, translating languages, or recognizing speech.
They're designed for single, independent inputs, not for analyzing patterns in ordered data.

"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = torch.nn.CrossEntropyLoss()
    lr, weight_decay, momentum = 0.05, 0.005, 0.5
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Answer:**
1.  Number of Parameters:
    - *Regular Block:* Each $3x3$ convolution with 256 input and output channels has
    $(3 \cdot 3 \cdot 256) \cdot 256 + 256 (bias) = 589,824 + 256 = 590,080$ parameters.
    Since there are two such layers, the total number of parameters is $2 \cdot 590,080 = 1,180,160$.
    - *Bottleneck Block:*
        - $1x1$: $( 1 \cdot 1 \cdot 256) \cdot 64 + 64 (bias) = 16,384 + 64 = 16,448$
        - $3x3$: $(3 \cdot 3 \cdot 64) \cdot 64 + 64 (bias) = 36,864 + 64 = 36,928$
        - $1x1$: $(1 \cdot 1 \cdot 64) \cdot 256 + 256 (bias) = 16,384 + 256 = 16,640$
        - Total: $16,448 + 36,928 + 16,640 = 70,016$

    As we can see, the bottleneck block dramatically reduces the number of parameters.

2.  Number of floating point operations:
    to calculate the number of flops we use the following formula:
    $$FLOPs = 2 \cdot C_{in} \cdot K^2 \cdot C_{out} \cdot H_{out} \cdot W_{out}$$
    Due to the fact that in the bottleneck block we are using smaller kernel sizes than the regular block,
    then the number of flobs is the regular block is higher.
    
3.  Spatial and Cross-Channel Combination:
    - Both blocks effectively combine spatial information through the 3x3 convolutions.
    - The bottleneck block enhances cross-channel mixing through the 1x1 convolutions.
      The first 1x1 convolution projects the input to a lower-dimensional space, facilitating more efficient cross-channel interactions.
      The second 1x1 convolution then projects this combined representation back to the original dimensionality.
"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**

**1:**


Analyzing the graphs, it is evident that increasing the network depth led to a decrease in accuracy,
with the shallowest depth of 2 yielding the best results for both k=32 and k=64.
This could be attributed to overfitting: while deeper networks are capable of learning more complex patterns, 
they are also more susceptible to overfitting, especially on limited or noisy datasets. 
A depth of 2 may have been sufficient to capture the relevant features of the data without introducing excessive complexity. 
By avoiding deeper architectures, the network likely reduced the risk of overfitting, thereby achieving better generalization performance.


**2:**

In our experiment, we encountered instances where the network became non-trainable for values of 
$\mathbf{L}=8$ and $\mathbf{L}=16$. This issue is likely caused by the vanishing gradient problem or unstable gradients during training. 
As the depth of the network increases, the gradients used to update weights during backpropagation may diminish or explode,
making it difficult for the network to learn effectively.
to resolve this partially we can add Batch Normalization layers that helps normalize the input to each layer, 
stabilizing the training process and preventing gradients from vanishing or exploding.
other solution may be to incorporate residual connections (as used in ResNet architectures), 
which allow gradients to bypass certain layers and flow more directly through the network, mitigating the vanishing gradient problem.
"""

part5_q2 = r"""
**Your answer:**

The analysis of the graphs shows that for fixed values of L (specifically L=2 and L=4),
increasing K leads to higher training and test accuracies, a trend more pronounced than in experiment 1_1. 
However, L=8 results very low accuracies across all K values, indicating non-trainability when L exceeds 4.
The configuration that achieved the best results is L=2, K=64, same as experiment 1_1.

Overall, while increasing K improves accuracy for L=2 and L=4, L=8 remains consistently poor.

"""

part5_q3 = r"""
**Your answer:**

Upon examining the graphs, it becomes clear that the model's performance changes with varying depths ($\mathbf{L}$). 
Notably, in this experiment, $\mathbf{L} = 3$, representing the second shallowest depth, achieves the highest test accuracy. 
This indicates that a moderate network depth is most effective for achieving optimal accuracy.
we can see that unlike previouus expirments where $\mathbf{L}$ > 4 made the network non-trainable , 
here $\mathbf{L}$ = 4 makes the network non-trainable, The failure to train effectively at $\mathbf{L}$ = 4 
implies that gradients are vanishing as they propagate back through the deeper layers of the network.


"""

part5_q4 = r"""
**Your answer:**

with **K = 32**:

The provided graph demonstrates that ResNet effectively addresses the underfitting challenges observed in previous CNN results.
In the training accuracy plot, all ResNet configurations (L8, L16, and L32) exhibit smooth convergence, 
with L8 achieving the fastest and highest training accuracy. This contrasts sharply with CNN,
where deeper architectures were non-trainable due to vanishing or exploding gradients. 
The test accuracy plot highlights ResNet's ability to generalize,
as the L16 configuration achieves the best balance between training and test performance, 
outperforming L8 (mild underfitting) and L32 (optimization challenges).
Unlike the CNN, which struggled to learn meaningful representations,
ResNet's residual connections stabilize gradient flow, enabling successful training even in deeper networks.


with **K = 64-128-256**:


The graph shows that the deeper ResNet model (L8) achieves the best training and test accuracy, 
while shallower models (L2, L4) converge faster but underperform in both metrics.
The L8 model's stable test accuracy above 60% demonstrates strong generalization, unlike L2, 
which diverges and underfits. Compared to previous CNN results, which struggled with vanishing gradients and severe underfitting,
ResNet effectively overcomes these issues using residual connections,
enabling deeper architectures like L8 to train successfully and generalize well.


"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Answer:**
1.  In the first pic the model didn't get anything right, it recognized two dolphins as persons and the tail of another one as a surfboard.
    For one 'person' it was 90% confident, but for the others it was less than 50% confident.
    For the second image it also didn't do well, although it recognized that there are cats and dogs, 
    it classified two dogs as cats and one dog as a dog, but the bounnding box included another dog.  
    For all of them it was less than 70% confident.
    
2.  The model could have limited appility to capture are recognize complex patterns in pictures.
    Also, it could be that the model didn't have a good training, for example, getting trained on a training set that isn't diverse,
    more specifically, a training set that dosen't have many types pf dogs or dolphins.
    
    As for a solution, we can train the model on more data and much more diverse pictures, 
    and maybe switch to a better model that dosen't have limitations like YOLOv5 might have.
    
3.  If for example we want to make YOLO misclassify, we can iteratively modify the input image by calculating gradients of the loss with respect to the image and applying small updates in the direction that increases misclassification.
    These updates are constrained within a set bound to ensure the perturbation remains imperceptible to humans. 

"""


part6_q2 = r"""
**Your aAnswer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Answer:**
The model didn't do well at all, as it couldn't detect anything.
- For the first pic, the problem probably is that the leaves are apstracting the face of the monkey, making it hard to understand what is there in the pic.
- For the second one, the problem is for sure the blurry face and background.
- And for the last pic, the problem is the lighting for sure, as the model most definitely didn't understand that these is a person standing there.
"""

part6_bonus = r"""
**Answer:**

- For the monkey image we used AI to remove leaves abstracting the monkey's face, and then it detected a bird, it didn't classify a monkey
but it's still better than before.
- For the dog image, we filtered the image using the filter in the code above, and again, it misclassified the dog to a cat, 
but still better than before.
- And for the man image, we brightened up the image using the filter in the code above, and now YOLO got it right!! it said that this is a person.


"""