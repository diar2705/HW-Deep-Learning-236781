r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
1.  False:
    The in-sample error measures the error of the model we trained on the same training data,
    in other words, it measures how well the model fits the training data.
    But the test set allows us to estimate the out-sample error, or in other words, the generalization error,
    which measures how well the model fits unseen data.
    
2.  False:
    For example, if our task is to identify horses, and we only train the model on white horses 
    (the training set only has white horses), but we tested it on pictures of brown and black horses 
    (the test set contains black and brown horses).
    Here the model wouldn't classify them (black and brown horses) as horses, as opposed to a model that 
    was trained with a training set that contains all types and colors of horses.
    So not all splits constitute an equally useful train-test split.
    
3.  True:
    The cross-validation is used to tune the hyperparameters when training the model.
    So using the test set during cross-validation will lead to an unrealistic loss during the test of our model
    because the model saw this data before (A leak happened) which is against the rule testing the model on
    unseen data.

4.  True: 
    The cross-validation is used to tune the hyperparameters when training the model, and it is used on an unseen data
    during the training stage.
    Cross-validation provides a more realistic estimate of the model's generalization performance,
    Therefore each validation set is used as an approximation method for the model's generalization error.
"""

part1_q2 = r"""
**Answer:**
    No, his approach isn't justified.
    The tunning of the hyperparameters is done only during the training stage, using only the training set, 
    and the test set shouldn't be incorporated into the tuning of the hyperparameters. Using the test set would result
    an unrealistic loss and would overfit to the test data, something that we don't want to happen, as we want to generalize.
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Answer:**
If the margin threshold $\Delta$ is negative, the margin for correct classification would be set in
such a way that the classifier penalizes points that are correctly classified by a large margin.
Else, if it's positive, 
"""

part2_q2 = r"""
**Answer:**
The linear model learns by assigning likelihoods to each pixel based on how well it matches each class’s features. For example, it gives high scores for a straight line to classify the digit "1." However, it struggles with similar-looking digits like 5 and 6. It works by drawing hyperplanes to separate classes in space, but errors happen when a sample is closer to the wrong class boundary.



"""

part2_q3 = r"""
**Answer:**
1. Based on the training loss graph, we can conclude that the chosen learning rate is good. Initially, the loss decreases significantly, showing rapid progress, and then slows down as it approaches a low value, if our answer was too low then the graph should be close to a linear graph and if our answer was too high then the the loss graph would drop quickly at first but fluctuate around the minimum without settling smoothly.

2. Our model shows slightly overfitting to the training data, as the training accuracy is higher than the validation accuracy, However, the difference between them is small.
"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Answer:**

"""

part3_q2 = r"""
**Answer:**


"""

part3_q3 = r"""
**Answer:**


"""

# ==============

# ==============
