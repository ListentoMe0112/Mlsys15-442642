import functools
from typing import Callable, Tuple

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import auto_diff as ad


def logistic_regression(X: ad.Node, W: ad.Node, b: ad.Node) -> ad.Node:
    """Construct the computational graph of a logistic regression model.

    Parameters
    ----------
    X: ad.Node
        A node in shape (batch_size, in_features), denoting the input data.
    W: ad.Node
        A node in shape (in_features, num_classes), denoting the the weight
        in logistic regression.
    b: ad.Node
        A node in shape (num_classes,), denoting the bias term in
        logistic regression.

    Returns
    -------
    logits: ad.Node
        The logits predicted for the batch of input.
        When evaluating, it should have shape (batch_size, num_classes).
    """
    """TODO: Your code here"""
    z = ad.matmul(X, W)
    b_broad = ad.broadcast_to(b, z, axis=0)
    z = z + b_broad
    return z 


def softmax_loss(Z: ad.Node, y_one_hot: ad.Node, batch_size: int) -> ad.Node:
    """Construct the computational graph of average softmax loss over
    a batch of logits.

    Parameters
    ----------
    Z: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        logits for the batch of instances.

    y_one_hot: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        one-hot encoding of the ground truth label for the batch of instances.

    batch_size: int
        The size of the mini-batch.

    Returns
    -------
    loss: ad.Node
        Average softmax loss over the batch.
        When evaluating, it should be a zero-rank array (i.e., shape is `()`).

    Note
    ----
    1. In this assignment, you do not have to implement a numerically
    stable version of softmax loss.
    2. You may find that in other machine learning frameworks, the
    softmax loss function usually does not take the batch size as input.
    Try to think about why our softmax loss may need the batch size.
    """
    # 计算指数项 exp(z_i)
    exp_z = ad.exponentiation(Z)
    
    # 对每个样本的类别维度求和 (保持维度用于广播)
    sum_exp_z = ad.sum_along_axes(exp_z, axis=(1,))
    
    # 计算 log(sum(exp(z_i)))
    log_sum_exp = ad.logarithm(sum_exp_z)
    
    # 计算 z_y = sum(z * y_one_hot) 
    z_y = ad.sum_along_axes(ad.mul(Z, y_one_hot), axis=(1,))
    
    # 计算单个样本的损失: log_sum_exp - z_y
    per_sample_loss = log_sum_exp - z_y
    
    # 对所有样本的损失求和
    total_loss = ad.sum_along_axes(per_sample_loss, axis=0)
    
    # 计算平均损失: total_loss / batch_size
    avg_loss = ad.mul_by_const(total_loss, 1.0 / batch_size)
    
    return avg_loss


def sgd_epoch(
    f_run_model: Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ],
    X: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    batch_size: int,
    lr: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run an epoch of SGD for the logistic regression model
    on training data with regard to the given mini-batch size
    and learning rate.

    Parameters
    ----------
    f_run_model: Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]
        The function to run the forward and backward computation
        at the same time for logistic regression model.
        It takes the training data, training label, model weight
        and bias as inputs, and returns the logits, loss value,
        weight gradient and bias gradient in order.
        Please check `f_run_model` in the `train_model` function below.

    X: np.ndarray
        The training data in shape (num_examples, in_features).

    y: np.ndarray
        The training labels in shape (num_examples,).

    W: np.ndarray
        The weight of the logistic regression model.

    b: np.ndarray
        The bias of the logistic regression model.

    batch_size: int
        The mini-batch size.

    lr: float
        The learning rate.

    Returns
    -------
    W_updated: np.ndarray
        The model weight after update in this epoch.

    b_updated: np.ndarray
        The model weight after update in this epoch.

    loss: np.ndarray
        The average training loss of this epoch.
    """
    num_samples = X.shape[0]
    total_loss = 0.0
    num_batches = 0
    
    # 遍历所有mini-batch
    for start_idx in range(0, num_samples, batch_size):
        end_idx = start_idx + batch_size
        X_batch = X[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]
        
        # 将标签转换为one-hot编码
        y_one_hot = np.eye(10)[y_batch]  # 假设有10个类别
        
        # 运行模型获取梯度
        _, loss_val, grad_W, grad_b = f_run_model(X_batch, y_one_hot, W, b)
        
        # 更新参数
        W -= lr * grad_W
        b -= lr * grad_b.squeeze()
        
        # 累计损失
        total_loss += loss_val
        num_batches += 1
    
    # 计算平均损失
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return W, b, avg_loss


def train_model():
    """Train a logistic regression model with handwritten digit dataset.

    Note
    ----
    Your implementation should NOT make changes to this function.
    """
    # - Set up the training settings.
    num_epochs = 100
    batch_size = 50
    lr = 0.05

    # - Define the forward graph.
    x = ad.Variable(name="x")
    W = ad.Variable(name="W")
    b = ad.Variable(name="b")
    y_predict = logistic_regression(x, W, b)
    # - Construct the backward graph.
    y_groundtruth = ad.Variable(name="y")
    loss = softmax_loss(y_predict, y_groundtruth, batch_size)
    grad_W, grad_b = ad.gradients(loss, nodes=[W, b])
    # - Create the evaluator.
    evaluator = ad.Evaluator([y_predict, loss, grad_W, grad_b])
    test_evaluator = ad.Evaluator([y_predict])

    # - Load the dataset.
    #   Take 80% of data for training, and 20% for testing.
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=0
    )
    num_classes = 10
    in_features = functools.reduce(lambda x1, x2: x1 * x2, digits.images[0].shape, 1)

    # - Initialize model weights.
    np.random.seed(0)
    stdv = 1.0 / np.sqrt(num_classes)
    W_val = np.random.uniform(-stdv, stdv, (in_features, num_classes))
    b_val = np.random.uniform(-stdv, stdv, (num_classes,))

    def f_run_model(X_val, y_val, W_val, b_val):
        """The function to compute the forward and backward graph.
        It returns the logits, loss, and gradients for model weights.
        """
        z_val, loss_val, grad_W_val, grad_b_val = evaluator.run(
            input_values={x: X_val, y_groundtruth: y_val, W: W_val, b: b_val}
        )
        return z_val, loss_val, grad_W_val, grad_b_val

    def f_eval_model(X_val, W_val, b_val):
        """The function to compute the forward graph only and returns the prediction."""
        logits = test_evaluator.run({x: X_val, W: W_val, b: b_val})
        return np.argmax(logits[0], axis=1)

    # - Train the model.
    for epoch in range(num_epochs):
        X_train, y_train = shuffle(X_train, y_train)
        W_val, b_val, loss_val = sgd_epoch(
            f_run_model, X_train, y_train, W_val, b_val, batch_size, lr
        )

        # - Evaluate the model on the test data.
        predict_label = f_eval_model(X_test, W_val, b_val)
        print(
            f"Epoch {epoch}: test accuracy = {np.mean(predict_label == y_test)}, "
            f"loss = {loss_val}"
        )

    # Return the final test accuracy.
    predict_label = f_eval_model(X_test, W_val, b_val)
    return np.mean(predict_label == y_test)


if __name__ == "__main__":
    print(f"Final test accuracy: {train_model()}")
