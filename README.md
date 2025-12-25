# DeepSeek-R1-Paper-Implementations
I am going to learn and code various components of the popular DeepSeek-R1 paper and share my findings using the NumPy library.

## Knowledge Distillation

### What is Knowledge Distillation?
A technique used in machine learning to transfer knowledge from a large, complex model called the 'teacher' to a smaller, more efficient model called the 'student'. By doing this, the student is able to learn from the teacher's expertise without needing to replicate the full training process. This allows individuals to deploy lightweight models that maintain most of the model's accuracy, but are also faster and require less resources.

### Soft Labels and Temperature Scaling
Traditional hard labels tend to use one-hot encoding (where a correct class is marked with 1 and incorrect class is marked with 0), however, Knowledge Distillation changes it up and uses soft labels. Soft labels are probability distributions over all classes, reflecting the teacher's confidence in each class. For example, instead of saying 'dog' is the correct class, the teacher might indicate 90% dog, 5% cat, and 5% bird, which reveals relationships between classes.

To create these soft labels, the teacher's output logits are passed through a temperature-scaled softmax function. The temperature parameter T controls how softened or peaked the output probabilities are. A higher temperature means a softer, more uniform distribution that reveals more detailed information for the student to learn from.

### Distillation Loss Function
The student model is trained to mimic the teacher's softened output distribution using a loss function based on the Kullback-Leibler (KL) divergence. KL divergence measures how different two probability distributions are. In this case, it calculates the difference between the teacher's soft labels and the student's prediction.

To ensure stability and effective learning, you scale the KL divergence by the square of the  temperature \( T^2 \). By doing this, it compensates for the effect of the temperature on the gradients during back propagation, keeping them consistent regardless of the temperature chosen.

Mathematically, the distillation loss is given by the formula: \[
L_{\text{distill}} = T^2 \times D_{KL}\left(\sigma\left(\frac{z_t}{T}\right) \parallel \sigma\left(\frac{z_s}{T}\right)\right)
\]
where \( z_t \) and \( z_s \) are the logits from the teacher and student models respectively, and \( \sigma \) represents the softmax function.

## Challenges in Evaluating Reasoning Models
Reasoning models tend to output multiple possible solutions through different reasoning paths, not just a single answer. This results in variability between responses. This variability makes it difficult to assess model performance accurately using traditional metrics. Single-sample evaluation tends to fail to reflect the model's true abilities. Thus, robust evaluation metrics are critical to account for this variation and provide a more reliable measure of reasoning quality.

## Evaluation Metrics Overview

### Pass@1 -> Baseline Accuracy
Pass@1 measures the probability that a single response the model generates is correct. It is calculated by averaging the correctness over all responses. This simple metric is efficient and serves as a baseline for comparing different models or configurations. Pass@1 reflects the expected accuracy when only one response per problem is considered.

### Majority Voting (Consensus) -> Improving Accuracy
Majority voting shows how correct answers tend to be consistent across multiple generated responses, whereas incorrect answers are more variable. By choosing the most frequently occurring answer from multiple samples, majority voting will filter out inconsistent errors and will significantly improve accuracy for problems with a single unambiguous solution. This useful approach shines in domains like math problem-solving.

### Pass@k -> Probability of Success in k Attempts
Pass@k measures the probability that at least one out of k generated responses is correct. This metric becomes useful in scenarioes like code generation, where any correct output among multiple attempts counts as success. Pass@k is computed using an unbiased estimator involving combinations. Efficient computational methods avoid dealing with large factorials, ensuring numerical stability.

Pass@k formula: \[\text{pass@k} = 1 - \frac{\binom{n-c}{k{\binom{n}{k}}\] where n is the total number of samples, c is the number of correct samples, and k is the number of attempts. This formula calculates the complement of the probability that all k samples are incorrect.

## KL Divergence in RLHF
KL Divergence measures the difference between two probability distributions. In this case, it is the current policy and the original pre-trained policy. It acts as a penalty to prevent the model from drifting too far. However, the standard KL divergence formula requires summing over all possible outputs,  which is infeasible for language models with vast outputs.

### GRPO KL Divergence Estimator
The Group Relative Policy Optimization algorithm introduces an unbiased, per-sample estimator for KL divergence. Instead of summing over all outputs, it calculates divergence based on the ratio of probabilities for a single sampled output. The formula is \[\text{KL estimate} = r-\ln(r)-1\] where \[r=\frac{\pi {ref}(o|q)}{\pi_{\theta}(o|q)}\]. \(\pi_{\theta}\) is the current policy and \(pi_{\ref}\) is the reference policy, \(o\) is the output and \(q\) is the input.