# DeepSeek-R1-Paper-Implementations
I am going to learn and code various components of the popular DeepSeek-R1 paper and share my findings using the NumPy library.

## Knowledge Distillation

### What is Knowledge Distillation?
A technique used in machine learning to transfer knowledge from a large, complex model called the 'teacher' to a smaller, more efficient model called the 'student'. By doing this, the student is able to learn from the teacher's expertise without needing to replicate the full training process. This allows individuals to deploy lightweight models that maintain most of the model's accuracy, but are also faster and require less resources.

### Soft Labels and Temperature Scaling
Traditional hard labels tend to use one-hot encoding (where a correct class is marked with 1 and incorrect class is marked with 0), however, Knowledge Distillation changes it up and uses soft labels. Soft labels are probability distributions over all classes, reflecting the teacher's confidence in each class. For example, instead of saying 'dog' is the correct class, the teacher might indicate 90% dog, 5% cat, and 5% bird, which reveals relationships between classes.

To create these soft labels, the teacher's output logits are passed through a temperature-scaled softmax function. The temperature parameter T controls how softened or peaked the output probabilities are. A higher temperature means a softer, more uniform distribution that reveals more detailed information for the student to learn from.
