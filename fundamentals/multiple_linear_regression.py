"""Lesson starter: two inputs, two weights, one bias.

No extra packages needed. We will implement prediction together first,
then add loss, gradients, and training one step at a time.
"""


def main():
    # Each example is (input_1, input_2, target).
    # These are invented numbers for learning, not market measurements.
    training_examples = [
        (0, 0, 1),
        (1, 0, 3),
        (0, 1, 4),
        (1, 1, 6),
        (2, 1, 8),
        (1, 2, 9),
    ]

    print("Next lesson: prediction = weight_1 * x1 + weight_2 * x2 + bias")
    print("Each row contains two inputs and one correct answer:")

    for x1, x2, target in training_examples:
        print(f"x1={x1}, x2={x2}, target={target}")

    # First exercise for our next session:
    # 1. Compare rows where only x1 changes. What changes in the target?
    # 2. Compare rows where only x2 changes. What changes in the target?
    # 3. Write a predict() function with two weights and one bias.
    # This starter only displays data; it does not train a model yet.


if __name__ == "__main__":
    main()
