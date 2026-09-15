def predict(x1, x2, weight_1, weight_2, bias):
    # Each input has its own weight. Both share one bias.
    return weight_1 * x1 + weight_2 * x2 + bias


def main():
    # Start without knowing the correct parameter values.
    weight_1 = 0.0
    weight_2 = 0.0
    bias = 0.0

    # Controls how large each adjustment is.
    learning_rate = 0.05

    # Each example contains (input_1, input_2, correct_answer).
    training_examples = [
        (0, 0, 1),
        (1, 0, 3),
        (0, 1, 4),
        (1, 1, 6),
        (2, 1, 8),
        (1, 2, 9),
    ]

    # Each epoch is one pass through all training examples.
    for epoch in range(1000):
        # Reset these totals each epoch.
        # Keep the weights and bias learned in previous epochs.
        total_squared_error = 0.0
        weight_1_gradient_sum = 0.0
        weight_2_gradient_sum = 0.0
        bias_gradient_sum = 0.0

        for x1, x2, target in training_examples:
            prediction = predict(x1, x2, weight_1, weight_2, bias)
            error = prediction - target

            total_squared_error += error ** 2

            # Each weight's gradient uses its corresponding input.
            weight_1_gradient_sum += 2 * error * x1
            weight_2_gradient_sum += 2 * error * x2

            # Bias affects every prediction directly.
            bias_gradient_sum += 2 * error

        # Average the totals across all examples.
        example_count = len(training_examples)

        loss = total_squared_error / example_count
        weight_1_gradient = weight_1_gradient_sum / example_count
        weight_2_gradient = weight_2_gradient_sum / example_count
        bias_gradient = bias_gradient_sum / example_count

        # Report loss before this epoch's parameter updates.
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss={loss:.6f}")

        # Move opposite each gradient to reduce loss.
        # All gradients came from the same parameter settings.
        weight_1 -= learning_rate * weight_1_gradient
        weight_2 -= learning_rate * weight_2_gradient
        bias -= learning_rate * bias_gradient

    # Training is finished.
    print(f"\nFinal weight_1: {weight_1:.4f}")
    print(f"Final weight_2: {weight_2:.4f}")
    print(f"Final bias: {bias:.4f}")

    # Use the learned parameters on a new input pair.
    # No parameter updates happen here.
    prediction = predict(2, 2, weight_1, weight_2, bias)
    print(f"Prediction for x1=2, x2=2: {prediction:.4f}")


if __name__ == "__main__":
    main()