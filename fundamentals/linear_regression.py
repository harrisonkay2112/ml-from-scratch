def predict(x, weight, bias):
    return weight * x + bias


def main():
    weight = 0.0
    bias = 0.0
    learning_rate = 0.1

    training_examples = [(1, 6), (2, 11), (3, 13)]
    test_examples = [(4, 18), (5, 22)]

    # TRAINING: use errors to adjust weight and bias.
    for epoch in range(1000):
        total_squared_error = 0.0
        weight_gradient_sum = 0.0
        bias_gradient_sum = 0.0

        for x, target in training_examples:
            prediction = predict(x, weight, bias)
            error = prediction - target

            total_squared_error += error ** 2
            weight_gradient_sum += 2 * error * x
            bias_gradient_sum += 2 * error

        loss = total_squared_error / len(training_examples)
        weight_gradient = weight_gradient_sum / len(training_examples)
        bias_gradient = bias_gradient_sum / len(training_examples)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: training loss={loss:.6f}")

        weight -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient

    # Training is finished. The learned values remain in memory.
    print(f"\nFinal weight: {weight:.4f}")
    print(f"Final bias: {bias:.4f}")

    # TESTING: evaluate the learned values without changing them.
    test_squared_error = 0.0

    for x, target in test_examples:
        prediction = predict(x, weight, bias)
        error = prediction - target
        test_squared_error += error ** 2

        print(
            f"Test: x={x}, "
            f"prediction={prediction:.4f}, "
            f"target={target}"
        )

    test_loss = test_squared_error / len(test_examples)
    print(f"Test loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()