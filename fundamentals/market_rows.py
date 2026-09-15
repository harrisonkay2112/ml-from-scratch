import csv
from pathlib import Path


def percent_change(current, previous):
    return (current / previous - 1) * 100


def main():
    # These prices and volumes are invented for this exercise.
    # Locate the CSV beside this script, regardless of the terminal's folder.
    data_path = Path(__file__).with_name("sample_market.csv")

    # DictReader uses the first CSV line as column names.
    with data_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    print("Synthetic data: not actual stock observations.")
    print("Imagine making each prediction after today's market close.\n")

    training_examples = []

    # Each example needs yesterday, today, and the next session.
    # Skip the first and last rows because a neighbor is missing.
    for index in range(1, len(rows) - 1):
        yesterday = rows[index - 1]
        today = rows[index]
        next_session = rows[index + 1]

        # CSV values start as strings. float converts them to numbers.
        # Input 1: today's percentage price change from yesterday's close.
        x1 = percent_change(float(today["close"]), float(yesterday["close"]))

        # Input 2: today's percentage volume change from yesterday.
        x2 = percent_change(float(today["volume"]), float(yesterday["volume"]))

        # Target: the NEXT session's close-to-close percentage return.
        # This is known in historical data, but unknown at prediction time.
        # It must never be passed into predict() as an input.
        target = percent_change(float(next_session["close"]), float(today["close"]))

        # Same (x1, x2, target) structure as our multiple regression exercise.
        training_examples.append((x1, x2, target))
        print(
            f"{today['date']}: x1={x1:+.2f}%, x2={x2:+.2f}%, "
            f"target={target:+.2f}% ({next_session['date']})"
        )

    print(f"\nBuilt {len(training_examples)} examples. No training happens yet.")
    # These are percentage points: 2.0 means 2%, not 200%.
    # Before training, we will learn to scale inputs and split by date.
    # This timing exercise is not a simulation of executable trades.


if __name__ == "__main__":
    main()
