# Digit Recognition with Scikit-learn

This project builds a handwritten digit classifier using the built-in `load_digits` dataset from `scikit-learn`. It trains a small neural network, saves the trained model with `joblib`, and includes a prediction script that tests the model on a sample image from the test set.

## Features

- Loads and normalizes the `scikit-learn` digits dataset
- Splits data into training and test sets
- Trains an `MLPClassifier` for digit classification
- Saves the trained model as `digit_model.joblib`
- Runs a sample prediction from the test set
- Saves a preview image as `prediction_sample.png`

## Project Structure

```text
digit-recognition/
|-- dataset.py
|-- model.py
|-- train.py
|-- predict.py
|-- requirements.txt
|-- digit_model.joblib
|-- prediction_sample.png
`-- README.md
```

## How It Works

`dataset.py`
- Loads the handwritten digits dataset
- Normalizes pixel values by dividing by `16.0`
- Splits the dataset into train and test sets

`model.py`
- Defines the neural network using `MLPClassifier`
- Uses two hidden layers: `128` and `64`
- Enables early stopping for quicker, more stable training

`train.py`
- Trains the model on the training set
- Evaluates it on the test set
- Prints validation accuracy and a classification report
- Saves the trained model to `digit_model.joblib`

`predict.py`
- Loads the saved model
- Selects one sample from the test set
- Predicts the digit
- Saves an enlarged grayscale preview to `prediction_sample.png`

## Installation

1. Move into the project folder:

```bash
cd digit-recognition
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

The project uses:

- `scikit-learn`
- `numpy`
- `joblib`
- `Pillow`

## Usage

### Train the model

```bash
python train.py
```

Example output:

```text
Model saved to: .../digit_model.joblib
Validation accuracy: 0.97xx
Classification report:
...
```

### Run a prediction

```bash
python predict.py
```

Example output:

```text
Predicted digit: 5
Actual digit: 5
Saved preview to: .../prediction_sample.png
```

## Output Files

- `digit_model.joblib`: saved trained model
- `prediction_sample.png`: enlarged image of the predicted test sample

## Notes

- The dataset used here is the small built-in digits dataset from `scikit-learn`, not the larger MNIST dataset.
- Images are `8x8` grayscale digits.
- If `digit_model.joblib` is missing, run `python train.py` before using `python predict.py`.

## Future Improvements

- Add command-line arguments for selecting different test samples
- Allow prediction on custom handwritten digit images
- Add model comparison with SVM or Random Forest
- Save training metrics to a log file

## License

This project is open for personal learning and experimentation.
