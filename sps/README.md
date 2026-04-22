# SPS - Human Activity Recognition with LSTM

SPS is a machine learning project for recognizing human activities from sensor readings using a deep learning LSTM model. The project includes a simple command-line interface for training the model, saving it, and running predictions on sample sensor data.

The system is designed around a Human Activity Recognition (HAR) workflow using 9 sensor-style input features and 6 activity classes.

## Activities

The model predicts one of the following activities:

| Label | Activity |
| --- | --- |
| 0 | WALKING |
| 1 | WALKING_UPSTAIRS |
| 2 | WALKING_DOWNSTAIRS |
| 3 | SITTING | 
| 4 | STANDING |
| 5 | LAYING |

## Features

- LSTM-based deep learning model for activity classification
- StandardScaler-based preprocessing
- Training and prediction scripts
- Saved model support using Keras `.h5` format
- Menu-based command-line interface
- Modular project structure for easy extension

## Project Structure

```text
sps/
|-- data/
|   `-- data.csv              # Sensor dataset with activity labels
|-- main.py                   # Menu-based CLI entry point
|-- model.py                  # LSTM model architecture
|-- train.py                  # Model training pipeline
|-- predict.py                # Prediction script
|-- utils.py                  # Preprocessing and reshaping helpers
|-- requirements.txt          # Python dependencies
`-- README.md                 # Project documentation
```

After training, the project creates:

```text
sps/
`-- saved_model/
    `-- har_model.h5          # Trained Keras model
```

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn

## Installation

1. Open a terminal in the project folder:

```bash
cd sps
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Run the menu-based application:

```bash
python main.py
```

You will see these options:

```text
1. Train LSTM Model
2. Predict Activities
3. View Project Info
4. Exit
```

## Train the Model

You can train from the menu or run the training script directly:

```bash
python train.py
```

The training script:

1. Loads `data/data.csv`
2. Separates features and activity labels
3. Scales feature values with `StandardScaler`
4. Reshapes data for LSTM input
5. Splits data into training and testing sets
6. Trains the LSTM model for 10 epochs
7. Saves the trained model to `saved_model/har_model.h5`

## Run Predictions

After training, run:

```bash
python predict.py
```

The prediction script loads the saved model and predicts activities for sample rows from `data/data.csv`.

Example output:

```text
Sample 1: Predicted = WALKING (confidence: 0.89)
Sample 2: Predicted = WALKING (confidence: 0.86)
```

## Model Architecture

The model is defined in `model.py`:

```text
LSTM(64, return_sequences=True)
Dropout(0.2)
LSTM(32)
Dropout(0.2)
Dense(32, activation='relu')
Dense(num_classes, activation='softmax')
```

Training configuration:

| Setting | Value |
| --- | --- |
| Optimizer | Adam |
| Loss | Sparse categorical crossentropy |
| Metric | Accuracy |
| Epochs | 10 |
| Batch size | 32 |
| Test split | 20% |

## Dataset Format

The dataset should be stored at:

```text
data/data.csv
```

Expected format:

```csv
feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8,feature9,activity
0.98,0.04,-0.2,0.01,-0.1,0.15,0.02,-0.05,0.1,0
```

The final column, `activity`, must contain numeric labels from `0` to `5`.

## Notes

- Train the model before running `predict.py`.
- The saved model path is `saved_model/har_model.h5`.
- For better real-world performance, replace the sample dataset with a larger HAR dataset such as the UCI Human Activity Recognition dataset.
- The current preprocessing fits a new scaler each time data is processed. For production use, save the fitted scaler during training and reuse it during prediction.

## Future Improvements

- Save and load the fitted scaler for consistent inference
- Add model evaluation metrics such as confusion matrix and classification report
- Add support for real smartphone sensor data
- Add a Flask or FastAPI backend
- Add a simple web dashboard for predictions
- Convert the trained model to TensorFlow Lite for mobile or edge deployment

