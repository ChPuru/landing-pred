# Project 2: The Professional-Grade Ensemble Predictor

This project represents the final, state-of-the-art version of our CanSat landing zone predictor. It is a complete, professional-grade ground station application that uses a robust, hybrid system of advanced physics, multiple machine learning models, and statistical analysis to provide the most accurate and reliable landing predictions possible.

## Project Philosophy

The core philosophy is that no single prediction method is perfect. This system is designed as a "board of expert advisors" who all analyze the flight data simultaneously, with a final "CEO" model that intelligently fuses their wisdom into a single, high-confidence prediction. This ensemble approach ensures maximum accuracy and robustness.

## Key Features & Innovations

- **Advanced Physics Engine:** Utilizes an iterative, second-by-second simulation with a layered wind model and dynamic air density calculations to provide a highly accurate physics-based baseline.

- **Diverse Machine Learning Ensemble:** Employs two distinct and powerful ML models:
    1. **XGBoost (Extreme Gradient Boosting):** A state-of-the-art model that excels at analyzing a snapshot of the current flight data with high accuracy. This is a direct upgrade from a standard Random Forest.
    2. **LSTM (Long Short-Term Memory) Neural Network:** A deep learning model that excels at understanding patterns, trends, and momentum over the last 30 seconds of flight.

- **Stacked Generalization (The "Meta-Model"):** The predictions from all three "experts" (Physics, XGBoost, LSTM) are fed into a final Linear Regression model. This "meta-model" has been trained to learn the optimal way to combine the predictions, intelligently trusting each expert more or less depending on the flight conditions.

- **Professional Data Pipeline:** The entire system is built on a robust data pipeline that prevents common ML errors. This includes:
  - **Group-Aware Data Splitting:** Prevents "data leakage" by ensuring data from any single flight does not appear in both the training and testing sets.
  - **Feature Scaling:** All data is normalized before being fed to the neural network, significantly improving its learning and performance.

## Architecture: The Prediction Pipeline

For every telemetry packet received, the system performs a multi-stage analysis:

1. **The Ensemble:** The Physics, XGBoost, and LSTM models all independently calculate a predicted landing spot.
2. **The Fusion:** The three predictions are fed into the Meta-Model, which outputs a single, fused, high-confidence coordinate.
3. **The Display:** The final, fused prediction is displayed on the screen, providing the team with the single best estimate of the landing zone.

## File Structure

- `config.py`: The single, master configuration file for all project settings.
- `utils.py`: Contains helper functions for logging and calculating geodesic error.
- `physics_engine_advanced.py`: The advanced physics model, serving as one of our "experts."
- `generate_rich_dataset.py`: The script to create the advanced dataset needed to train our AI models.
- `data_preparation.py`: **(New in this version)** A critical script that correctly splits the data and applies feature scaling.
- `train_ensemble.py`: The master script that trains all three ML models (XGBoost, LSTM, and the Meta-Model) and saves them.
- `main_app_ultimate.py`: The final, real-time application that runs the full prediction pipeline.

## Setup & Usage

1. **Install Libraries:** Create a `requirements.txt` file with the necessary libraries and run:

    ```bash
    pip install -r requirements.txt
    ```

2. **Generate the Rich Dataset:**

    ```bash
    python generate_rich_dataset.py
    ```

3. **Prepare the Data for ML:** (This is a crucial new step)

    ```bash
    python data_preparation.py
    ```

4. **Train the Ensemble of Models (This will take a few minutes):**

    ```bash
    python train_ensemble.py
    ```

5. **Run the Ultimate Predictor:**

    ```bash
    python main_app_ultimate.py
    ```

## Future Work & Potential Improvements

While this system is functionally complete, the next steps in a real-world project would be:

- **Hyperparameter Tuning:** Systematically tuning the parameters of the XGBoost and LSTM models to squeeze out the last few percent of accuracy.
- **Real-World Data:** Re-training the entire system on data collected from actual CanSat test flights instead of simulated data.
- **GUI Development:** Integrating this Python backend with a graphical user interface (like your MERN stack GUI) to display the predictions on a live map.
