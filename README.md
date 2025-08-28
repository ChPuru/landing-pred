# CanSat Landing Zone Prediction Projects

Welcome to our collection of landing prediction projects! This repository contains a series of Python applications, each demonstrating a different and increasingly sophisticated method for predicting our CanSat's landing zone in real-time.

The goal of this work is to develop a powerful and reliable ground station tool that can provide our recovery team with the most accurate possible landing information during a mission.

## The Prediction Approaches

My work is broken down into two distinct projects, each building upon the last.

### [Project 1: The Advanced Physics Predictor]

- **The Idea:** To create the best possible predictor using only physics and algorithms.
- **How it Works:** This model runs a high-speed, second-by-second simulation of the entire remaining descent. It uses a realistic "layered wind model" and calculates how changing air density affects the parachute's descent rate.

### [Project 2: The Ultimate AI Ensemble Predictor]

- **The Idea:** To build the most powerful and accurate system possible by combining multiple models.
- **How it Works:** This is our top-tier system. It gets predictions from three different "expert" models (an advanced physics engine, a Random Forest AI, and an LSTM neural network). A final "CEO" model then intelligently fuses their opinions to get the best possible answer, and even calculates a realistic "search radius" for the recovery team.

## How to Use This Repository

Each project is located in its own folder and is a standalone application. To explore an approach, please navigate to its folder and follow the instructions in its dedicated `README.md` file.

Our general workflow for each project is:

1. Generate a large set of simulated flight data.
2. (For AI projects) Train the machine learning models on this data.
3. Run the main application, which replays the flight data as if it were a live mission, showing the real-time predictions.

## REFRENCES

Movable Type Scripts - Calculate distance, bearing and more between Latitude/Longitude points: <https://www.movable-type.co.uk/scripts/latlong.html>
(This is a highly respected, clear, and citable source for geospatial formulas with code examples.)

NASA - The Drag Equation: <https://www.grc.nasa.gov/www/k-12/airplane/drageq.html>
NASA - U.S. Standard Atmosphere, 1976: <https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html>
(Citing NASA is the gold standard for aerospace principles.)

Towards Data Science - The Monte Carlo Simulation: A Practical Guide: <https://towardsdatascience.com/the-monte-carlo-simulation-a-practical-guide-88da86187763>
(This provides a clear, practical explanation of the concept and its application.)

Scikit-learn Official Documentation - Ensemble methods: <https://scikit-learn.org/stable/modules/ensemble.html>
TensorFlow Official Documentation - LSTMs: <https://www.tensorflow.org/guide/keras/rnn#lstms_and_grus>
(Always cite the official documentation for the tools you used. It is the most direct and professional reference.)```
