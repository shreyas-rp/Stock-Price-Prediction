# Stock Price Prediction
##  Stock Price Prediction using LSTM
This repository demonstrates stock price prediction using LSTM (Long Short-Term Memory) neural networks. We utilize historical stock price data for Apple Inc. (AAPL) obtained from Yahoo Finance and build an LSTM model to predict future stock prices. The model is trained on a portion of the data and tested on unseen data to evaluate its predictive performance.

## Getting Started
### Prerequisites
To run the code and experiments, you'll need the following:

  - Python (version 3.6 or later)    
  - Jupyter Notebook (to interact with the provided notebook)  
  - TensorFlow (for deep learning operations)   
  - Keras (for building and training the neural network) 
  - numpy (for array operations)   
  - matplotlib (for visualization)
  - pandas(Data manipulation and analysis)
  - yfinance(Fetch historical stock price data from Yahoo Finance)

You can install the required libraries using the following command:   
 ```bash
pip install tensorflow keras yfinance pandas numpy matplotlib


```
## Installation
To get started, follow these steps:   
  1. Clone the repository to your local machine:
```bash
git clone https://github.com/shreyas-rp/Stock-Price-Prediction
```
  2. Navigate to the project directory:
```bash
cd Stock-Price-Prediction
```
  3. Run the Jupyter Notebook file:
```bash
jupyter notebook Stock_Price_Prediction_(LSTM).ipynb
```
## Procedure 
1. **Data Collection:** Historical stock price data for AAPL is fetched from Yahoo Finance using the yfinance library. The data includes daily closing prices from '2012-01-01' to '2023-07-18'.  
2. **Data Preprocessing:** The closing price data is scaled using Min-Max normalization to bring it into the range of [0, 1], which is beneficial for training neural networks. We also split the data into training and testing sets. The training set contains 80% of the data, and the testing set contains the remaining 20%.  
3. **Model Architecture:** We build an LSTM neural network using Keras. The model consists of two LSTM layers with 50 units each, followed by two dense layers with 25 and 1 unit, respectively. The model is compiled using the Adam optimizer and the mean squared error (MSE) loss function. 
4. **Model Training:** The model is trained on the training dataset with a batch size of 1 and 10 epochs. The LSTM layers enable the model to capture temporal dependencies in the stock price data.
5. **Prediction:** The trained model is used to predict future stock prices. We select the last 60 days of closing prices from the testing dataset as input to the model for prediction. The predicted values are then scaled back to their original range using inverse transformation.   
6. **Visualization:** We visualize the actual and predicted stock prices on a plot for better understanding and evaluation.
# Results
  - Upon running the Jupyter Notebook, you'll see the training progress, model predictions, and visualizations of the actual and predicted stock prices.
    
  - Feel free to experiment with different model architectures and hyperparameters to improve prediction accuracy.
