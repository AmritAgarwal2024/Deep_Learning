# Stock Price Prediction and Impact Analysis Using LSTM Model

Contributors:

*   Amrit Agarwal: 055004
*   Oishik Banerjee: 055028

Group No: 23

## 1. Objective

This project aims to develop a Long Short-Term Memory (LSTM) based deep learning model to forecast NIFTY 50 and Tata Steel stock prices using historical daily market data. 
The model seeks to provide insights into stock price movements and analyze how Tata Steel's stock performance influences the NIFTY 50 index.

## 2. Problem Statement

The main challenge addressed is predicting stock prices using time series data, which involves capturing complex patterns and trends. Additionally, this project evaluates how fluctuations in a key stock (Tata Steel) impact the Nifty 50 index, considering its substantial market capitalization.

**Key Questions:**
- Can we accurately predict future stock prices using historical data?
- What is the correlation between NIFTY 50 and Tata Steel stock movements?
- How effectively does the LSTM model capture stock market trends compared to actual market fluctuations?


## 3. Analysis and Approach

### 3.1 Data Collection and Preprocessing

- **Data Source:** Yahoo Finance (yfinance package)
- **Stocks:** Nifty 50 Index (^NSEI) and Tata Steel (TATASTEEL.NS)
- **Date Range:** 2005-01-31 to 2025-01-31
- **Data Focus:** Closing prices(Only the closing prices were extracted, as they provide a reliable summary of the stock’s performance on a given day.)
- **Preprocessing:** Normalized using MinMaxScaler (0-1 range)

### 3.2 Sequence Creation for LSTM Model

- **Window Size:** 60 days(A window size of 60 days was chosen, meaning the model looks at the past 60 days’ closing prices to predict the next day’s price.)
- **Sequence Preparation:** Sliding window approach
- **Input (X):** 60 consecutive closing prices
- **Target (y):** Next day's closing price

### 3.3 Data Splitting

- **Train-Test Split:** 80% training, 20% testing
- **LSTM Input Shape:** (samples, time steps, features)


## 4. LSTM Model Architecture

- **Input Layer:** 60 time steps, 1 feature
- **LSTM Layers:** 3 layers (50 units each) with Dropout(0.2)(Dropout(0.2) applied after each LSTM layer to prevent overfitting.)
- **Dense Layers:** 1 hidden layer (25 units, ReLU activation), 1 output layer (1 unit)
- **Compilation:** Adam optimizer (learning rate = 0.001), MSE loss function, MAE metric

## 5. Model Training and Evaluation

- **Training:** 10 epochs(The model was trained for 10 epochs, with training and validation losses reducing steadily over epochs.)
- **Training Time:** ~13 seconds per epoch
- **Performance:** Decreasing training and validation losses across epochs

**Training Loss:** The training loss (MSE) consistently decreased across epochs, indicating that the model was learning effectively.

**Validation Loss:** Validation loss followed a similar trend, suggesting the model generalized well on unseen data.

## 6. Observations

### 6.1 NIFTY 50 Prediction
The LSTM model effectively captured stock price trends but showed minor deviations. Prediction lag was observed, indicating a limitation in highly volatile market conditions. The predicted vs. actual plot showed a good correlation, but slight under-prediction in high peaks.

**INSIGHTS:**

1.  **Strong Overall Trend Capture**
    *   The predicted prices (orange line) closely follow the actual price movements (blue line), indicating that the LSTM model effectively captured the overall upward trend of the Nifty 50 index over the years.
    *   Although some deviations are visible, especially during periods of sharp fluctuations, the model generally aligns well with the actual trend.

2.  **Underestimation During Volatile Periods**
    *   The model tends to underestimate the price during highly volatile periods, such as market corrections or sharp rallies.
    *   This is evident in 2021-2022 and mid-2023, where the predicted prices lag behind the actual prices during strong market swings. This suggests that while the model captures trends, it struggles with extreme price movements.

3.  **Better Performance During Stable Growth Phases**
    *   The prediction accuracy improves significantly during stable market growth phases.
    *   From early 2023 to mid-2024, the predicted price is consistently close to the actual price, reflecting the model’s ability to perform better when the market is relatively stable.

4.  **Lag in Price Prediction During Rapid Market Movements**
    *   There is a noticeable lag in predictions during rapid price surges or dips.
    *   For instance, during sharp upward movements in early 2024, the predicted price takes some time to adjust to the new trend, highlighting that the model is better suited for predicting gradual price changes rather than sudden market shifts.

**Overall**
- Effective trend capture with minor deviations
- Slight prediction lag in volatile conditions
- Good correlation between predicted and actual prices
- Underestimation during high volatility periods
- Better performance in stable growth phases



### 6.2 Tata Steel Prediction

The trend of Tata Steel's stock price was more volatile compared to NIFTY 50. The LSTM model predicted the movements well, but the fluctuations were slightly smoothed. The model worked better for long-term trend estimation rather than short-term spikes.

**INSIGHTS:**

1.  **Good Overall Trend Estimation**
    *   The model successfully captured the overall trend of Tata Steel’s stock price over time.
    *   Both the actual and predicted prices follow an upward trajectory, indicating that the LSTM model effectively learned the long-term patterns and price movements of Tata Steel.

2.  **Consistent Underestimation During High Volatility**
    *   Similar to the Nifty 50 prediction, the model tends to underestimate prices during periods of high volatility, especially in the sharp upward or downward movements observed between mid-2022 and early 2024.
    *   This highlights the model's limitations in responding to rapid and large price swings, indicating that it is better at predicting smoother trends rather than extreme fluctuations.

3.  **Improved Prediction in Moderate Price Movements**
    *   The model performs well when the stock price exhibits moderate fluctuations, as seen during 2021 and 2022.
    *   During these periods, the predicted price aligns closely with the actual price, suggesting that the LSTM model is more reliable during relatively stable price periods.

4.  **Lag in Capturing Price Peaks and Dips**
    *   The model shows a lag effect when predicting price peaks and dips.
    *   During periods of sharp increases or decreases in stock prices, such as in 2023 and early 2024, the predicted price line trails the actual price, suggesting that the model takes time to adapt to rapid changes in market conditions.
**Overall**

- More volatile trend compared to NIFTY 50
- Good movement prediction with smoothed fluctuations
- Better long-term trend estimation
- Consistent underestimation during high volatility
- Improved prediction in moderate price movements
- Lag in capturing price peaks and dips

### 6.3 Comparison of NIFTY 50 & Tata Steel

The correlation analysis showed that Tata Steel's stock price had a significant impact on NIFTY 50. Graphical analysis demonstrated that Tata Steel's stock movements often preceded or coincided with movements in NIFTY 50, supporting the hypothesis that large-cap stocks influence index behavior.

**INSIGHTS:**

1.  **Strong Correlation Between Nifty 50 and Tata Steel**
    *   A high positive correlation is evident between the movement of Tata Steel’s stock price (red line) and the Nifty 50 index (blue line), especially after 2022.
    *   As Tata Steel is a heavyweight in the Nifty 50 index, its price movement often mirrors and significantly influences the Nifty 50 index.

2.  **Divergence During Initial Years, Strong Convergence Post-2022**
    *   From 2010 to 2021, Tata Steel's stock price remained relatively flat, while the Nifty 50 index showed moderate growth.
    *   However, post-2020, the stock price of Tata Steel exhibited a sharp increase and closely followed the upward trend of the Nifty 50 index, indicating its growing dominance in the market.

3.  **Rapid Surge in Tata Steel’s Stock Post-2022**
    *   There was a steep rise in Tata Steel's stock price starting in 2022, outpacing the Nifty 50 index in terms of growth rate.
    *   This period coincides with [*insert potential Tata Steel specific event, e.g., strategic expansion, infrastructure projects, etc.*], which had a significant positive impact on its market performance.

4.  **Tata Steel Stock Driving Market Movement During Volatility**
    *   Periods of sharp fluctuations in Tata Steel's stock price were often accompanied by similar volatility in the Nifty 50 index.
    *   Notably, the dip during the COVID-19 pandemic in early 2020 shows synchronized downward movement in both the Nifty 50 and Tata Steel, reinforcing the influence of Tata Steel on the overall market.But after the COVID outbreak it bounced backed strongly showing a clear peak in this last 18 years.

5.  **Mutual Impact During Market Peaks and Dips**
    *   Major peaks and dips observed in the graph (such as during 2020 and 2024) show that Tata Steel and the Nifty 50 index tend to rise and fall in tandem, emphasizing Tata Steel’s role as a bellwether for the Indian stock market.
    *   This synchronization suggests that tracking Tata Steel’s stock price can serve as a leading indicator for broader market trends.
**Overall**

- Significant impact of Tata Steel on NIFTY 50
- Strong correlation, especially post-2022
- Divergence in initial years, strong convergence after 2022
- Rapid surge in Tata Steel's stock post-2022
- Synchronized movements during market volatility

### 6.4 Comparison of Predicted Prices of NIFTY 50 & Predicted Stock Prices of Tata Steel

The graph shows a strong correlation between the predicted Nifty 50 index values (blue) and Tata Steel’s stock prices (red), with synchronized peaks and troughs. This suggests that predicted changes in Tata Steel’s stock price often align with shifts in the Nifty 50, reinforcing its influence on the broader market. Tracking Tata Steel’s movements can serve as a leading indicator for predicting Nifty 50 trends.

**INSIGHTS:**

1.  **Strong Correlation in Predicted Trends**
    *   The predicted prices of Tata Steel and the Nifty 50 index exhibit a high degree of synchronization, with peaks and troughs occurring around the same time.
    *   This correlation suggests that Tata Steel's stock price continues to play a pivotal role in driving overall market trends.

2.  **Mutual Influence During Market Fluctuations**
    *   Both predicted series display cyclical behavior with noticeable fluctuations, indicating that changes in Tata Steel’s stock price are reflected in the predicted movements of the Nifty 50.
    *   Periods of upward or downward trends in Tata Steel’s stock price often coincide with similar trends in the Nifty 50 index.

3.  **Potential for Predictive Advantage**
    *   Since Tata Steel's stock price often shows trends before or simultaneously with the Nifty 50, monitoring predicted movements in Tata Steel can offer a leading indicator for broader market predictions.
    *   Investors can leverage this relationship to anticipate shifts in the Nifty 50 and adjust their portfolio strategies accordingly.

4.  **Divergence in Short-Term Volatility**
    *   While the long-term trends align well, there are periods of short-term divergence where the predicted price of Tata Steel shows higher volatility compared to the Nifty 50.
    *   This indicates that while Tata Steel’s stock drives the index, the Nifty 50 is also influenced by other components, leading to occasional deviations.
**Overall**

- Strong correlation between predicted Nifty 50 and Tata Steel prices
- Mutual influence during market fluctuations
- Potential for predictive advantage using Tata Steel's movements
- Short-term volatility divergence

## 8. Project Statistics

**Dataset:** Historical closing prices of Nifty 50 and Tata Steel  

**Date Range:** Data spanning from January 31, 2005, to January 31, 2025

**Train-Test Split:** 80% training, 20% testing.

**Sequence Length:** Sliding window of past 60 days  

**Model Architecture:** LSTM with three layers and dropout regularization  

**LSTM Layers:** 3 layers with 50 units and dropout regularization.

**Optimizer:** Adam (learning rate = 0.001).


**Loss Function:** Mean Squared Error (MSE).

**Evaluation Metric:** MAPE (Mean Absolute Percentage Error)  

**Model Accuracy:**
 - Nifty 50 Prediction: Achieved an accuracy of ~97.78%
 - Tata Steel Prediction: Achieved an accuracy of ~89.23%%

## 9. Conclusion and Future Scope

The project successfully demonstrated LSTM neural networks' ability to predict stock prices with high accuracy using historical data. It highlighted Tata Steel's significant impact on the Nifty 50 index, providing valuable insights for market analysts and investors.

**Future Scope:**
- Incorporate external factors such as trading volume, sentiment analysis, and economic indicators.

- Experiment with ensemble models for enhanced prediction accuracy.

- Multi-variate Analysis: Incorporate additional variables such as trading volume, P/E ratios, and dividend yields to enhance prediction accuracy.

- Real-time Prediction System: Develop a system that can provide near real-time predictions by continuously updating the model with incoming market data.

- Cross-market Analysis: Extend the model to analyze correlations between different stock markets globally, providing insights into international market dynamics.

- Alternative Data Integration: Explore the use of alternative data sources such as satellite imagery, credit card transactions, or social media sentiment to improve prediction accuracy.

- Explainable AI Techniques: Implement methods to interpret the LSTM model's decision-making process, enhancing trust and understanding in the predictions.

- Hybrid Models: Combine LSTM with other machine learning algorithms or traditional statistical methods to create more robust prediction systems.

- Automated Trading Systems: Develop AI-driven trading systems that can execute trades based on the model's predictions, subject to predefined risk parameters.

- Customized Investor Profiles: Create personalized prediction models tailored to individual investor risk profiles and investment goals.

- Blockchain Integration: Explore the use of blockchain technology for secure and transparent data sharing, potentially improving the model's access to diverse datasets.

- Transfer Learning: Investigate the application of transfer learning techniques to adapt the model quickly to new stocks or markets with limited historical data.
