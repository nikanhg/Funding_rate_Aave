# High-Frequency Trend Prediction of Borrowing and Lending Rates on Aave

## Overview

This project aims to predict high-frequency trends in borrowing and lending rates for various coins and tokens on the Aave platform. By utilizing diverse data sources and advanced machine learning techniques, we provide insights into market trends that are critical for optimizing lending and borrowing strategies.

Our best-performing model, an Attention-based neural network, achieves a 10% improvement in F1 score compared to the baseline, making it highly effective for this use case.

---

## Folder Structure

- **Data Gathering**:

  - Data is sourced from:
    - Binance scraping
    - Aave API
    - Messari.io API
  - Scripts and notebooks:
    - `0_prices.ipynb`: Retrieves and preprocesses price data.
    - `1_rates.ipynb`: Collects borrowing and lending rate information.

- **Feature Engineering**:

  - Extensive preprocessing steps to transform raw data into model-ready features.
  - Notebook:
    - `5_Feature_Engineering.ipynb`: Handles data cleaning, feature generation, and selection.

- **Model Development**:
  - Multiple model architectures were explored:
    - Long Short-Term Memory (LSTM):
      - `3_LSTM_grid_search_v1.ipynb`
      - `4_LSTM_grid_search_v2.ipynb`
      - `5_LSTM_grid_search_v3.ipynb`
    - LSTM Encoder-Decoder:
      - `7_LSTM_Encoder_Decoder_Model.ipynb`
    - Variational Autoencoder (VAE): Experimental model for trend prediction.
    - Attention Model: Finalized as the best-performing model.
      - Results summarized in `8_Attention_Model_Results.ipynb`.

---

## Highlights

1. **Data Sources**:

   - Integrated multiple data sources for robust input.
   - Fetched historical rates, prices, and additional financial metrics.

2. **Feature Engineering**:

   - Designed features to capture temporal patterns.
   - Optimized data pipelines to streamline preprocessing.

3. **Modeling**:

   - Conducted extensive hyperparameter tuning for LSTM models.
   - Incorporated Variational Autoencoder (VAE) as an experimental architecture.
   - Attention-based model demonstrated significant performance improvements.

4. **Performance**:
   - Improved F1 score by 10% with the Attention model compared to baseline.

---

## Results

The Attention model's ability to predict borrowing and lending trends with a 10% higher F1 score demonstrates its effectiveness. Insights derived from this model can significantly impact decision-making for stakeholders in the Aave ecosystem.

---

## Future Work

- Extend the dataset to include additional market metrics.
- Explore advanced architectures like Transformer-based models.
- Fine-tune the Attention model for real-time prediction tasks.
