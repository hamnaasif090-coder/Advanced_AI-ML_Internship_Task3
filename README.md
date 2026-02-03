# Task 3: Multimodal Housing Price Prediction

## Objective
To predict housing prices by combining structured property data with visual features extracted from house images using a multimodal machine learning approach.

## Dataset
The dataset used in this project was sourced from Kaggle and consists of tabular housing data along with corresponding property images.

- Tabular data (CSV):  
  https://www.kaggle.com/datasets/ericpierce/austinhousingprices?select=austinHousingData.csv

- House images:  
  https://www.kaggle.com/datasets/ericpierce/austinhousingprices?select=homeImages

## Methodology / Approach
- Used a Convolutional Neural Network (CNN) to extract visual features from house images
- Preprocessed numerical and categorical tabular features
- Fused image embeddings with tabular features into a single regression model
- Trained the model end-to-end using PyTorch
- Evaluated performance using MAE and RMSE metrics

## Key Results or Observations
- Mean Absolute Error (MAE): approximately 153,000
- Root Mean Squared Error (RMSE): approximately 393,000
- The model successfully learned from both image and tabular data modalities
- Higher RMSE indicates sensitivity to high-priced property outliers and highlights potential areas for model improvement
