# House Price Prediction

An applied Statistical Machine Learning project that uncovers the most significant drivers behind real estate values, employing advanced feature selection and robust Multiple Linear Regression techniques.

## Overview
Estimating property value accurately is a classic problem in data science and economics. This structured analytical pipeline takes raw housing attributes (area, bedrooms, bathrooms, proximity to main roads) and transforms them into predictive insights. The workflow emphasizes rigorous Exploratory Data Analysis (EDA), thoughtful data cleaning (outlier handling via IQR), and complex algorithmic feature selection leveraging `scikit-learn` and `statsmodels`.

## Features
- **Extensive EDA**: Uses native `matplotlib` and `seaborn` plotting (boxplots, pairplots, heatmaps) to detect collinearity and underlying data structures.
- **Robust Feature Engineering**: Automates binary mapping for categorical boolean fields and creates dynamically dropped dummy variables for multi-category columns (e.g., furnishing status).
- **Automated Feature Selection**: Implements Recursive Feature Elimination (RFE) natively resolving the most optimal variables before statistical fitting.
- **Statistical Rigor**: Extends standard SKLearn models through `statsmodels.api` to provide detailed Ordinary Least Squares (OLS) summaries and Variance Inflation Factor (VIF) indexing for analyzing multi-collinearity.
- **Dual Availability**: Accessible both as an interactive Jupyter Notebook (`housing.ipynb`) and a linear Python script (`housing.py`).

## Tech Stack
- Python
- Scikit-Learn
- Statsmodels
- Pandas & NumPy
- Seaborn & Matplotlib

## Project Architecture
```text
house-price-prediction-master/
  housing.ipynb      # Complete interactive analysis, visualizations, and modeling blocks
  housing.py         # The consolidated plain-text Python script running the entire pipeline
  Housing.csv        # Underlying structured dataset driving the predictions
```

## Installation
Ensure your Python environment possesses standard data science and machine learning dependencies:
```bash
pip install numpy pandas scikit-learn statsmodels matplotlib seaborn jupyter
```

## Running the Project
The prediction pipeline can be executed traditionally via terminal or explored interactively:

**Script Execution**:
```bash
python housing.py
```

**Interactive Exploration**:
```bash
jupyter notebook housing.ipynb
```

## Model Card

### Model Overview
The core algorithm is a Multiple Linear Regression model aimed at mapping discrete and continuous property characteristics against varying continuous property prices.

### Model Architecture
- **Data Scaling**: Min-Max Scaling applied to continuous inputs to restrict dimensions between $0$ and $1$, preventing over-weighted gradient descent logic on massively skewed integer attributes (like pure area).
- **Core Algorithm**: Non-regularized Ordinary Least Squares (OLS) targeting minimum residual divergence.

### Training Process
- The initial dataset features $100\%$ validation confirming zero null entries.
- Outlier truncation executed using the Interquartile Range (IQR) bounded approach for critical attributes.
- Sklearn subsets the population (70% Train, 30% Test, `random_state=100`) directly feeding into an RFE estimator.
- RFE whittles the features down to the defined optimal parameters which are consecutively evaluated via `statsmodels` for p-values and R-Squared optimization.

### Limitations
- The current implementation enforces strict non-regularizations bounds. Expanding to ElasticNet, Ridge, or Lasso regressions could inherently support dynamic feature dropping outside of iterative RFE approaches.
- OLS models remain acutely sensitive to outlier anomalies which necessitates heavy pre-processing.

## Professional Highlights
- **Demonstrated strong statistical intuition** by incorporating rigorous secondary tests like Variance Inflation Factor (VIF) and residual histogram analysis to validate regression assumptions (Normality of Errors, Homoscedasticity).
- **Maximized feature space efficiency** leveraging Scikit-Learn’s Recursive Feature Elimination (RFE) toolsets out of the box.
- **Showcased data manipulation supremacy** using deeply integrated Pandas methods (`apply`, `pd.get_dummies`, `concat`) effectively normalizing complex datasets for immediate mathematical interpretation.

## License
MIT License

## Contributing
Contributions are welcome. Feel free to open issues or submit pull requests for enhancements.

## Author
Lih Ingabo
