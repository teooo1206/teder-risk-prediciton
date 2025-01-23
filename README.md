### Summary of the Project: Analysis of Corruption Risk Index for Public Procurement Procedures in the UK

This project focused on analyzing the **Corruption Risk Index (CRI)** for public procurement procedures in the UK, using a dataset that included harmonized contract data from various procurement categories such as services, works, and supplies. The goal was to identify patterns and risk factors associated with corruption in public procurement, leveraging **Python** and various **machine learning techniques** to build predictive models and uncover insights.

#### Key Skills and Techniques Applied:
1. **Data Collection and Preparation**:
   - The dataset was sourced from a public repository and cleaned to handle missing values and inconsistencies.
   - **Pandas** and **NumPy** were used for data manipulation, including filtering, dropping null values, and creating new features.

2. **Exploratory Data Analysis (EDA)**:
   - **Seaborn** and **Matplotlib** were used to visualize data distributions, correlations, and trends.
   - Key insights included identifying the predominance of service-oriented procurement, the popularity of the MEAT (Most Economically Advantageous Tender) selection method, and the positive correlation between bid prices and final tender prices.

3. **Feature Engineering**:
   - New features were created to capture business logic and improve model performance, such as:
     - **Total Requirements Length**: Sum of description, personal, technical, and economic requirements.
     - **Price Risk Index**: Product of final price and CRI.
     - **Requirement Density**: Ratio of CPV code count to total requirements length.
     - **Price Difference**: Difference between final price and bid price.
   - Categorical variables were encoded, and boolean values were converted to numerical ones for modeling.

4. **Model Selection and Training**:
   - **Unsupervised Learning (K-Means Clustering)**:
     - Used to group tenders into clusters based on CRI and final price, identifying low-risk, medium-risk, and high-risk tenders.
     - Visualized clusters to understand the distribution of risk and price.
   - **Regression Models**:
     - **Linear Regression** was used as a baseline model, but it showed limited explanatory power (R² = 0.067).
     - **Gradient Boosting Regressor** improved performance (R² = 0.37), capturing non-linear relationships better.
   - **Decision Tree and Random Forest**:
     - **Random Forest** achieved the highest R² score (0.68), with key predictors being decision period and submission period.
     - Feature importance analysis highlighted the significance of time-related variables in predicting corruption risk.
   - **Neural Network**:
     - A custom neural network was built using **TensorFlow**, achieving an R² score of 0.58.
     - The model included dropout layers to prevent overfitting and was trained with early stopping to optimize performance.

5. **Performance Evaluation**:
   - Models were evaluated using **R-squared** and **Mean Squared Error (MSE)**.
   - The **Random Forest** model performed the best, explaining 68% of the variance in CRI, while the **Neural Network** was the most promising for capturing non-linear patterns in corruption risk.

6. **Limitations and Future Work**:
   - The linear regression model struggled with non-linear relationships, and the inclusion of derived features like `corr_singleb` and `corr_subm` inflated performance metrics.
   - **K-Means clustering** was limited by its reliance on Euclidean distance, suggesting the need for alternative clustering methods like **DBSCAN** or **GMM**.
   - The **Random Forest** model's reliance on time-related variables may limit its generalizability, and further validation is needed to ensure robustness.
   - The **Neural Network** could be improved by expanding the feature set to include bidder information, market conditions, and sector-specific risk factors.

#### Key Insights and Recommendations:
- **Decision and submission periods** were identified as critical predictors of corruption risk, suggesting that streamlining these processes could reduce corruption.
- **Buyer concentration** and **procurement requirements** also played significant roles in determining corruption risk.
- The **Neural Network** model, while not perfect, showed potential for real-time risk detection in procurement processes.
- Future work should focus on **expanding the feature set**, improving model interpretability, and incorporating **ensemble methods** for more robust predictions.

#### Skills Gained:
- **Data Cleaning and Preparation**: Handling missing data, feature engineering, and data transformation.
- **Exploratory Data Analysis (EDA)**: Visualizing data distributions, correlations, and trends using **Seaborn** and **Matplotlib**.
- **Machine Learning Modeling**: Building and evaluating models using **K-Means clustering**, **Linear Regression**, **Gradient Boosting**, **Random Forest**, and **Neural Networks**.
- **Model Evaluation**: Using metrics like **R-squared** and **MSE** to assess model performance.
- **Feature Engineering**: Creating new features to improve model performance and capture business logic.
- **Neural Network Development**: Building and training a custom neural network using **TensorFlow**, including techniques like dropout and early stopping to prevent overfitting.

This project provided hands-on experience in applying **machine learning** techniques to a real-world problem, highlighting the importance of **data preparation**, **model selection**, and **performance evaluation** in predictive analytics.
