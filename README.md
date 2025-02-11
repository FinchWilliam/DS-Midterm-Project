# Data Science Midterm Project

## Project/Goals
### We are working in a team to process raw json files into a usable dataframe, then trial various machine learning algorithms to determine the optimal one to predict housing prices, lastly we will create a pipeline from what we have learned that will allow us to reproduce quickly and succintly.

## Process
### Load and Clean data
- Load in supplied data
- Drop unrequired columns
- Fill in all nulls as required
- Replace some categorical values with numerical (cities/state got replaced by longitude/latitude)
- One-hot encoding on the remaining categorical values

### Trial various Machine Learning Algorithms
To determine the best model for predicting housing prices, we experimented with multiple machine learning algorithms:
- Linear Regression – A simple baseline model for understanding linear relationships.
- Decision Trees – Used to capture non-linear patterns in the data.
- Random Forest – An ensemble method that reduces overfitting and improves generalization.
- Gradient Boosting (XGBoost & LightGBM) – Explored boosting techniques for improved predictive performance.
- Support Vector Machines – Attempted, but computationally expensive for large datasets.
Each model was evaluated using RMSE (Root Mean Squared Error) and R² scores to determine its effectiveness.
### Hyperparamater testing to streamline algorithms
To optimize model performance, we performed hyperparameter tuning using:
- GridSearchCV – Exhaustive search over specified hyperparameter values.
- RandomizedSearchCV – More efficient search for optimal hyperparameters.

Key hyperparameters tuned included:
- Learning rate, max depth, and number of estimators for boosting models.
- Minimum samples per leaf and maximum tree depth for decision trees.
- Regularization parameters (L1/L2) for linear regression.

### Pipeline building
- Create a pipeline that will load and clean data quickly and easily
- Create a second pipeline that will train the algorithm off of the data acquired from the first pipeline
    - Unfortunately these must remain as seperate pipelines so a train/test/split can be performed on the data between first and second pipelines.

## Results
The model's performance based on RMSE and R² scores is as follows:

- Best Model: Random forest
- RMSE: 223,739.53
- R² Score: 0.5045

## Challenges 
### Various challenges we faced included:
- Processing the json files
    - What columns are required and what are garbage? (the url of the images is garbage, but is the fact that it has images useful information?)
    - How to handle cities/states (do we One-Hot encode? do we Label Enconde?)
        - We decided on replacing the information with Latitude and Longitude
    - Handling the tags for the properties, we decided to go with One-Hot encoding on those and drop any tags that did not show up in the dataset more than 20 times
        - Further research into the optimal number of tags to drop is also on my wishlist for future goals

## Future Goals
### We would love to explore adding in additional data sources, various ones I was looking into but was lacking time (and resources as some cost money) to complete:
- Average income and age of the populations
- Major landmarks in proximity (forest/ocean/theme parks/etc..)
### Along with that we would have loved to play around more with feature selection to work out how much of the dataset was useful and how much could we drop without impact
