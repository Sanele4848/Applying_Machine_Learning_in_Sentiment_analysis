# Applying Machine Learning to Sentiment Analysis

This project provides a comprehensive walkthrough of building a sentiment analysis model for movie reviews. It uses the Large Movie Review Dataset to classify reviews as either positive or negative. The notebook covers essential Natural Language Processing (NLP) and machine learning techniques, from basic text cleaning to advanced topic modeling.

-----

## Notebook Workflow

The notebook is structured to guide the user through the entire process of a text classification project.

### 1\. Data Preparation

  * **Downloads and Extracts** the IMDb movie review dataset, which contains 50,000 reviews.
  * **Processes and combines** the raw text files from the 'train' and 'test' directories into a single Pandas DataFrame.
  * **Assigns labels**: `1` for positive reviews and `0` for negative reviews.
  * The final prepared dataset is saved to `movie_data.csv` for easier access in future sessions.

### 2\. Text Preprocessing and Feature Extraction

  * **Bag-of-Words (BoW) Model**: Introduces the concept of converting text into numerical feature vectors using `CountVectorizer`.
  * **Term Frequency-Inverse Document Frequency (TF-IDF)**: Explains how to use `TfidfTransformer` to down-weight frequently occurring words that are less informative.
  * **Text Cleaning**: A custom `preprocessor` function is defined to remove HTML tags and non-alphabetical characters.
  * **Tokenization and Stemming**: The notebook demonstrates how to break down text into individual words (tokens) and reduce them to their root form using the Porter stemmer. It also covers the removal of common English "stop words" (e.g., 'and', 'the', 'a').

### 3\. Training a Sentiment Analysis Model

  * The dataset is split into training (25,000 reviews) and testing (25,000 reviews) sets.
  * A **Scikit-learn `Pipeline`** is constructed to chain the text vectorization (`TfidfVectorizer`) and classification (`LogisticRegression`) steps.
  * **Hyperparameter Tuning**: `GridSearchCV` is used to systematically find the best combination of parameters (like n-gram range, stop words, and regularization strength `C`) to achieve the highest model accuracy through 5-fold cross-validation.

### 4\. Handling Large Datasets (Out-of-Core Learning)

  * For datasets too large to fit in memory, this section demonstrates an **out-of-core learning** approach.
  * It uses `HashingVectorizer` for memory-efficient feature hashing and `SGDClassifier` (Stochastic Gradient Descent) to train the model incrementally on mini-batches of data streamed from the disk.

### 5\. Topic Modeling

  * The final section explores unsupervised learning to discover underlying topics in the movie reviews.
  * **Latent Dirichlet Allocation (LDA)** is applied to the text corpus to identify 10 distinct topics.
  * The top words for each topic are printed, allowing for interpretation of the themes (e.g., horror movies, war movies, comedies).

-----

## 🛠️ Key Concepts Demonstrated

  * **Sentiment Analysis**: Classifying text as positive or negative.
  * **Text Preprocessing**: Cleaning, tokenization, stemming, and stop-word removal.
  * **Feature Engineering**: Bag-of-Words and TF-IDF.
  * **Model Training**: Logistic Regression.
  * **Hyperparameter Optimization**: `GridSearchCV`.
  * **Workflow Management**: Scikit-learn `Pipeline`.
  * **Big Data Techniques**: Out-of-core learning with `HashingVectorizer` and `SGDClassifier`.
  * **Unsupervised Learning**: Topic modeling with Latent Dirichlet Allocation (LDA).

-----

## How to Run

1.  **Install Dependencies**: Make sure you have the required Python libraries installed.

    ```bash
    pip install pandas numpy scikit-learn pyprind nltk
    ```

2.  **Download NLTK Stopwords**: The first time you run this, you'll need to download the list of stopwords from NLTK. The notebook includes a cell for this:

    ```python
    import nltk
    nltk.download('stopwords')
    ```

3.  **Execute the Notebook**: Run the cells in the Jupyter Notebook `Chapter_8_Applying_Machine_Learning_To_Sentiment_Analysis.ipynb` sequentially.

    > **Note:** The `GridSearchCV` cell can take a significant amount of time (30-60 minutes) to run, as it trains many models to find the best one. The notebook provides suggestions for reducing the parameter grid or dataset size for a faster run.
