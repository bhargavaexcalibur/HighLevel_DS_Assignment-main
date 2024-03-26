HighLevel_DS_Assignment
Assignment Steps:
1) Handling Missing Values:
a) ['Index', 'Title', 'publisher', 'categories', 'Impact']: 0 missing
b) 'description': 12749 missing - left empty as it's combined with the title for embeddings
c) 'authors': 2723 missing - left as an empty string handled by multi-hot encoding
d) 'publishedDate': 393 missing values - replaced with a default date

2) Handling Data Types:
a) 'publishedDate' to date format: varied formats ['1996', '01-01-2005', '2005-02']
b) Convert string representations of lists to actual lists for 'authors' and 'categories'.
c) Convert 'Impact' from string to double

3) Feature Engineering:
a) Title and Description combined: embeddings using Sentence Transformers
b) 'Authors' are dropped: reason ~120k unique authors
c) For 'categories', multi-hot encodings are generated.
d) Additional features related to the 'Title' column such as word count, average word length, character count, presence of digits, uppercase letters, punctuation, and sentiment score using TextBlob.

4) Model Training, Evaluation, and Logging:
a) Splits the data into training and testing sets.
b) Creates a GBT Regressor model using Spark's MLlib.
c) Sets up a pipeline that includes vector assembling, PCA, and the GBT Regressor.
d) Parameter grid for hyperparameter tuning (maxDepth and maxIter) using cross-validation.
e) Fits the model using the training data.
f) Logs model, parameters, training, and test Mean Absolute Percentage Error (MAPE) metrics to MLflow.
