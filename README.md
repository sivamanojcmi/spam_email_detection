# Email Spam Detection

## Aim

This project is aimed at detecting spam emails using machine learning techniques.

## Dataset Description

The dataset consists of 5728 emails, with each email labeled as spam or non-spam. The distribution of spam and non-spam emails is as follows:

- Spam emails: 23.9%
- Non-spam emails: 76.1%

## Data Split

The dataset was split into three subsets: training, validation, and test sets. The distribution of the data split is as follows:

- Training set: 70% of the dataset
- Validation set: 10% of the dataset
- Test set: 20% of the dataset

The training set was used to train the models, while the validation set was used for model selection. Finally, the test set was used to evaluate the performance of the tuned models.

## Feature Extraction

The text data underwent preprocessing, and additional feature extracetd that is the number of words in each email. Statistical analysis revealed that the number of words feature significantly impacts the classification of spam emails.

## Observation

An interesting observation from the dataset analysis is that the average number of words in non-spam (ham) emails tends to be higher than in spam emails. Additionally, within the spam emails, there is a wide variation in the distribution of the number of words

## Preprocessing Steps

The following preprocessing steps were applied to the email data:

1. Convert text to lowercase.
2. Remove punctuation.
3. Remove numbers from the text.
4. Tokenize the text.
5. Remove stop words, including custom stop words such as "subject" and "re".
6. Lemmatize words to their base form.
7. Remove single-character words.
8. Join tokens back into strings.

## Usage

To use this project, follow these steps:

1. Clone the repository to your local machine.
2. Open `prepare.ipynb` in a Jupyter Notebook environment.
3. Execute the code cells to preprocess the data and generate the train, validation, and test CSV files.
4. Use the generated CSV files for training and evaluating your spam detection model.
5. Open `train.ipynb` in a Jupyter Notebook environment.
6. Execute the code cells to run the model.
   

## Files

- `prepare.ipynb`: Jupyter Notebook containing the preprocessing and data preparation code.
- `train.csv`: CSV file containing the training data.
- `validation.csv`: CSV file containing the validation data.
- `test.csv`: CSV file containing the test data.
- `train.pynb`: Jupyter Notebook containing the Modelling and results.
  

## Model Selection and Evaluation

### Feature Vectorization

The dataset was split into training, validation, and test sets, and the text data was vectorized using the Term Frequency-Inverse Document Frequency (TF-IDF) method. Additionally, a new feature representing the number of words in each email was added to the feature vector.

### Model Evaluation

Five different models were trained on the training data and evaluated on the validation data: Multinomial Naive Bayes, Decision Trees, Logistic Regression, Random Forest, and Support Vector Classifier (SVC). The evaluation metric used was the F beta score, with beta set to 0.8 to emphasize precision and minimize false positives (labelling nonspam as spam).

### Model Selection

Based on the evaluation results, Decision Trees, Logistic Regression, and Random Forest models were selected for further tuning due to their promising performance.

### Hyperparameter Tuning

The selected models were hyperparameter tuned using cross-validation to improve their performance further. Grid search was used to search for the best hyperparameters for each model.

### Performance Metric

The F beta score was chosen as the performance metric due to its ability to balance precision and recall, with emphasis placed on precision to minimize false positives. A beta value of 0.8 was used to prioritize precision while still considering recall.

### Precision Recall Curve

In addition to evaluating the models based on the F beta score, the Precision-Recall Curve was plotted for visual comparison. This curve provides a graphical representation of the trade-off between precision and recall for each model.

### Model Comparison

The Precision-Recall Curve allows for easy identification and comparison of the models' performance. Models with higher precision and recall values will exhibit curves closer to the upper-right corner of the plot, indicating superior performance.

### Final Results

After hyperparameter tuning, the models were evaluated on the test set. The tuned models achieved the following F beta scores and AUC of PR Curve:

| Model               | F beta Score | AUC of PR Curve |
|---------------------|--------------|-----------------|
| Decision Trees      |    0.913     |     0.92        |
| Logistic Regression |    0.983     |     0.98        |
| Random Forest       |    0.978     |     0.98        |

The performance of these models demonstrates their effectiveness in classifying spam emails. The logistic regression model, in particular, achieved the highest F beta score and AUC of PR Curve, making it the top-performing model for this task. Additionally, the `Logistic Regression` model offers simplicity and ease of interpretation.


### Conclusion

By evaluating the models using both quantitative metrics (F beta score) and graphical methods (Precision-Recall Curve), we gained a comprehensive understanding of their performance in classifying spam emails. The combination of these evaluation techniques ensures robust model selection and validation, leading to reliable results and insights.

## Next Steps

With the selected and tuned models, the next steps include:

- Deploying the models for real-time email spam detection.
- Monitoring model performance and making adjustments as needed.
- Exploring additional features or model architectures to further improve classification accuracy.

