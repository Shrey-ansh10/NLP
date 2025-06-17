# FeelTrack: Comparative Sentiment Analysis using Vader & TextBlob

## Overview
This project, FeelTrack, focuses on comparative sentiment analysis of movie reviews using two popular NLP tools: **Vader** and **TextBlob**. The goal is to classify movie reviews as either positive or negative based on their textual content and to compare the effectiveness of both sentiment analysis techniques. The project demonstrates the end-to-end process of data collection, preprocessing, model building, evaluation, and interpretation of results, highlighting the strengths and weaknesses of each approach.

## Project Structure
- `Sentiment Analysis.ipynb`: Main Jupyter notebook containing code, methodology, and results.
- `Sentiment Analysis.html`: Exported HTML version of the notebook for easy viewing.
- `description.txt`: Brief description of the project.
- `txt_files/neg/`: Folder containing negative movie review text files.
- `txt_files/pos/`: Folder containing positive movie review text files.

## Technologies Used
- **Python**
- **Jupyter Notebook**
- **VaderSentiment**
- **TextBlob**
- **Natural Language Toolkit (NLTK)**
- **scikit-learn**
- **Pandas & NumPy**
- **Matplotlib/Seaborn** (for visualization)

## Methodology
1. **Data Collection**
   - The dataset consists of movie reviews categorized into positive and negative folders.
   - Each review is stored as a separate `.txt` file.

2. **Data Preprocessing**
   - Reading and aggregating all text files from both positive and negative folders.
   - Cleaning the text: removing punctuation, converting to lowercase, removing stopwords, and tokenization.
   - Optional: Stemming or lemmatization to normalize words.

3. **Sentiment Analysis**
   - Applying both **Vader** and **TextBlob** to each review to obtain sentiment scores.
   - Comparing the polarity and classification results from both tools.

4. **Feature Extraction & Model Building**
   - Converting text data into numerical features using techniques like Bag-of-Words (CountVectorizer) or TF-IDF (if used for ML models).
   - Splitting the dataset into training and testing sets (if using ML models for comparison).
   - Training machine learning models such as Naive Bayes, Logistic Regression, or SVM for classification (optional, for benchmarking).

5. **Evaluation**
   - Evaluating the performance of Vader and TextBlob using metrics like accuracy, precision, recall, and F1-score.
   - Visualizing confusion matrices and comparative plots.

6. **Results & Interpretation**
   - Reporting the comparative performance of Vader and TextBlob.
   - Discussing misclassifications, strengths, and limitations of each tool.
   - Suggestions for improvement and further research.

## Results
- Both Vader and TextBlob were able to classify movie reviews with high accuracy, but their performance varied on certain types of reviews (e.g., sarcasm, negations).
- Vader generally performed better on social-media-like text and short reviews, while TextBlob provided more nuanced polarity scores for longer texts.
- The confusion matrices and classification reports for both tools are included in the notebook for detailed comparison.
- Some misclassifications occurred due to ambiguous or sarcastic reviews, which is a common challenge in sentiment analysis.

## Conclusions
- Comparative sentiment analysis using Vader and TextBlob provides valuable insights into the strengths and weaknesses of each tool.
- Vader is well-suited for social media and short texts, while TextBlob offers more detailed polarity analysis for longer reviews.
- Proper preprocessing and tool selection are crucial for optimal performance.
- The approach can be extended to other domains such as product reviews, social media analysis, and customer feedback.

## Use Cases
- **Movie Review Aggregators**: Automatically classify and summarize user reviews using multiple sentiment analysis tools for better accuracy.
- **Product Review Analysis**: Analyze customer sentiment for e-commerce platforms using comparative approaches.
- **Social Media Monitoring**: Track public sentiment on brands, products, or events with tool-specific strengths.
- **Customer Support**: Prioritize and respond to negative feedback efficiently by leveraging the best-suited sentiment tool.

## Future Work
- Incorporate deep learning models (e.g., LSTM, BERT) for improved accuracy.
- Handle neutral and mixed sentiments.
- Deploy the model as a web service or API for real-time sentiment analysis.
- Explore additional sentiment analysis libraries and ensemble approaches.

## How to Run Project
1. Open `Sentiment Analysis.ipynb` in Jupyter Notebook.
2. Run all cells to execute the analysis pipeline.
3. Review the results and visualizations in the notebook.

---

*This project demonstrates the practical application of NLP and machine learning for real-world text classification problems, and provides a comparative study of two leading sentiment analysis tools. For more details, refer to the notebook and code files.*
