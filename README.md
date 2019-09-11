# NLP-for-Wine-Varirty-Prediction (The project was a group work of 5 members)
The repository contains implementation of NLP for wine variety prediction using reviews on a big data platform
1.	Abstract
This project utilizes Natural Language processing in the Big Data Environment to build a model that consumes user wine reviews from the ‘Kaggle Wine Dataset’ to predict a categorical variable, wine variety. Various supervised learning models were implemented and tuned for the same, namely, Logistic Regression, Decision Tress, Random Forest and Naïve Bayes. Additionally, the overall performance of the system was enhanced by creating an ensemble which sits on top of these trained models and selects the result from the model which has the highest confidence for every prediction. To discriminate between the models weighted F-score, Accuracy, class wise AUC was used as the evaluation metrics.  

2.	Introduction & Literature Review
Natural Language Processing (NLP) is a branch of artificial Intelligence that helps computers understand, interpret and manipulate human language [1]. The need for NLP rose from the high abundance of unstructured data that is generated every day such as those on social media. NLP entails the process of providing this data a structure suitable to be processed by other intelligent tools and models. It forms the backbone for all the customer service chatbots, speech recognition and even our virtual assistants such as Siri and Alexa. 

The classical NLP approach [figure. 1] is to first tokenize i.e. break the sentence into each word, followed by the removal of unnecessary words (stopwords). The words are then tagged into groups of relevant words called bigrams and unigrams. Then features are extracted for every record in the dataset for a set of bigrams/unigrams selected on the base of an importance parameter such as the Term Frequency and Inverse Frequency Document.     

 
Figure 1: Classical NLP Approach[5]

The inherent capacity of NLP to be applied in every sector for a varied number of use cases has led to its extensive research in academia as well as in the Industry. [2], [3] utilizes a recurrent neural network and Support Vector Machine with RBF kernel respectively to predict wine variety using the user reviews. The models described in both the resources are computationally more expensive than standard machine learning models. On the other hand, [4] uses Decision Tress to make predictions. However, the accuracy is achieved by the author is subpar in comparison to the first two. 
Given the huge amount of data that is generally available our team experimented with applying NLP to the same Dataset on a Big Data Environment (Spark-Scala) and used standard supervised machine learning models. We were able to achieve a decent accuracy of 74% given the resource constraints.  This would allow us to grow our dataset using web scrapped reviews from the web, twitter, Facebook and many more to improve our dataset. Our model would be able to leverage the growing data because of the availability of parallel processing and storage in a Big Data Environment
 
3.	Business Problem
The aim of our project is to construct and compare machine learning models for wine variety prediction using user reviews. By precisely predicting the wine variety and the sentiment related to it we can forecast the demand for the specific wine variety in the future. This would help wine businesses to take a position and address their marketing campaigns as well as determine the stock required. 

4.	Data set
The Dataset used for the project is the ‘Kaggle Wine Dataset’ which contained two separate files each having 130k and 150k records. In both the files every record contains a review of a wine by a certain taster along with the geographic location of the winery, taster’s information and wine price and points. For reducing the computational complexity only, the file with 130k records was chosen to be processed. The distribution of number of unique wine varieties can be seen in Figure 2. As expected Australia, North America and European regions such as Italy have the most number of varieties.  

The points assigned to a certain wine were directly correlated to its price. As can be seen from Figure 3, higher the points of a wine more is the probability of its price being higher. Moreover, every wine variety was exclusive to a point i.e. a specific variety had not been assigned two different points.
		





5.	Target Variable 
This is a supervised classification machine learning approach where the target variable (Wine Variety) is categorical. The dataset of 130 k records had 702 wine varieties. However as seen in Figure 4(a) the dataset is highly imbalanced. the number of records for majority of the varieties is very less to make any subjective prediction. Moreover, the business use case aims at making prediction for trending wine varieties. Therefore, the team took the records for the top 10 varieties only, as shown in Figure 4(b).  


6.	Methodology
Initially, the data was pre-processed to bring it in the required format and remove any null values. Following the cleaning feature engineering and different models were implemented. The performance of all the models was improved by hyper parameter tuning in cross validation. Finally, the best performing model was selected based on Accuracy. 

a.	Data preprocessing
The unnecessary columns such as taster twitter handle, title of the wine were dropped. This was done because we already had the taster names. There were no cases where the taster name was empty but his handle was present. Therefore, his twitter handle was repeated information. We dropped the wines title because if that is known there would be no requirement of predicting its variety. This was followed by removing special characters, punctuations and html tags because these characters do often occur in the dataset but do not essentially play a role in deciphering the language. Lastly, the nulls in the price and taster columns were replaced with modal values of group based on other correlating columns.

b.	Feature Engineering 
The resultant columns except price and points (because they had a high numerical correlation with the wine variety) were concatenated for every record to give a description column. The description column was then tokenized to obtain each word of the description as an element of an array. Each word was then reduced to its root word i.e. the tense and other grammar related syntaxes were removed from the words.  A corpus of few words popularly known as stop words in NLP were removed in every record. These words do not essentially play an important role in the meaning of a sentence. 
Following this a set of 500 features was selected ordered on their importance determined by Term Frequency – Inverse Document Frequency. TFIDF is better than Count Vectorizer in NLP as high frequent words might actually not play a role in distinguishing the categories. 

Following this PCA was implemented for dimensionality reduction to reduce the feature space to 50.  Then, price and points were aggregated into this feature space to give the final feature vectors.   

c.	Model Implementation and Cross Validation

Following the feature engineering different models namely, Decision Trees, OVR (Logistic Regression), Random Forest and Naïve Bayes were implemented. All the models were hyper tuned with cross validation with 3 cross folds. These models were selected because these were the only ones supported in scala (except Multi Layer Perceptron) for a multiclass classification problem.  Multi Layer perceptron was left out due to its computational complexity. 

Additionally, an ensemble was created that sat on top of the hypertuned models and selects the prediction of the model which is the most confident about its prediction. This is done in the hope that a model might be able to learn more about a specific variety in comparison to other. In such a situation the ensemble would be able to make the best of the models.   
d.	Model Comparison: 

Evaluation Metric: Since the selected data is quite balance and the use case does not warrant preferential behaviour to a certain category therefore Accuracy was chosen as the criteria to Judge. However, the team also looked at class wise AUC, weighted F-score, Recall, Precision for judging the models. For keeping it simple only accuracy has been listed here. 

All the 4 models were trained with 50 features selected from PCA. As can be seen from Table 1. Naïve Bayes performed very poorly and Random Forest and OVR performed the best and almost the same. However, the computational complexity for RF was huge in comparison to the others therefore it was not evaluated with 200 features selected from PCA. Similarly, due to very low performance of Naïve Bayes it was not taken forward.  

When compared with 200 features OVR outperformed Decision Trees and it also showed very less overfitting as training and testing error were almost the same. Whereas for Decision trees the testing and training error had a considerable difference. 
 
Table 1. Training and Testing Error for the 4 models
As OVR did not show any signs of overfitting the team decided to further increase the feature space for OVR. Therefore, it was implemented with 300 and 400 features selected from PCA. As can be seen in Table 2 the model accuracy improved with 300 features however with 400 it started overfitting and did not show an improvement in in comparison to the one with 300. 

 
Table 2. Training and Testing error for OVR with PCA 300 and 400
The ensemble created was made to sit on top of the best performing models of the4 architectures. The overall accuracy of the system increased by 0.4%. 
e.	Model Selection - Results: 

The fact that the accuracy of the Ensemble and OVR was approximately the same was indicative of that OVR was the most confident in making predictions for most of the cases. Therefore, the team deemed the increase in accuracy to be insignificant in comparison to the computational complexity associated with training all he four models.  

Therefore, OVR trained with a feature space of 300 features was selected as the most optimal solution. The model resulted in a test Accuracy of 73.6%, Precision of 73.7%, Recall of 73.6% and weighted F1-Score of 73.1%. The difference in training and testing accuracy was also very less therefore the model was not over fitting (Figure 7). Figure 8 illustrates how the model performs (ROC Curve) for every class. The AUC for every class is greater than 0.89 which showcases that the model is very confident in making its predictions.   



Lastly, as can be seen in the Confusion Matrix Heat Map the darkest colours occur on the diagonal.  Therefore, the model can be deemed to perform very well. The two classes where the model seems to make mistakes are Syrah and Bordeaux-style Red Blend. 


7.	Feature Importance
Because of performing dimensionality reduction using PCA the model lost its transparency. The new principle vectors obtained as a result of numerous features could not be used to pinpoint which features actually played an important role in making the decisions. Therefore, if feature importance need to be defined one needs to avoid using dimensionality reduction by such techniques.  

8.	Recommendations
1.	Using Bag of words to capture sentence context in the feature space. 
2.	Multi layer perceptron could be implemented to add more complexity to the system and thus learn more features. 
3.	Avoid using dimensionality reduction as it diminishes the effect of features and one cannot also determine which features play an important role in the model.
4.	If dimensionality reduction is a must one could implement non linear techniques such as TSNE to capture more variance.
5.	Stratified split could be used for testing and training to maintain the homogeneity of the data.

9.	Challenges: 
We faced multiple limitations during the project and some of them are
1.	Lack of libraries in scala for Machine Learning. SVM, Boosted Tress could not be implemented as they were not supported in scala for multi class classification.
2.	Limited computational resources in Databricks Community Edition. Whereas the cluster had a fixed set of specifications which could not be updated to utilize the latest libraries available. Therefore, the team switched to Azure Databricks and utilized the free credits provided as a promotion by Azure. Even with azure, we had very long run times for cross validation. 
3.	Many errors in spark lead to the same exception being thrown. Therefore, it was very hard to debug the code at times as finding support for it from the open source community was difficult.
