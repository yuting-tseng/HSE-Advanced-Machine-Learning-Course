# Week 1 Note

## Competition Mechanics

#### Platforms: 
* Kaggle
* DrivenData
* CrowdAnalityx
* CodaLab
* DataScienceChallange.net
* Single-competition site (like KDD, VizDooM)

####  Conclusion
* There is no "silver bullet" algorithm
* Linear models split space into 2 subspaces
* Tree-based methods split space into boxes
* k-NN methods heavy rely on how to measure points "closeness"
* Feed-forward NNs produce smooth non-linear decision boundry

The most powerful methods are <b>Gradient Boost Decision Trees</b> and <b>Neural Networks</b>. 
<br/>
But you souldn't underestimate the others.

#### Overview of methods
* [Scikit-Learn (or sklearn) library](http://scikit-learn.org/)
* [Overview of k-NN](http://scikit-learn.org/stable/modules/neighbors.html) (sklearn's documentation)
* [Overview of Linear Models](http://scikit-learn.org/stable/modules/linear_model.html) (sklearn's documentation)
* [Overview of Decision Trees](http://scikit-learn.org/stable/modules/tree.html) (sklearn's documentation)
* Overview of algorithms and parameters in [H2O documentation](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science.html)

#### Additional Tools
* [Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit) repository
* [XGBoost](https://github.com/dmlc/xgboost) repository
* [LightGBM](https://github.com/Microsoft/LightGBM) repository
* [Interactive demo](http://playground.tensorflow.org/) of simple feed-forward Neural Net
* Frameworks for Neural Nets: [Keras](https://keras.io/),[PyTorch](http://pytorch.org/),[TensorFlow](https://www.tensorflow.org/),[MXNet](http://mxnet.io/), [Lasagne](http://lasagne.readthedocs.io/)
* [Example from sklearn with different decision surfaces](http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)
* [Arbitrary order factorization machines](https://github.com/geffy/tffm)

<br/>

## Software/Hardware requirements

#### StandCloud Computing:
* [AWS](https://aws.amazon.com/), [Google Cloud](https://cloud.google.com/), [Microsoft Azure](https://azure.microsoft.com/)

#### AWS spot option:
* [Overview of Spot mechanism](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-spot-instances.html)
* [Spot Setup Guide](http://www.datasciencebowl.com/aws_guide/)

#### Stack and packages:
* [Basic SciPy stack (ipython, numpy, pandas, matplotlib)](https://www.scipy.org/)
* [Jupyter Notebook](http://jupyter.org/)
* [Stand-alone python tSNE package](https://github.com/danielfrg/tsne)
* Libraries to work with sparse CTR-like data: [LibFM](http://www.libfm.org/), [LibFFM](https://www.csie.ntu.edu.tw/~cjlin/libffm/)
* Another tree-based method: RGF ([implemetation](https://github.com/baidu/fast_rgf), [paper](https://arxiv.org/pdf/1109.0887.pdf))
* Python distribution with all-included packages: [Anaconda](https://www.continuum.io/what-is-anaconda)
* [Blog "datas-frame" (contains posts about effective Pandas usage)](https://tomaugspurger.github.io/)

<br/>

## Feature preprocessing and generation with respect to models

#### Numeric features
* Scaling and Rank for numeric features:
 	* Tree-based models dowsn't depend on them
	* Non-tree-based models hugely depend on them
* Most often used preprocessings are:
	* MinMaxScaler: to [0,1]
	* StandardScaler: to mean==0, std==1
	* Rank: sets spaces between sorted values to be equal
	* `np.log(1+x)` and `np.sqrt(1+x)`
* Feature genertion is powered by:
	* Prior knowledge
	* Exploratory data analysis


#### Categorical and ordinal features
* Values in ordinal features are sorted in some meaningful order
* Label encoding maps categories to numbers
* Frequency encoding maps categories to their frequencies
* Label and Frequency encodings are often used for tree-based models
* One-hot encoding is often used for non-tree-based models
* Interactions of categorical features can help Linear models and KNN


#### Datetime
* Periodicity
* Time since row-independent/row-dependent event
* Difference between dates

#### Coordinates 
* lnteresting places from train/test data or additional data
* Centers of clusters
* Aggregated statistics
	

#### Handling missing values
* The choice method to fill NaN depends on the situation
* Usual way to deal with missing values is to place them with -999, mean or median
* Missing values already can be replaced with something by organizers
* Binary feature "isnull" can be beneficial
* In general, avoid filling nans before feature generation
* Xgboost can handle NaN


#### Additional Materials

Feature preprocessing
* [Preprocessing in Sklearn](http://scikit-learn.org/stable/modules/preprocessing.html)
* [Andrew NG about gradient descent and feature scaling](https://www.coursera.org/learn/machine-learning/lecture/xx3Da/gradient-descent-in-practice-i-feature-scaling)
* [Feature Scaling and the effect of standardization for machine learning algorithms](http://sebastianraschka.com/Articles/2014_about_feature_scaling.html)


Feature generation
* [Discover Feature Engineering, How to Engineer Features and How to Get Good at It](https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/)
* [Discussion of feature engineering on Quora](https://www.quora.com/What-are-some-best-practices-in-Feature-Engineering)

<br/>

## Feature extraction from text and images

#### Feature extraction from text

* Preprocessing 
	* Lowercase, stemming, lemmatization, stopwords
* Bag of words
	* [Feature extraction from text with Sklearn](http://scikit-learn.org/stable/modules/feature_extraction.html)
	* [More examples of using Sklearn](https://andhint.github.io/machine-learning/nlp/Feature-Extraction-From-Text/)
* Word2vec
	* [Tutorial to Word2vec](https://www.tensorflow.org/tutorials/word2vec)
	* [Tutorial to word2vec usage](https://rare-technologies.com/word2vec-tutorial/)
	* [Text Classification With Word2Vec](http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/)
	* [Introduction to Word Embedding Models with Word2Vec](https://taylorwhitten.github.io/blog/word2vec)
* NLP Libraries
	* [NLTK](http://www.nltk.org/)
	* [TextBlob](https://github.com/sloria/TextBlob)

BOW and w2v comparison
* Bag of words
 	* Very large vectors
 	* Meaning of each value in vector is known
 	* N-grams can help to use local context
	* TFiDF can be use as postpreprocessing
* Word2vec
	* Relatively small vectors
	* Values in vector can be interpreted only in some cases
	* The words with similiar meaning often have similiar embeddings 
	* Pretrained model


#### Feature extraction from images

* Features can be extracted from different layers
* Careful choosing of pretrained netword can help
* Finetuning allows to refine pretrained models
* Data augumentation can improve the model


Pretrained models
* [Using pretrained models in Keras](https://keras.io/applications/)
* [Image classification with a pre-trained deep neural network](https://www.kernix.com/blog/image-classification-with-a-pre-trained-deep-neural-network_p11)

Finetuning
* [How to Retrain Inception's Final Layer for New Categories in Tensorflow](https://www.tensorflow.org/tutorials/image_retraining)
* [Fine-tuning Deep Learning Models in Keras](https://flyyufelix.github.io/2016/10/08/fine-tuning-in-keras-part2.html)

<br/>

## Final competition

The competition is hosted on Kaggle Inclass and can be found [here](https://www.kaggle.com/c/competitive-data-science-predict-future-sales). You may also find useful the first version of the competition hosted on InClass/