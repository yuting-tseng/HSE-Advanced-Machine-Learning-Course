# Week 2 Note

## Exploratory data analysis (EDA)

With EDA we can:
* Better understand the data
* Build an intuition about the data
* Generate hypothesizes
* Find insight (<I>magic features</I>)

Do EDA first! Do not immediately dig into modelling.

#### Building intuition about the data
* Get domain knowledge
	* It helps to deeper understand the problem
* Check if data is intuitive
	* And agrees with domain knowledge
* Understand how the data was generated
	* It is crucial to understand the generation process to set up a proper validation scheme

#### Exploring anonymized data/features
* Try to decode the fetures
	* Guess the true meaning of the feature
* Guess the feature types
	* Each type needs its own preprocessing

### Visualization

Explore individual features

``` python
plt.hist(x) # Histograms
plt.plot(x, '.') # Plot (index vs value)

df.describe() # Statistics
x.mean()
x.var()

x.value_counts() # Other tools
x.isnull()
```

Explore feature relations
* Pairs
	* Scatter plot, scatter matrix
	* Corrplot
* Groups
	* Corrplot + clustering
	* Plot (index vs feature statistics)

``` python
plt.scatter(x1, x2)
pd.scatter_matrxix(df)
df.corr(), plt.matshow(...)
df.mean.sort_values().plot(style='.')
```

Visualization tools
* [Seaborn](https://seaborn.pydata.org/)
* [Plotly](https://plot.ly/python/)
* [Bokeh](https://github.com/bokeh/bokeh)
* [ggplot](http://ggplot.yhathq.com/)
* [Graph visualization with NetworkX](https://networkx.github.io/)

Others
* [Biclustering algorithms for sorting corrplots](http://scikit-learn.org/stable/auto_examples/bicluster/plot_spectral_biclustering.html)

<br/>

## Validation

#### Validation and overfitting
<img src="images/image1.png" width="30%">

* Validation helps us evaluate a quality of the model
* Validation helps us select the model which will perform best on the unseen data
* Underfitting refers to not capturing enough patterns in the data
* Generally, overfitting regers to 
	* capturing noize
	* capturing patterns which do not generalize to test data
* In competitions, overfitting refers to 
	* low model's quality on test data, which was unexpected due to validation scores


#### Validation strategies

* Hodout
	* `sklearn.model_selection.ShuffleSplit`
	* ngroups = 1
* K-fold
	* `sklearn.model_selection.Kfold`
	* ngroups = k
* Leave-one-out
	* `sklearn.model_selection.LeaveOneOut`
	* ngroups = training-sample-length

**Stractification** preserve the same target distribution over different folds, which is useful for:
* Small datasets
* Unbalanced datasets
* Multiclass classification

Notice, that these are validation schemes are supposed to be used to estimate quality of the model. <br/>
When you found the **right hyper-parameters** and want to get test predictions don't forget to **retrain your model using all training data**.


#### Data splitting strategies

* In most cases data is split by
	* Random, Row numbrt (rowwise)
	* Timewise
	* By Id
* Logic of feature generation depends on the dta splitting strategy
* Set up your validation to mimic the train/test split od the competition


#### Problems occurring during validation

* If we have big dispersion of scores on validation stage, we should do extensive validation
	* Average scores from different KFold splits
	* Tune model on one split, evaluate score on the other
* If submission's score do not match local validation score, we should 
	* Check if we have too little data in public LB
	* Check if we overfitted
	* Check if we chose correct splitting strategy
	* Check if train/test have different distributions
* Expect LB shuffle because of 
	* Randomness
	* Little amount of data 
	* Different public/private distributions

Materials
* [Validation in Sklearn](http://scikit-learn.org/stable/modules/cross_validation.html)
* [Advices on validation in a competition](http://www.chioka.in/how-to-select-your-final-models-in-a-kaggle-competitio/)
	
<br/>

## Data Leakages

Materials
* [Perfect score script by Oleg Trott](https://www.kaggle.com/olegtrott/the-perfect-score-script) -- used to probe leaderboard
* [Page about data leakages on Kaggle](https://www.kaggle.com/wiki/Leakage)














