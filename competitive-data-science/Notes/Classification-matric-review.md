# Classification metrics review

* soft labels (soft prediction) are classifier's scores
* Hard labels (hard predictions)
  * ![hard label-1](Images/Classification-matric-review/hard-label-1.svg)
  * ![hard label-2](Images/Classification-matric-review/hard-label-2.svg) (![hard label-3](Images/Classification-matric-review/hard-label-3.svg) = threshold)

<br/>

#### Logarithmic loss (logloss)

* In binary classification <br/>
  ![logloss binary classification formula](Images/Classification-matric-review/logloss-1.svg) 
* In multicalss classification <br/>
  ![logloss multicalss classification formula](Images/Classification-matric-review/logloss-2.svg) , where ![logloss multicalss classification where](Images/Classification-matric-review/logloss-3.svg)
* In practice <br/>
  ![logloss practice classification formula](Images/Classification-matric-review/logloss-4.svg)

The clssifier output probability, and the binary and multiclass classigication formula are different

![N](Images/Classification-matric-review/n.svg)=number of objects, ![L](Images/Classification-matric-review/l.svg)=number od classes, ![y](Images/Classification-matric-review/y.svg)=ground truth, ![y-hat](Images/Classification-matric-review/y-hat.svg)=prediction, ![indicator](Images/Classification-matric-review/indicator.svg)=indicator function

<br/>

![logloss formula alpha](Images/Classification-matric-review/logloss-5.svg)

<img src="Images/Classification-matric-review/logloss.png" alt="logloss loss" width="40%" />

* Logloss usually penalizes completely wrong answers and prefers to make a lot of small mistakes to one but severer mistake. 
* Best constant: set ![alpha](Images/Classification-matric-review/alpha.svg) to frequency of the ![i](Images/Classification-matric-review/i.svg)-th class 
  * Example dataset: 10 cats, 90 dogs ![logloss example](Images/Classification-matric-review/logloss-example.svg)

<br/>

#### Area Under Curve (AUC ROC)

<img src="Images/Classification-matric-review/auc.png" alt="auc plot" width="40%" />

We usually take soft predictions from our model and apply threshold. This metric kind of tries all possible ones and aggregates those scores. We find the maximum value of AUC, and don’t need to define the threshold.

<br/>

![auc formul](Images/Classification-matric-review/auc-1.svg)

* Only for binary tasks
* Depends only on ordering of the of the predictions, not on absolute values
* Several explanations
  * Area under curve
  * Pairs ordering
* Best constant:
  * All constants give same scores
* Random predictions lead to AUC=0.5

<br/>

#### Cohen's Kappa

In Cohen’s Kappa we take another value as baseline. We take the higher predictions for the dataset and shuffle them, like randomly permute. And then we calculate an accuracy for these shuffled predictions.
and that we be our baseline.

![Cohen's Kappa formul](Images/Classification-matric-review/kappa-1.svg)  , where ![Cohen's Kappa where](Images/Classification-matric-review/kappa-2.svg)

![pe](Images/Classification-matric-review/pe.svg)=what accuracy would be on average, if we randomly permute our predictions

<br/>

![Cohen's Kappa formul](Images/Classification-matric-review/kappa-3.svg)

Example 

* dataset: 10 cats and 90 dogs
* Predict 20 cats and 80 dogs at random
  * ![Cohen's Kappa accuracy](Images/Classification-matric-review/kappa-4.svg)
  * ![Cohen's Kappa error](Images/Classification-matric-review/kappa-5.svg)

