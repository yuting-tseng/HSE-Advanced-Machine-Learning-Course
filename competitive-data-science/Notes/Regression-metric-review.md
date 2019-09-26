# Regression metrics review
Compare the constants

* MSE: lot bias
* MAE: much less biased
* MSPE, MAPE
	* Weighted version of MSE/MAE
	* Biased towards smaller targets because they assign higher weight to the object with small targets.
* (R)MSLE
	* MSE in log space
	* frequently considered as better metrics than MAPE, since it is less biased towards small targets, yet works with relative errors.

## MSE & MAE 


![MSE formula](Images/Regression-metric-review/MSE.svg)

![MAE formula](Images/Regression-metric-review/MAE.svg)

![RMSE formula](Images/Regression-metric-review/RMSE1.svg)

, where <img src="https://latex.codecogs.com/svg.latex?\Large&space;N" title="img1" />=number of objects, 
![where y](Images/Regression-metric-review/MSE2.svg)= target for i-th object, 
![where y hat](Images/Regression-metric-review/MSE3.svg)=predictions for i-th object

<br/>

If our target matrix is RMSE, we can still optimize our model with MSE , MSE is a little bit easy to work with, so everybody use MSE instead of RMSE (gradient of RMSE = constant * gradient of MSE)


![RMSE formula](Images/Regression-metric-review/RMSE2.svg)

![RMSE formula](Images/Regression-metric-review/RMSE3.svg)

<br/>

#### R-squared
<p>
<img align="middle" src="https://latex.codecogs.com/svg.latex?\Large&space;R^2=1-\frac{\frac{1}{N}\sum_{i=1}^{N}{(y_{i}-\hat{y_{i}})}^2}{\frac{1}{N}\sum_{i=1}^{N}{(y_{i}-\bar{y_{i}})}^2}=1-\frac{MSE}{\frac{1}{N}\sum_{i=1}^{N}{(y_{i}-\bar{y_{i}})}^2}" title="R2" /> , where <img align="middle" src="https://latex.codecogs.com/svg.latex?\Large&space;\bar{y}=\frac{1}{N}\sum^{N}_{i=1}y_{i}" title="R2-1" />
</p>

<p>
We use R-squared matrix to measure our model is better than a constant baseline. By optimize <img src="https://latex.codecogs.com/svg.latex?\Large&space;R^2" title="R2-2" />, we could also optimize MSE (equavilant). The <img src="https://latex.codecogs.com/svg.latex?\Large&space;R^2" title="R2-2" /> between [0, 1], <b>0</b> means not better than baseline, <b>1</b> means better than baseline.
</p>


* **MSE (Mean Square Error), RMSE, R-suqared**
  
  * Best constant: target mean
  * they are the same from optimization perspective
  
* **MAE (Mean Absolute Error)**
  
  * Best constant: target medium
  * MAE is more robust than MSE (it is not that influence by outliers, doesn’t mean it’s always better use MAE then MSE)

<br/>

## MSPE & MAPE 

![MSPE formula](Images/Regression-metric-review/MSPE.svg)

![MAPE formula](Images/Regression-metric-review/MAPE.svg)



* MSPE (Mean Squared Percentage Error)
  * Best constant: weighted target mean
* MAPE (Mean Absolute Percentage Error)
  * Best constant: weighted target medium

<br/>

## RMSLE

![RMSLE](Images/Regression-metric-review/rmsle.svg)


RMSLE just RMSE calculated in <img src="https://latex.codecogs.com/svg.latex?\Large&space;log" title="img-1" /> scale. The target are usually not negative, but can equal to 0 (<img src="https://latex.codecogs.com/svg.latex?\Large&space;log0" title="img-1" /> is undefined). A constant is always add to target, there we choose 1.

* Best constant in log space is a mean target value
* We need to exponentiate it to get an answer

