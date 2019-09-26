# Classification metrics optimization

#### library that support Logloss optimization

* Tree-based

  > `XGBoost`,` LightGBM`
  >
  > ~~`sklearn, RandomForetClassifier`~~

* Linear model

  > `sklearn.<>Regression`
  >
  > `sklearnSGDRegressor`
  >
  > `Vowpal Wabbit`

* Neural nets

  > `PyTorch`, `Keras`, `TF`, etc

<br/>

#### Correct probabilities

* Take all objects with score e.g. ~0.8
  * 80% of them of class 1
  * 20% of them class 0

Incorrect probabilities:

* Take all objects with score e.g. ~0.8
  * 50% of them of class 1
  * 50% of them of class 0

<br/>

#### library that support AUC optimization

- Tree-based

  > `XGBoost`,` LightGBM`
  >
  > ~~`sklearn, RandomForetClassifier`~~

- Linear model

  > ~~`sklearn.<>Regression`~~
  >
  > ~~`sklearnSGDRegressor`~~
  >
  > ~~`Vowpal Wabbit`~~

- Neural nets

  > `PyTorch`, `Keras`, `TF` -- not out of the box

<br/>

#### Pairwise Loss

There exists an algorithm to optimize AUC with gradient-based methods: **Pairwise loss**

![pairwise loss formula 1](Images/Classification-matric-review/pairwise-loss-1.svg)

<img src="Images/Classification-matric-review/pairwise-loss-1.png" alt="pairwise-loss-1" width="35%;" />



![pairwise loss formula 2](Images/Classification-matric-review/pairwise-loss-2.svg)

<img src="Images/Classification-matric-review/pairwise-loss-2.png" alt="pairwise-loss-2" width="35%;" />

<br/>