# Final Project Note

The competition is hosted on Kaggle Inclass and can be found [here](https://www.kaggle.com/c/competitive-data-science-predict-future-sales). You may also find useful the first version of the competition hosted on InClass/

### Final Project Advice #1

Competition data is rather challenging, so the sooner you get yourself familiar with it - the better. You can start with submitting sample_submission.csv from "Data" page on Kaggle and try submitting different constants.

### Final Project Advice #2

A good exercise is to reproduce previous_value_benchmark. As the name suggest - in this benchmark for the each shop/item pair our predictions are just monthly sales from the previous month, i.e. October 2015.

The most important step at reproducing this score is correctly aggregating daily data and constructing monthly sales data frame. You need to get [lagged](https://en.wikipedia.org/wiki/Lag_operator) values, fill NaNs with zeros and clip the values into [0,20] range. If you do it correctly, you'll get precisely 1.16777 on the public leaderboard.

Generating features like this is a necessary basis for more complex models. Also, if you decide to fit some model, don't forget to clip the target into [0,20] range, it makes a big difference.

