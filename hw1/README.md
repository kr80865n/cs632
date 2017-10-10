# cs632
Deep Learning

Part 1b) Conceptual questions

1. In a Nearest Neighbor classifier, is it important that all features be on the same scale?
Think: what would happen if one feature ranges between 0-1, and another ranges
between 0-1000? If it is important that they are on the same scale, how could you
achieve this?

Answer: Yes it is important that all features should be on the same scale so normalization is necessary,
as K-Means considers Euclidean distance to be meaningful.If a feature has a big scale compared to another,
but the first truly represents greater diversity, then clustering in that dimension should be penalized.
Normalization Formula Z(i)=(X(i)-Min(x))/(Max(x)-Min(x)).

2. What is the difference between a numeric and categorical feature? How might you
represent a categorical feature so your Nearest Neighbor classifier could work with it?

Answer: The difference between numerical and categorical feature is Numerical feature are continious in nature 
and are always is between 0.0 till 1.0 when it is normalized. Categorical feature are not continous in nature they
are either 0 or 1.AS categorical features are always 0's or 1's so if the dataset's contain 4 features which is used to 
train the model so when it is normalized the feautres are represented with 0's and 1's.

Example: If the model has a feature named Education in which the values are like Undergraduate, Graduate, PHD, Dropout so the dataset would look like as below:

Sr. No.    Undergraduate      Graduate       PHD       Dropout

1)                0               0         1       0

2)                1               0         0       1

3)                0                1        0       0

From above example it states that one row is one observation where acutaul dataset is having just a column named Education but when it 
is normalized it gets converted as above example shown.

3. What is the importance of testing data?

Answer: Importance of testing data is to test the model which is been trained on training data where actual labels are compared with
predicted value. Testing data is similar to the training data so we can get exact idea or accuracy how does the model works.

4. If you were to include additional features for the Iris dataset, what would they be, and
why?

Answer:Color feature would help predict the label with more higher accuracy.



Part2b) Conceptual  questions


1. What are the strengths and weaknesses of using a Bag of Words? (Tip: would this
representation let you distinguish between two sentences the same words, but in a
different order?)

Answer:
Bag of words ignores the context of words. This can simplify the problem at some cost.
It can fail badly depending on specific case.
My specific example:
Toy Dog != Dog Toy
The first phrase is a type of breed (small), the second phrase is an object of play for a canine of any size.

2. Which of your features do you think is most predictive, least predictive, and why?

Answer: BOW feature is most predictive and least predictive is the stop words which is also the part of the body and stop words are the
general word which are included in both spam and non spam emails.


3. Did your classifier misclassify any examples? Why or why not?

Answer: Yes it did miss classify one of the example. mail body starts with the owrd spam.
