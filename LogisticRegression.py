import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# ----------------------------------------------Load dataset
TestScore= np.loadtxt('score.txt')
Passed=np.loadtxt('pass.txt')
# step size in the meshgirid
MeshgiridStep = .02


#---------------------------------------------Logistic regression
# Create Logistic regression object
logreg = linear_model.LogisticRegression(C=1e5)

# Train the model using the training sets and target
logreg.fit(TestScore, Passed)

# ------------------------------------------------Plot outputs
# meshgrid data and target to plot
TestScore_min, TestScore_max = TestScore[:, 0].min() - .5, TestScore[:, 0].max() + .5
Passed_min, Passed_max = TestScore[:, 1].min() - .5, TestScore[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(TestScore_min, TestScore_max, MeshgiridStep), np.arange(Passed_min, Passed_max, MeshgiridStep))
# predict class for ech input data
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# Plot the result
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
# Plot the decision boundary. For that, we will assign a color to each
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
# Plot training points
plt.scatter(TestScore[:, 0], TestScore[:, 1], c=Passed, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('First Test Score')
plt.ylabel('Second Test Score')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()
