#Распознование рукописных цифр
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay as cmd
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

digits = datasets.load_digits()
print('Digits.images ' + str(digits.images.shape))
print('digits.target' + str(digits.target.shape))
digits.images[0]

%matplotlib inline
import matplotlib.pyplot as plt
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
plt.imshow(digits.images[0], cmap=plt.cm.gray_r)

fig, axes = plt.subplots(5, 10, figsize=(12, 7), subplot_kw={'xticks':[], 'yticks': []})
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap=plt.cm.gray_r)
    ax.text(0.45, 1.05, str(digits.target[i]), transform=ax.transAxes)

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=0)
model = LogisticRegression(max_iter=5000)
model.fit(x_train, y_train)
model.score(x_test, y_test)


fig, ax = plt.subplots(figsize=(8,8))
ax.grid(False)
cmd.from_estimator(model, x_test, y_test, cmap="Blues", colorbar=False, ax=ax)
