#Распознование рукописных цифр
from sklearn import datasets

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

