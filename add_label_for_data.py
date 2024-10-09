#Создать метки для категорированных данных
from sklearn.preprocessing import LabelEncoder
import pandas as pd

data = [[10, 'red'], [20, 'blue'], [12, 'red'], [16, 'green'], [22, 'blue']]

df = pd.DataFrame(data, columns=['length', 'Color'])

encoder = LabelEncoder()
df['Color'] = encoder.fit_transform(df['Color'])

#Горячие кодирование
df = pd.get_dummies(df, columns=['Color'])
df.head()
