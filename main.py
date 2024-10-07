import numpy as np
from scipy import sparse
import scipy

#Вектор как строка 
vector_row = np.array([1,2,3])
#вектор как столбец
vector_column = np.array([[1], 
                         [2],
                         [3]])
#Матрица 
matrix = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
])

# сжатая разреженная матрица
matrix_sparse = sparse.csr_matrix(matrix)


#Выбор элемента из массивов NumPy
matrix_second = np.array([[1, 2, 3],
                  [14, 5, 6],
                  [7, 8, 9],
                  ])
vector = np.array([1, 2, 3, 4, 5, ])

print(f"Второй элемент вектора -> {vector[1]}")
print(f"3 строка матрицы и 3 элемент -> {matrix_second[2, 2]}")
print(f"Выбрать все элементы вектора -> {vector[:]}")
print(f"Выбрать до 3 элемента вектора -> {vector[:3]}")
print(f"Выбрать все после 3 элемента вектора -> {vector[3:]}")
print(f"Последний элемент вектора -> {vector[-1]}")

#Выбрать элементы из матрицы
print(f"Выбрать первые две строки и все столбы -> {matrix_second[:2, :]}")
print(f"Выбрать все строки и второй столбец -> {matrix_second[:,1: 2]}")

#Взглянуть на кол-во строк и столбцов
print(f"Количество строк и столбцов в матрице -> {matrix_second.shape}")
print(f"Кол-во элементов строки * столбцы -> {matrix_second.size}")
print(f"Кол-во размерностей -> {matrix.ndim}")

#Максимальный и минимальный элемент массива
print(f"Максимальный элемент -> {np.max(matrix_second)}")
print(f"Минимальный элемент -> {np.min(matrix_second)}")
print(f"Максимальный элемент в каждом столбце -> {np.max(matrix_second, axis=0)}")
print(f"Минимальный элемент в каждой строке -> {np.min(matrix_second, axis=1)}")

#Вывод данных 
data_numpy = [vector_row, vector_column, matrix, matrix_sparse]
# for i in data_numpy:
#     print()
#     print(i)


#Применить некоторую функцию ко всем элементам массива
add_sum = lambda i: i + 100
vectorize_summa = np.vectorize(add_sum)
vectorize_summa(matrix_second)