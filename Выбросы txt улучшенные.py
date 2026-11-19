import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# Загрузка данных из файла .txt
filename = ''
file_basename = os.path.splitext(os.path.basename(filename))[0]
output_dir = ''


# Параметры для обработки данных
delimiter = ';'  # Разделитель в файле
skiprows = 20  # Пропускаем 20 строк, где содержатся метаданные
threshold = 25  # Порог амплитуды
interval_length = 600  # Размер отрезка в сэмплах (300 мс при 2000 Гц)
sampling_rate = 500  # Частота дискретизации в Гц
z_score_threshold = 3  # Порог для z-оценки (выбросы)

try:
    # Загрузка данных с пропуском метаданных
    data = np.loadtxt(filename, delimiter=delimiter, skiprows=skiprows)
except ValueError as e:
    print(f"Error reading the file: {e}")
    
    # Альтернативная загрузка данных с обработкой запятых
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        data = []
        for line in lines[skiprows:]:
            if not line.startswith(';') and line.strip():  # Игнорируем комментарии и пустые строки
                try:
                    # Заменяем запятые на точки и конвертируем строку в массив чисел
                    data.append([float(x.replace(',', '.')) for x in line.strip().split(delimiter)])
                except ValueError as ve:
                    print(f"Error converting line: {line.strip()} - {ve}")
    data = np.array(data).flatten()

# Время для оси X (в секундах)
time_in_seconds = np.arange(0, len(data)) / sampling_rate

# Фильтруем данные по указанному условию
fdata = data[(data > threshold) | (data < -threshold)]

# Удаление выбросов с использованием z-оценки
if len(fdata) > 0:
    mean = np.mean(fdata)
    std = np.std(fdata)
    z_scores = np.abs((fdata - mean) / std)

    # Находим индексы выбросов в исходном массиве данных
    above_threshold_indices = np.where((data > threshold) | (data < -threshold))[0]
    outlier_indices = above_threshold_indices[z_scores > z_score_threshold]

    # Создание массива данных без выбросов
    non_outlier_mask = np.ones(len(data), dtype=bool)  # Маска для всех данных
    non_outlier_mask[outlier_indices] = False  # Исключаем выбросы
    filtered_data = data[non_outlier_mask]
else:
    # Если нет данных выше threshold, используем исходные данные
    filtered_data = data

# Добавление обработки выбросов с учетом их длительности
ms_to_samples = int(75 * sampling_rate / 1000)  # Переводим n мс в количество сэмплов
data_cleaned = data.copy()  # Создаем копию данных для модификации

# Обработка выбросов: замена значений в интервале ± n мс на 0
for idx in outlier_indices:
    # Определяем границы интервала вокруг выброса
    start_idx = max(0, idx - ms_to_samples)  # Левая граница (не меньше 0)
    end_idx = min(len(data_cleaned) - 1, idx + ms_to_samples)  # Правая граница (не больше длины массива)
    
    # Заменяем значения в интервале на 0
    data_cleaned[start_idx:end_idx + 1] = 0

# Сохранение данных без выбросов в новый файл
filtered_filename = os.path.join(output_dir, f'{file_basename}.txt')
np.savetxt(filtered_filename, data_cleaned, delimiter='\n', fmt='%.3f')

# Вывод информации о фильтрации
print(f"Выбросы: {len(data) - len(filtered_data)}")

# Визуализация данных
plt.figure(figsize=(12, 6))

# Исходные данные
plt.subplot(2, 1, 1)
plt.plot(time_in_seconds, data, label="Исходные данные", color="gray", alpha=0.7)
plt.scatter(outlier_indices / sampling_rate, data[outlier_indices], color="red", label="Выбросы", zorder=5)
plt.title("Исходные данные с выбросами")
plt.xlabel("Время (с)")
plt.ylabel("Амплитуда")
plt.legend()
plt.grid(True)

# Очищенные данные
plt.subplot(2, 1, 2)
plt.plot(time_in_seconds, data_cleaned, label="Очищенные данные", color="blue", alpha=0.7)
plt.title("Очищенные данные (выбросы обнулены)")
plt.xlabel("Время (с)")
plt.ylabel("Амплитуда")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
