import numpy as np
import matplotlib.pyplot as plt
import csv

# Загрузка данных из файла .txt
filename = 'D:/Учеба/Диплом/Данные/Tyunkin Experiments/курсовая/Pachomova Darina_13.02.txt'
#filename = 'D:/Учеба/Диплом/Данные/Tyunkin Experiments/курсовая/ЭМГ без выбросов/Pachomova Darina_13.02.txt'

# Параметры для обработки данных
delimiter = ';'
skiprows = 20  # Пропускаем 20 строк, где содержатся метаданные
threshold = 25  # Порог амплитуды 
interval_length = 600  # Размер отрезка в сэмплах (300 мс при 2000 Гц)
sampling_rate = 500  # Частота дискретизации в Гц

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
    data = np.array(data)

# Время для оси X (в секундах)
time_in_seconds = np.arange(0, len(data)) / sampling_rate

# Построение графика ЭМГ сигнала
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(time_in_seconds, data, label='EMG Signal')

# Массивы для хранения индексов максимумов и минимумов
maxima = []
minima = []

for start in range(0, len(data), interval_length):
    end = start + interval_length
    segment = data[start:end]
    
    # Находим максимальное значение в отрезке, если оно превышает порог
    max_value = np.max(segment)
    if max_value > threshold:
        max_index = np.argmax(segment) + start  
        maxima.append(max_index)
    
    # Находим минимальное значение в отрезке (для отрицательных пиков)
    min_value = np.min(segment)
    if abs(min_value) > threshold:
        min_index = np.argmin(segment) + start 
        minima.append(min_index)

# Преобразуем индексы пиков в секунды
maxima_time = np.array(maxima) / sampling_rate
minima_time = np.array(minima) / sampling_rate

# Отображаем максимумы и минимумы на графике
ax.plot(maxima_time, data[maxima], "rx", label="Maxima > Threshold")  # Отображаем только максимумы
ax.plot(minima_time, data[minima], "bx", label="Minima < -Threshold")  # Отображаем только минимумы

# Настройка оси Y
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.set_title('EMG Data Visualization')
ax.legend()
ax.grid(True)

# Вычисление разницы по модулю между пиками и запись в CSV с разделителем ";"
with open('amplitude_differences.csv', mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=';')  # Указываем разделитель ";"
    writer.writerow(['Amplitude (mV)', 'Time (s)'])  # Заголовки столбцов
    
    for i in range(len(maxima)):
        # Проверка наличия соответствующего минимума в интервале ±150 мс
        max_time = maxima_time[i]
        min_times_in_interval = minima_time[(minima_time >= max_time - 0.15) & (minima_time <= max_time + 0.15)]
        
        if len(min_times_in_interval) == 1:
            # Находим амплитуду между пиками
            min_index = np.argmin(abs(minima_time - max_time))
            min_time = minima_time[min_index]
            min_value_at_peak = data[minima[min_index]]
            
            # Разница по модулю между максимумом и минимумом
            amplitude_diff = abs(data[maxima[i]] - min_value_at_peak)
            
            # Записываем разницу и временную метку в CSV
            writer.writerow([amplitude_diff, max_time])

# Показываем график
plt.tight_layout()
plt.show()
