import numpy as np
import matplotlib.pyplot as plt
import csv

# Функция для разрезания данных с возможностью задавать пропорции

def crop(data, sampling_rate, segment_ratios):
    total_length = len(data)
    segment_lengths = [int(total_length * ratio) for ratio in segment_ratios]
    segments = []
    start = 0
    for length in segment_lengths:
        segments.append(data[start:start + length])
        start += length
    return segments

# Загрузка данных из файла
filename = 'D:/Учеба/Диплом/Данные/Tyunkin Experiments/курсовая/Gorelov Rostislav.txt'
delimiter = ';'
skiprows = 20  # Пропуск метаданных
sampling_rate = 500  # Частота дискретизации
segment_ratios = [0.32, 0.35, 0.33]  # Пропорции разделения данных

# Читаем данные с учетом возможных ошибок форматирования
try:
    data = np.loadtxt(filename, delimiter=delimiter, skiprows=skiprows)
except ValueError:
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        data = []
        for line in lines[skiprows:]:
            if line.strip():
                try:
                    data.append(float(line.replace(',', '.').strip()))
                except ValueError:
                    print(f"Skipping invalid line: {line.strip()}")
    data = np.array(data)

# Разделяем данные на сегменты с заданными пропорциями
segments = crop(data, sampling_rate, segment_ratios)

# Сохраняем каждый сегмент в отдельный файл
for i, segment in enumerate(segments):
    np.savetxt(f'segment_{i+1}.txt', segment, delimiter=';', fmt='%.5f')

# Визуализация сегментов
fig, axes = plt.subplots(len(segments), 1, figsize=(10, 12))
for i, (segment, ax) in enumerate(zip(segments, axes)):
    time_axis = np.arange(0, len(segment)) / sampling_rate
    ax.plot(time_axis, segment, label=f'Segment {i+1}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
