import numpy as np
import matplotlib.pyplot as plt
import csv
import pickle

#%% Функция для загрузки данных из .txt
def load_data_from_txt(filename, delimiter=';', skiprows=15):
    try:
        data = np.loadtxt(filename, delimiter=delimiter, skiprows=skiprows)
    except ValueError as e:
        print(f"Error reading the file: {e}")
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            data = []
            for line in lines[skiprows:]:
                if not line.startswith(';') and line.strip():
                    try:
                        data.append([float(x.replace(',', '.')) for x in line.strip().split(delimiter)])
                    except ValueError as ve:
                        print(f"Error converting line: {line.strip()} - {ve}")
        data = np.array(data)
    return data

#%% Функция для загрузки данных классификации (сейчас не используется файл, а просто передаем вручную)
def load_classification_data(file_path=None):
    true_states = [
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    predicted_states = []  
    return true_states, predicted_states


#%% Функция для поиска пиков (максимумов и минимумов)
def find_peaks(data, threshold=25, interval_length=1500):
    maxima, minima = [], []  # Для хранения индексов пиков

    for start in range(0, len(data), interval_length):
        end = start + interval_length
        segment = data[start:end]
        
        # Находим максимальное значение в отрезке, если оно превышает порог
        max_value = np.max(segment)
        if max_value > threshold:
            max_index = np.argmax(segment) + start  # Индекс максимума относительно всего массива
            maxima.append(max_index)

        # Находим минимальное значение в отрезке (для отрицательных пиков)
        min_value = np.min(segment)
        if abs(min_value) > threshold:
            min_index = np.argmin(segment) + start  # Индекс минимума относительно всего массива
            minima.append(min_index)

    return maxima, minima

#%% Функция для записи разницы амплитуд в CSV файл
def save_amplitude_differences_to_csv(maxima, minima, data, maxima_time, minima_time, true_states, start_time, output_filename='amplitude_differences.csv'):
    with open(output_filename, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['Amplitude (mV)', 'Time (s)', 'True State'])  # Заголовки столбцов

        for i in range(len(maxima)):
            max_time = maxima_time[i]

            # Учитываем start_time и преобразуем время пика в индекс true_states
            true_state_index = int(max_time - start_time)  # Каждое состояние длится 1 секунду
            if true_state_index < 0 or true_state_index >= len(true_states):
                print(f"Warning: Peak at time {max_time} is outside the range of true_states.")
                continue

            true_state = true_states[true_state_index]

            # Поиск ближайшего минимума
            min_times_in_interval = minima_time[(minima_time >= max_time - 0.15) & (minima_time <= max_time + 0.15)]
            if len(min_times_in_interval) == 1:
                min_index = np.argmin(abs(minima_time - max_time))
                min_time = minima_time[min_index]
                min_value_at_peak = data[minima[min_index]]

                # Разница по модулю между пиками
                amplitude_diff = abs(data[maxima[i]] - min_value_at_peak)

                # Записываем разницу, время и состояние в CSV
                writer.writerow([amplitude_diff, max_time, true_state])

#%% Функция для создания графиков
def plot_data(time_in_seconds, data, maxima_time, minima_time, maxima, minima, true_states, start_time):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Отображаем максимумы и минимумы на графике
    ax1.plot(maxima_time, data[maxima], "rx", label="Maximim")
    ax1.plot(minima_time, data[minima], "bx", label="Minimum")
    
    # График EMG сигнала
    ax1.plot(time_in_seconds, data, label='EMG Signal', color='tab:blue')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('EMG Amplitude (mV)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Добавление второй оси для состояний с учетом смещения по времени
    adjusted_time = np.arange(start_time, start_time + len(true_states))
    ax2 = ax1.twinx()
    ax2.plot(adjusted_time, true_states, label='True States', linestyle="--", marker="o", color='tab:red')
    ax2.set_ylabel('States', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Настройка легенды
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Настройка заголовка и сетки
    plt.title("EMG Signal and True States Visualization")
    plt.grid(True)

    # Показываем график
    plt.tight_layout()
    plt.show()

#%% Основная логика
def main():
    # Параметры
    filename = 'D:/Учеба/Диплом/Данные/Tyunkin Experiments/курсовая/ЭМГ без выбросов/Kopylova Olga.txt'
    start_time = 301
    # Время для сдвига
    delimiter = ';'
    skiprows = 15
    sampling_rate = 500  # Частота дискретизации
    
    # Загрузка данных
    data = load_data_from_txt(filename, delimiter, skiprows)
    true_states, predicted_states = load_classification_data()  
    
    # Время для оси X
    time_in_seconds = np.arange(0, len(data)) / sampling_rate

    # Поиск пиков
    maxima, minima = find_peaks(data)

    # Преобразуем индексы пиков в время (секунды)
    maxima_time = np.array(maxima) / sampling_rate
    minima_time = np.array(minima) / sampling_rate

    # Сохранение разниц амплитуд в CSV
    save_amplitude_differences_to_csv(maxima, minima, data, maxima_time, minima_time, true_states, start_time)

    # Построение графика
    plot_data(time_in_seconds, data, maxima_time, minima_time, maxima, minima, true_states, start_time)

if __name__ == "__main__":
    main()
