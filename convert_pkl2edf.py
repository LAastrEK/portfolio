#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import mne

path = "D:/BCIresults/SSVEP/2026-03-02_17-36-36/ssvep_data.pkl"

def convert_pkl_to_edf(pkl_path, output_dir=None):
    print(f"Загрузка: {pkl_path}")
    
    # Загружаем PKL
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print("Содержимое PKL файла:")
    for key in data.keys():
        if isinstance(data[key], np.ndarray):
            print(f"  {key}: {data[key].shape} {data[key].dtype} | min={data[key].min():.2f}, max={data[key].max():.2f}")
        else:
            print(f"  {key}: {type(data[key])}")
    
    # === ИЗВЛЕЧЕНИЕ ДАННЫХ ===
    eeg_data = data['EEG_data'].astype(np.float64)
    srate = float(data['srate'])
    print(f"\nДо обработки: мин={eeg_data.min():.2f}, макс={eeg_data.max():.2f}")

    # Проверка на масштаб и единицы измерения (V or mV)
    if abs(eeg_data.max()) > 10000:
        print("Обнаружены значения >1000. Предполагаем, что данные в микровольтах (μV).")
        eeg_data = eeg_data / 1_000_000
        print(f"После конвертации: мин={eeg_data.min():.6f}, макс={eeg_data.max():.6f}")
    
    # Убираем NaN и бесконечности
    eeg_data = np.nan_to_num(eeg_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # === КАНАЛЫ ===
    if 'channel_names' in data and len(data['channel_names']) == eeg_data.shape[0]:
        ch_names = data['channel_names']
        if isinstance(ch_names[0], bytes):
            ch_names = [ch.decode('utf-8') for ch in ch_names]
        elif isinstance(ch_names[0], np.ndarray):
            # Если это numpy массивы строк
            ch_names = [str(ch) for ch in ch_names]
    else:
        ch_names = [f'EEG {i+1:03d}' for i in range(eeg_data.shape[0])]
    
    print(f"\nКаналы: {len(ch_names)}")
    print(f"Частота: {srate} Hz")
    print(f"Длительность: {eeg_data.shape[1]/srate:.2f} сек")
    
    # === СОЗДАНИЕ MNE RAW ===
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=srate,
        ch_types=['eeg'] * len(ch_names)
    )
    
    # Устанавлием единицы измерения (важно для EDF!)
    # MNE по умолчанию использует Вольты для EEG
    for ch in info['chs']:
        ch['unit'] = mne.io.constants.FIFF.FIFF_UNIT_V  # Вольты
        ch['unit_mul'] = 0  # Без множителя
    
    raw = mne.io.RawArray(eeg_data, info, verbose=False)
    
    # === СОБЫТИЯ / АННОТАЦИИ ===
    if 'events_type' in data and len(data['events_type']) > 0:
        descriptions = []
        onsets = []
        durations = []
        
        marker_ids = data.get('marker_ids', [])
        frequencies = data.get('frequencies', [])
        
        for i in range(len(data['events_type'])):
            m_type = float(data['events_type'][i])
            latency = float(data['events_latency'][i]) - 1  # 1-based to 0-based
            duration = float(data['events_duration'][i])
            
            onset_sec = latency / srate
            duration_sec = duration / srate
            
            if len(marker_ids) > 0 and len(frequencies) > 0:
                try:
                    freq_idx = np.where(np.array(marker_ids) == m_type)[0]
                    if len(freq_idx) > 0:
                        desc = f'Stim/{frequencies[freq_idx[0]]}Hz'
                    else:
                        desc = f'Stim/ID_{int(m_type)}'
                except:
                    desc = f'Stim/ID_{int(m_type)}'
            else:
                desc = f'Stim/ID_{int(m_type)}'
            
            onsets.append(onset_sec)
            durations.append(duration_sec)
            descriptions.append(desc)
        
        annotations = mne.Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions
        )
        raw.set_annotations(annotations)
        print(f"Добавлено {len(descriptions)} аннотаций")
    
    # === СОХРАНЕНИЕ ===
    if output_dir is None:
        output_dir = os.path.dirname(pkl_path)
    base_name = os.path.splitext(os.path.basename(pkl_path))[0]
    edf_path = os.path.join(output_dir, f"{base_name}.edf")
    print(f"\nСохраняем edf: {edf_path}")
    raw.info['meas_date'] = None  # Очищаем дату измерения

    try:
        raw.export(edf_path, overwrite=True, verbose=False)
        print(f"Файл edf успешно сохранен: {edf_path}")
    except Exception as e:
        print(f"Ошибка при сохранении edf: {e}")

    # Используем write_edf вместо export (более надежный метод)
    #mne.write_edf(edf_path, raw, overwrite=True, verbose=False)
    
    # Также сохраняем FIF для Python
    fif_path = os.path.join(output_dir, f"{base_name}_raw.fif")
    raw.save(fif_path, overwrite=True, verbose=False)
    print(f"Сохраняем FIF: {fif_path}")
    
    # === ПРОВЕРКА ===
    print("\n=== ПРОВЕРКА ЗАПИСАННОГО ФАЙЛА ===")
    raw_check = mne.io.read_raw_edf(edf_path, verbose=False)
    check_data, _ = raw_check[:]
    print(f"EDF данные: мин={check_data.min():.6f} V, макс={check_data.max():.6f} V")
    print(f"EDF данные: мин={check_data.min()*1e6:.2f} μV, макс={check_data.max()*1e6:.2f} μV")
    
    print("\nDone")
    return edf_path

def batch_convert(directory):
    """Конвертирует все PKL файлы в директории"""
    pkl_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    print(f"Найдено {len(pkl_files)} PKL файлов")
    for pkl_file in pkl_files:
        pkl_path = os.path.join(directory, pkl_file)
        try:
            convert_pkl_to_edf(pkl_path, directory)
        except Exception as e:
            print(f"❌ Ошибка при конвертации {pkl_file}: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    if os.path.isfile(path) and path.endswith('.pkl'):
        convert_pkl_to_edf(path)
    elif os.path.isdir(path):
        batch_convert(path)
