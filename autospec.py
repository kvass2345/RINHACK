import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def process_audio_files(input_directory, output_directory):
    # Создайте новую директорию для сохранения изображений спектрограмм, если ее еще нет
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    def audio_cube(trace):
        # Создание полного пути к аудиофайлу
        audio_path_wav = os.path.join(input_directory, trace)
        
        # Загрузка аудиоданных и частоты дискретизации
        audio, sr = librosa.load(audio_path_wav)
        
        # Получение спектрограммы
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
        
        return spectrogram

    # Получение списка файлов входной директории
    all_audio_filenames = os.listdir(input_directory)

    # Проход по всем файлам в директории
    for audio_file in all_audio_filenames:
        # Проверка, что это файл формата WAV
        if audio_file.endswith('.wav'):
            # Обработка аудиофайла
            spectrogram = audio_cube(audio_file)
            
            # Определение пути для сохранения изображения
            ################################################################################
            ################################################################################
            ################################################################################
            """
            output_path = os.path.join(output_directory, "generated_"+audio_file.replace('.wav', '.png'))
            """
            output_path = os.path.join(output_directory, "real_"+audio_file.replace('.wav', '.png'))

            ################################################################################
            ################################################################################
            ################################################################################

            # Размер фигуры в дюймах для получения нужного разрешения
            fig_width_inch = 480 / plt.rcParams['figure.dpi']
            fig_height_inch = 360 / plt.rcParams['figure.dpi']

            # Создание фигуры с нужным размером
            fig, ax = plt.subplots(figsize=(fig_width_inch, fig_height_inch))
            plt.tight_layout()

            # Отображение спектрограммы
            librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), cmap='gray', y_axis='mel', x_axis='time')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel spectrogram')

            # Сохранение изображения спектрограммы с заданным разрешением
            plt.savefig(output_path, dpi=plt.rcParams['figure.dpi'], bbox_inches='tight')

            plt.close()

# Пример использования функции
input_dir = "C:\\Users\\Kerim\\Documents\\rthbv\\Выявление синтезированного голосаречи с использованием методов машинного обучения (ООО «Даталаб»)\\звук\\audio to spect"
output_dir = "C:\\Users\\Kerim\\Documents\\rthbv\\Выявление синтезированного голосаречи с использованием методов машинного обучения (ООО «Даталаб»)\\звук\\spect png to train"
process_audio_files(input_dir, output_dir)
