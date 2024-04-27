# Реализация алгоритма поиска наиболее вероятной глобальной гипотезы в рамках Хакатона "Phystech Radar Tech Challenge"

## Состав команды:
-   Песоцкая Алиса
-   Юсупов Сафуан
-   Ивандаев Роман


# Настройка окружения

Установка окружения

```
pip install -r requirements.txt
```


# Подготовка данных

Алгоритм принимает на вход 4 аргумента:
- путь к файлу в формате .csv с матрицей совместности 
```
Пример содержимого файла input_matrix.csv:
   0,   1,   1, 
   0,   0,   1,  
   0,   0,   0,   
```
- путь к файлу в формате .csv с весами гипотез 
```
Пример содержимого файла  weights.csv:
  7.7132, 0.2075, 8.2374
```
-  путь к файлу в формате .csv в который сохранятся топ-5 глобальных гипотез и их веса
```
Пример содержимого файла pred.csv:
 TH1,TH2,TH3,sum(w)
 1,0,1,69.91878
 1,0,1,69.23338
 1,0,1,67.88508
 1,0,1,67.27018
 1,0,1,66.58478
```
- **опционально** путь к файлу с ответами, полученными полным перебором
```
Пример содержимого файла out.csv:
 TH1,TH2,TH3,sum(w)
 1,0,1,69.91878
 1,0,1,69.23338
 1,0,1,67.88508
 1,0,1,67.27018
 1,0,1,66.58478
```

## Files overview

* [/src](https://github.com/SAFUANlip/Hakaton-Almaz/tree/master/src) - исходный код
* [/data](https://github.com/SAFUANlip/Hakaton-Almaz/tree/master/src) - исходный код

### Main code

* [data/utils/create_norm_data.py]() - нормировка и обрезка исходных данных для дальнейшей нарезки тайлов
* [data/splitter.py]() - создать тайлы
* [data/dataset_pipeline.py]() - создать обучающую и валидационную выборку
* [train.py]() - обучение
* [inference/detect.py]() - инференс и утилиты для него

### Tools
* [utils/logger.py]() - логгеры TF и neptune
* [data/utils/transform.py]() - разные трансформации для cv2
* [losses.py]() - функции ошибок
