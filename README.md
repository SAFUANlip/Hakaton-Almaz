# Реализация алгоритма поиска наиболее вероятных глобальных траекторных гипотез в рамках Хакатона "Phystech Radar Tech Challenge"

Визуализация обхода графа при поиске наилучшей глобальной гипотезы появится через пару секунд
![визуализация_расширенная.gif](gifs/визуализация_расширенная.gif)

## Состав команды Staying Alive:
-   Песоцкая Алиса
-   Юсупов Сафуан
-   Ивандаев Роман


# Настройка окружения

Установка окружения

```
pip install -r requirements.txt
```

**Версия Python: 3.10**

# Подготовка данных

Алгоритм принимает на вход 2 аргумента:
- путь к файлу в формате .csv с матрицей совместности и весами
```
Пример содержимого файла input_with_weights_path.csv:
   0,   1,   1, 
   0,   0,   1,  
   0,   0,   0, 
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


## Обзор содержимого директорий

* [/src](https://github.com/SAFUANlip/Hakaton-Almaz/tree/master/src) - исходный код
* [/data](https://github.com/SAFUANlip/Hakaton-Almaz/tree/master/data) - входные и выходные данные
* [/gifs](https://github.com/SAFUANlip/Hakaton-Almaz/tree/master/gifs) - визуализация обхода графа

## Основной код

* [src/main.py]() - поиск топ-5 глобальных траекторных гипотез

## Как получить оптимальные гипотезы с помощью нашего алгоритма
- подготовить данные, как указано выше
- положить их в папку /data
- указать пути к входным данным в src/main.py

```
Пример путей к входным данным в main.py(), main()
    input_with_weights_path = "../data/input_with_weights_path.csv"
    pred_path = "../data/pred1.csv"
```

- запустить main.py

**По заданному раннее пути pred_path соханятся глобальные гипотезы и их веса**


Источники:
- Kati Rozman, AnGhysels, Dušanka Janežič & Janez Konc “An exact algorithm to find a maximum weight clique in a weighted undirected graph.” (2024)
- Jeffrey S, Hicks, Illya V. “Combinatorial Branch-and-Bound for the Maximum Weight Independent Set Problem.” Technical Report, Texas A&M University Warren,(2016).
- W.A., Neto, M.B.C., Rodrigues, C.D., Michelon, P. “Um algoritmo de branch and bound para o problema da clique máxima ponderada.” Proceedings of XLVII SBPO 1 Tavares, (2015).
