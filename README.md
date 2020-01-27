# GradBoostOptim
Distributed optimization of hyperparameters of gradient boosting algorithms

## Описание
Данный репозиторий содержит ресурсы, полученные в результате научной работы на тему "Распределенная оптимизация гиперпараметров алгоритмов градиентного бустинга"

Главный каталог содержит три подкаталога:
- **data** - директория, содержащая тестовый датасет и данные, полученные в ходе работы
- **graphs** - директория, которая сдержит графики, полученные в ходе проведения экспериментов
- **test 1** - директория, содержащая .ipynb файлы, использованные для проведения первого эксперимента, ход и описание которого содержатся в pdf файле на [диске](https://drive.google.com/file/d/1P81hWve80FIj2JRjVHmDaDiv85Va4k2v/view)

Папки **data** и **graphs** также содержат папку **test 1**.

## Первый эксперимент
__Цель__: Обосновать правомощность использования предлагаемого способа оптимизации гиперпараметров XGBoost.

Для достижения поставленной цели был проведен эксперимент, в рамках которого были сравнены резульатыт работы трех методов оптимизации: поиск по сетке, случайный поиск и поиск с разбиением на группы. Сравнение проводилось на наборе данных Facebook Comment Volume Dataset.

Был произведен сопутствующий анализ данных, в ходе которого было выявлено, что поиск с разбиением показал себя не хуже, чем случайный поиск, а также были сделаны интересные выводы, связанные с гиперпараметрами, временем работы и числом итераций, необходимым для достижения оптимума.



