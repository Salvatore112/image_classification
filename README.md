### Датасет

Этот набор данных содержит изображения предметов мусора, отсортированных по 10 классам, разработанным для проектов машинного обучения и компьютерного зрения, ориентированных на переработку и управление отходами. Он идеально подходит для построения моделей классификации или обнаружения объектов или разработки решений на базе ИИ для устойчивой утилизации отходов.

[Dataset link](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2/data)

### Классификатор

В качестве классификатора был выбран Random Forest

Результаты на тестовых изображениях получились следующие:

| Класс       | Корректные предсказания | Всего изображений | Процент (%) |
|-------------|-------------------------|-------------------|-------------|
| trash       | 87                      | 190               | 45.8%       |
| glass       | 365                     | 613               | 59.6%       |
| clothes     | 1006                    | 1066              | 94.4%       |
| metal       | 43                      | 204               | 21.1%       |
| plastic     | 171                     | 397               | 43.0%       |
| cardboard   | 216                     | 365               | 59.2%       |
| paper       | 183                     | 336               | 54.4%       |
| biological  | 136                     | 200               | 68.0%       |
| shoes       | 201                     | 396               | 50.8%       |

Более подробный тест представлен в ноутбуке `classifier_test.ipynb`

**NB:** Классификатор и тестирование производилось с зафиксированным сидом 42 для разбиения датасета на тестовую и тренировочную выборку
