### Dataset

This dataset contains images of garbage items categorized into 10 classes, designed for machine learning and computer vision projects focusing on recycling and waste management. It is ideal for building classification or object detection models or developing AI-powered solutions for sustainable waste disposal.

[Dataset link](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2/data)

### Classifier

В качестве классификатора был выбран Random Forest

Результаты на тестовых изображениях получились следующие:

| Class       | Correct Predictions | Total Images | Percentage (%) |
|-------------|---------------------|--------------|----------------|
| trash       | 87                  | 190          | 45.8%          |
| glass       | 365                 | 613          | 59.6%          |
| clothes     | 1006                | 1066         | 94.4%          |
| metal       | 43                  | 204          | 21.1%          |
| plastic     | 171                 | 397          | 43.0%          |
| cardboard   | 216                 | 365          | 59.2%          |
| paper       | 183                 | 336          | 54.4%          |
| biological  | 136                 | 200          | 68.0%          |
| shoes       | 201                 | 396          | 50.8%          |

Более подробный тест представлен в ноутбуке `classifier_test.ipynb`

**NB:** Классификатор и тестирование производилось с сидом 42 для разбиения датасета на тестовые и тренировочные изображения