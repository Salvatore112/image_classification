### Dataset

This dataset contains images of garbage items categorized into 10 classes, designed for machine learning and computer vision projects focusing on recycling and waste management. It is ideal for building classification or object detection models or developing AI-powered solutions for sustainable waste disposal.

[Dataset link](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2/data)

### Classifier

В качестве классификатора был выбран Random Forest

Результаты на тестовых изображениях получились следующие:

- Корректно было предсказано 87 из 190 изображений для класса 'trash'.
- Корректно было предсказано 365 из 613 изображений для класса 'glass'.
- Корректно было предсказано 1006 из 1066 изображений для класса 'clothes'.
- Корректно было предсказано 43 из 204 изображений для класса 'metal'.
- Корректно было предсказано 171 из 397 изображений для класса 'plastic'.
- Корректно было предсказано 216 из 365 изображений для класса 'cardboard'.
- Корректно было предсказано 183 из 336 изображений для класса 'paper'.
- Корректно было предсказано 136 из 200 изображений для класса 'biological'.
- Корректно было предсказано 201 из 396 изображений для класса 'shoes'.

Более подробный тест представлен в ноутбуке `classifier_test.ipynb`