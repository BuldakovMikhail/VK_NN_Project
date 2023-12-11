# Класс Dataset

```python
data = DetectionDataset(annot_path, transform, max_len)
```
- annot_path - путь до файла с аннотациями,
- transform - трансформации входного изображения

annot_path должен собой представлять .csv-файл со следующими столбцами:
- image_id: путь до изображения
- boxes: список bounding boxes, содержащихся в изображении. Их формат [x_min, y_min, x_max, y_max]
- labels: список классов, которые есть на изображении
