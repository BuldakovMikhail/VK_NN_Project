import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


transform = A.Compose([
        A.Resize(416, 416),
        ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
)
