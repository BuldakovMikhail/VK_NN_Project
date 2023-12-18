import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

from config import CLASSES, config
from models.food_detector_regressor import FoodDetectorRegressor


def denormalizer(tensor):
    tensor *= 255

    return tensor.type(torch.uint8)


def load_model():
    class_dict = {key: value for key, value in enumerate(CLASSES)}
    model = FoodDetectorRegressor(class_dict)
    model.load_state_dict(torch.load(config["REGRESSOR_WEIGHTS_PATH"], map_location=torch.device('cpu')))
    model.retina.load_state_dict(torch.load(config["DETECTOR_WEIGHTS_PATH"], map_location=torch.device('cpu')))
    return model


def sanitize_prediction(prediction, min_score):
    indices = torchvision.ops.nms(prediction['boxes'], prediction['scores'], 0.2)
    mask = prediction['scores'][indices] > min_score
    indices = torch.masked_select(indices, mask)
    for param in prediction:
        prediction[param] = prediction[param][indices]
    return prediction


def get_prediction(image_path, model, min_score):
    transform = T.Compose([
        T.Resize(size=(416, 416)),
        T.ToTensor(),
    ])

    default_image = Image.open(image_path).convert("RGB")
    test_image = transform(default_image)

    model.eval()
    model.cpu()
    with torch.no_grad():
        model.retina = model.retina.cpu()
        prediction_detection = model.retina(test_image)[0]
        embeddings = model.backbone(test_image[None, :, :, :].cpu())
        embed7 = embeddings['p7'].cpu()
        processed_embeddings = embed7.view(-1, 256 * 4 * 4)
        prediction_regression = model.regressor(processed_embeddings)[0]

    # print(
    #     f"calories: {prediction_regression[0]}, fats: {prediction_regression[1]}, carbs: {prediction_regression[2]}, proteins: {prediction_regression[3]}")

    # prediction_detection = predictions_detection[0]
    prediction_detection = sanitize_prediction(prediction_detection, min_score)

    image = denormalizer(test_image)
    boxes = torch.Tensor(prediction_detection['boxes'])
    labels = prediction_detection['labels']
    labels = [CLASSES[idx] for idx in labels.tolist()]
    colors = 'blue'
    processed_image = torchvision.utils.draw_bounding_boxes(image, boxes, labels, colors=colors, width=3)
    processed_image = processed_image.permute(1, 2, 0)

    return processed_image, prediction_regression


def visualize_prediction(image_path, model, min_score):
    image, values = get_prediction(image_path, model, min_score)
    print(
        f"calories: {values[0]}, fats: {values[1]}, carbs: {values[2]}, proteins: {values[3]}")

    plt.imshow(image)
    plt.show()
