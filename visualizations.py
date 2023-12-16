import torchvision.transforms as T


def sanitize_prediction(prediction, min_score):
    indices = torchvision.ops.nms(prediction['boxes'], prediction['scores'], 0.2)
    mask = prediction['scores'][indices] > min_score
    indices = torch.masked_select(indices, mask)
    for param in prediction:
        prediction[param] = prediction[param][indices]
    return prediction


def visualize_prediction(image_path, model, min_score):
    transform = T.Compose([
        T.Resize(size=(416, 416)),
        T.ToTensor(),
    ])

    default_image = Image.open(image_path).convert("RGB")
    test_image = transform(default_image)

    with torch.no_grad():
        predictions = model(test_image)
    prediction = predictions[0]

    prediction = sanitize_prediction(prediction, min_score)

    image = denormalizer(test_image)
    boxes = torch.Tensor(prediction['boxes'])
    labels = prediction['labels']
    labels = [CLASSES[idx] for idx in labels.tolist()]
    colors = 'blue'
    processed_image = torchvision.utils.draw_bounding_boxes(image, boxes, labels, colors=colors, width=3)
    processed_image = processed_image.permute(1, 2, 0)
    plt.imshow(processed_image)
    plt.show()

    return prediction
