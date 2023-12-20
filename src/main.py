from predictions import visualize_prediction, load_model


if __name__ == "__main__":
    model = load_model()
    visualize_prediction('img_4.png', model, 0.3)
