from pipeline import load_models, process_image

if __name__ == '__main__':
    load_models()

    test_image = '066f992fdc_pred_kgo_empty_17.jpg'

    result = process_image(test_image)

    if result:
        print(f"{result}")
    else:
        print("Нет обнаружений")