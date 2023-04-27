# py -m pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0
# py .\examples\classify_image.py --model .\test_data\mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite --labels .\test_data\inat_bird_labels.txt --input .\test_data\parrot.jpg
# pip3 install tflite-model-maker


from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader
from tflite_model_maker import model_spec


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    # TODO: OPTIONAL
    # spec = model_spec.get('efficientdet_lite0')
    # data = DataLoader.from_folder("data/")
    data = DataLoader.from_folder("DAS/slike/")
    model = image_classifier.create(data, epochs=5)
    # TODO: OPTIONAL
    # model = image_classifier.create(data, epochs=5, model_spec=spec)
    loss, accuracy = model.evaluate(data=data)
    print("Here")
    print(loss)
    print(accuracy)
    # signatures_dict = {
    #   'encode': model.encode.get_concrete_function(),
    #   'decode': model.decode.get_concrete_function()
    # }
    # model.export(export_dir=".", sig=signatures_dict)
    model.export(export_dir=".")
    print("Here")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
