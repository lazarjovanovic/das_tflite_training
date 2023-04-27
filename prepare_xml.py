import os
from xml.etree import ElementTree as et
import imghdr

data_dir = "data/"

if __name__ == "__main__":
    files = os.listdir(data_dir)
    xml_files = list()
    image_files = list()
    for file in files:
        file_type = imghdr.what(os.path.join(data_dir, file))
        if file_type == "jpeg":
            image_files.append(file)
        else:
            xml_files.append(file)

    for xml_file in xml_files:
        file_path = os.path.join(data_dir, xml_file)
        tree = et.parse(file_path)
        field = tree.find("path")
        old_path = field.text
        new_path = old_path.split("/")[-1]
        field.text = new_path
        tree.write(file_path)
