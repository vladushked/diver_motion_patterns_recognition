import os

def find_dir(number, path, name):
    for dirname in os.listdir(path):
        splitted = dirname.split("-")
        if splitted[0] != name:
            continue
        if (int(splitted[1]) < number <= int(splitted[2])):
            subpath = os.path.join(path, dirname)
            for subdirname in os.listdir(subpath):
                subsplitted = subdirname.split(name)
                if subsplitted[0] != "":
                    continue
                if int(subsplitted[1]) == number:
                    dest_path = os.path.join(subpath, subdirname)
                    for dest_file in os.listdir(dest_path):
                        if dest_file.split(".")[1] == "csv":
                            yield dest_path, dest_file
