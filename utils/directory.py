import os

def create_dir(file):
    if not os.path.exists(file):
        os.makedirs(file)

def remove_dir(file):
    os.remove(file)