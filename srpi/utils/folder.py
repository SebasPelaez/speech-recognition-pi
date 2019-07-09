from pathlib import Path
import os
import shutil


def create(path):
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except:
        print("An error occured")


def delete(path: str):
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
    except:
        print("An error occured")


def get_info(path: str):
    try:
        folder_list = next(os.walk(path))
        return (folder_list)
    except:
        print("An error occured")


def get_folders(path: str):
    return get_info(path)[1]


def get_files(path: str):
    return get_info(path)[2]


def create_multiple(paths: list):
    for path in paths:
        create(path)


def delete_multiple(paths: list):
    for path in paths:
        delete(path)

