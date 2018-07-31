import pickle
import os


def does_file_exist(path):
    return os.path.isfile(path)

def read_pickle_file(file_path):
    print("Reading pickle file: %s" % file_path)
    pickle_file = open(file_path, "r")
    result = pickle.load(pickle_file)
    pickle_file.close()
    print("\tDone!!")
    return result


def save_using_pickle(python_object, file_path):
    print("Writing python object to file: %s" % file_path)
    pickle_file = open(file_path, "w")
    pickle.dump(python_object, pickle_file)
    pickle_file.close()
    print("\tDone!!")


def write_to_file(content, file_path):
    file_ = open(file_path, "w")
    file_.write(content)
    file_.close()


def read_txt_file(file_path):
    file_ = open(file_path, "r")
    data = file_.read()
    file_.close()
    return data