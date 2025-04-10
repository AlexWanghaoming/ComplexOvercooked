import io, json, pickle, pstats, cProfile,os,uuid,tempfile
import numpy as np
from pathlib import Path

# I/O

def save_pickle(data, filename):
    with open(fix_filetype(filename, ".pickle"), "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(filename):
    
    with open(fix_filetype(filename, ".pickle"), "rb") as f:
        return pickle.load(f)

def load_dict_from_file(filepath):
    with open(filepath, "r") as f:
        return eval(f.read())

def save_dict_to_file(dic, filename):
    dic = dict(dic)
    with open(fix_filetype(filename, ".txt"),"w") as f:
        f.write(str(dic))

def load_dict_from_txt(filename):
    return load_dict_from_file(fix_filetype(filename, ".txt"))

def save_as_json(filename, data):
    with open(fix_filetype(filename, ".json"), "w") as outfile:
        json.dump(data, outfile)

def load_from_json(filename):
    with open(fix_filetype(filename, ".json"), "r") as json_file:
        return json.load(json_file)

def iterate_over_files_in_dir(dir_path):
    pathlist = Path(dir_path).glob("*.json")
    return [str(path) for path in pathlist]

def fix_filetype(path, filetype):
    if path[-len(filetype):] == filetype:
        return path
    else:
        return path + filetype
