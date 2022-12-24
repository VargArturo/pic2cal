from os import path


# Repository root path
root_path = path.dirname(path.abspath(path.dirname(__file__)))

data_path = path.join(root_path, "data")
rep_json  = path.join(data_path, "recipes_with_nutritional_info.json")

