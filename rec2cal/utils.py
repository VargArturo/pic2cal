""" Module with parsing utilities/auxilirary functions
"""
import numpy as np
import csv 



def parse_ingredients(recipe):
  """
  Takes an input of the form:
    [{'text': 'yogurt, greek, plain, nonfat'},
     {'text': 'strawberries, raw'},
     {'text': 'cereals ready-to-eat, granola, homemade'}]

  And returns a concatenated string sentence.

  "yogurt-greek-plain-nonfat strawberries-raw, cereals-ready-to-eat, granola-homemade"

  You can brain storm better sentence representations, read papers etc.

  Args:
    :ingredients: lst as specified in the the description
  
  Returns:
    sentence in strong form
  """
  lst = []

  format = lambda x: x.replace(', ', '-')
  for i in range(len(recipe["ingredients"])):
    ingredient = format(recipe["ingredients"][i]["text"])
    quantity = format(recipe["quantity"][i]["text"])
    unit = format(recipe["unit"][i]["text"])

    lst.append(f"{quantity} {unit} of {ingredient}")

  return ", ".join(lst)



def parse_instructions(instructions):
  """
  Takes an input of the form:
    [{'text': 'Layer all ingredients in a serving dish.'}]

  And returns a concatenated string sentence.

  "Layer all ingredients in a serving dish."

  You can brain storm better sentence representations, read papers etc.

  Args:
    :instructions: lst as specified in the the description
  
  Returns:
    sentence in strong form
  """
  lst = []
  for ing in instructions:
    lst += [ing["text"]]
  
  return " ".join(lst)


def make_data(lst, partition="train"):
  """
  This function takes the data and produces a dataset ready for regression.

  Args:
    :lst[List]: parsed Recipie 1M json
    :partition[String]: can be either "train", "test", "val"
  """
  dataset_X = []
  dataset_y = []
  for x in lst:
    if x["partition"] == partition:
      ingredients = parse_ingredients(x)
      instructions = parse_instructions(x["instructions"])
      calories = (x["nutr_values_per100g"]["energy"]/100) * sum(x['weight_per_ingr'])
      sentence = ingredients + ". " + instructions 

      dataset_X += [sentence] 
      dataset_y += [calories]
  return dataset_X, dataset_y

 
def save_np_data(file_name, X,y):
    """
    
    saves X and y into npz file file_name in the format of a dicionary with keys X and Y

    """
    with open(file_name, "wb") as outfile:
        np.savez(outfile, X=X, y=y)


def load_np_data(file_name):
    """
    
    this opens the npz file file_name as a dicionary with keys X, y
    
    """
    with open(file_name, "rb") as outfile:
        np_dict = np.load(outfile)
        X, y = np_dict["X"], np_dict["y"]
    return (X, y)


def write_csv(data_set, data_path_name, csv_name="/open_ai_simple_prompts.csv"):
    """
    Saves data_set into a csv
    """
    data = [x.values() for x in data_set ]
    with open(data_path_name + "/open_ai_simple_prompts.csv", 'w', encoding='UTF8', newline='') as f:
      writer = csv.writer(f)

      # write the header
      writer.writerow(("prompt", "completion"))

      # write multiple rows
      writer.writerows(data)
