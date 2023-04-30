""" Module with parsing utilities/auxilirary functions
"""
import numpy as np
import csv 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.datasets import make_regression
from copy import deepcopy


def parse_ingredients(recipe):
  """
   this method takes an input from a dictionary of the form:
  
    {'fsa_lights_per100g': Dict[str,str],
     'id': str,
     'ingredients': List[Dict[str,str]] ,
     'instructions': List[Dict[str,str]] ,
     'nutr_per_ingredient': List[Dict[str,int]], 
     'nutr_values_per100g': Dict[str,int],
     'partition': str,
     'quantity': List[Dict[str,str]],
     'title': str,
     'unit': List[Dict[str,str]],
     'url': str,
     'weight_per_ingr': List[int]
    } 

  Args:
    :recipe(Dict[str, any]):
  
  Returns:
    :str: sentence with the ingredients and their measurements
  """
  lst = []
  format = lambda x: x.replace(', ', '-') # this creates the function format that will replace every comma with a dash
  for i in range(len(recipe["ingredients"])): # this for loop iterates for the specific keys "ingredients", "quantity" and "unit" within the recipe dictionary. 
    ingredient = format(recipe["ingredients"][i]["text"])
    quantity = format(recipe["quantity"][i]["text"])
    unit = format(recipe["unit"][i]["text"])

    lst.append(f"{quantity} {unit} of {ingredient}") # fills empty list with each ingredient, quantity and units

  return ", ".join(lst) # returns the list as a single string



def parse_instructions(instructions):
  """
  Takes an input of the form:
    [{'text': 'Layer all ingredients in a serving dish.'}, {'text': 'Bake in oven at 100 degrees.'}]

  And returns a concatenated string sentence.

  "Layer all ingredients in a serving dish. Bake in oven at 100 degrees"

  Args:
    :instructions(List[Dict[str,str]]): lst as specified in the  description
  
  Returns:
    :str: sentence with the recipes instructions
  """
  lst = []
  for i in instructions: # iterates for every unique dictionary within the list 
    lst += [i["text"]] # fills empty list with the values of each unique dictionary
  
  return " ".join(lst) # returns all the instructions seperated by spaces


def make_data(recipe, partition="train"): 
  """
  This function takes the input from a LIST of unique dictionaries each of the form:
  
    {'fsa_lights_per100g': Dict[str,str],
     'id': str,
     'ingredients': List[Dict[str,str]] ,
     'instructions': List[Dict[str,str]] ,
     'nutr_per_ingredient': List[Dict[str,int]], 
     'nutr_values_per100g': Dict[str,int],
     'partition': str,
     'quantity': List[Dict[str,str]],
     'title': str,
     'unit': List[Dict[str,str]],
     'url': str,
     'weight_per_ingr': List[int]
    } 

  and produces the datasets necessary for regression.

  Args:
    :recipe(List[Dict[str, any]]): parsed Recipie 1M json
    :partition[str]: parition is assigned a default value "train" in the absence of a specified partition either "train", "test" or "val"
  
  Returns:
    :dataset_X(List[str]): features/inputs for regression model 
    :dataset_y(List[int]): labels for regression model
  """
  dataset_X = [] # dataset of recipe information (ingridients + instructions), this is the input/feautures for the model
  dataset_y = [] # dataset of calorie amounts for each recipe, this is the labels for the model
  for x in recipe: # iterates for every item within the list of recipies
    if x["partition"] == partition: # This conditional statement establishes that items within the list that are only from the specified partition ('train','test' or 'val) in the params are selected.
      ingredients = parse_ingredients(x) # ingredients is assigned a sentence form of all the ingriedients and their information
      instructions = parse_instructions(x["instructions"]) # instructions is assigned a sentence form of all the instructions
      calories = (x["nutr_values_per100g"]["energy"]/100) * sum(x['weight_per_ingr']) 
      sentence = ingredients + ". " + instructions 

      dataset_X += [sentence] 
      dataset_y += [calories]
  return dataset_X, dataset_y

 
def save_np_data(file_name, X,y):
    """
    
    saves X and y into npz file file_name in the format of a dicionary with keys X and Y.
    
    Args:
      :file_name[str]:
      :X(List[str]): vector embeddings for the recipe information.
      :y(List[int]): calorie information (either real or predicted)
      
    Returns:
      None
    """
    with open(file_name, "wb") as outfile:
        np.savez(outfile, X=X, y=y)


def load_np_data(file_name):
    """
    
    this opens the npz file file_name as a dicionary with keys X, y and returns the items with keys X and y seperately
    
    Args:
      :file_name[str]:

    Returns:
      :X(List[str]): vector embeddings for the recipe information.
      :y(List[int]): calorie information (either real or predicted)
    
    """
    with open(file_name, "rb") as outfile:
        np_dict = np.load(outfile)
        X, y = np_dict["X"], np_dict["y"]
    return (X, y)


def write_csv(data_set, data_path_name, csv_name="/open_ai_simple_prompts.csv"):
    """
    Saves data_set for open_ai GPT model into a csv
    
    Args:
      :data_set(List[Dict[str,str]]): list of a unique dictionary for each recipe with keys prompt and completion
      :data_path_name(str): file destination
      :csv_name(str): filename
      
    Returns:
      None
    
    """
    data = [x.values() for x in data_set ]
    with open(data_path_name + "/open_ai_simple_prompts.csv", 'w', encoding='UTF8', newline='') as f:
      writer = csv.writer(f)

      # write the header
      writer.writerow(("prompt", "completion"))

      # write multiple rows
      writer.writerows(data)

def train_model_iteratively(
    model, X_train, y_train,
    X_val, y_val, n_iter=2000):
  """
  This is the same as model.fit but instead it is done by steps / iteration
  so that we can save the train and validation losses per iteration allowing
  us to diagnose overfitting and carry out early stopping.

  Note we also checkpoint the model at each iteration.
  partial fit carries out one step of gradient descent (or which ever optimser
  is being used).
  """
  train_loss = np.zeros(n_iter)
  val_loss = np.zeros(n_iter)
  partial_fits = []

  for i in range(n_iter):
      model.partial_fit(X_train, y_train)
      y_train_pred = model.predict(X_train)
      y_val_pred = model.predict(X_val)

      train_loss[i] = mean_squared_error(y_train, y_train_pred)
      val_loss[i] = mean_squared_error(y_val, y_val_pred)
      partial_fits.append(deepcopy(model))
  return train_loss, val_loss, partial_fits

def pick_best_model(validation_loses, partial_fits):
  """
  This function takes all the validation losses per iteration
  and the corresponding trained model per iteration then it selects
  the model and training iteration with the smallest validation error.
  """
  i = np.argmin(validation_loses)
  return partial_fits[i], i