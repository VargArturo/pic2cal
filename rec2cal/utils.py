""" Module with parsing utilities/auxilirary functions
"""



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
 
