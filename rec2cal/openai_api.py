from rec2cal.utils import parse_ingredients, parse_instructions
import openai
import re
import numpy as np
import time

# vvvvv

models = {
    "model1":{
        "model_name": "ada:ft-personal-2022-12-25-20-56-27",
        "openai.api_key": "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"
    },
    "model2":{
        "model_name": "ada:ft-personal-2022-12-27-00-26-31",
        "openai.api_key": "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"
    },
}


def format_data(lst, partition="train"):
  dataset_X = []
  dataset_y = []
  message = "Given the following recipe find the total calories using the following procedure:\n1.go through the list of ingredients and identify what each ingredient is one by one\n2.go through the instructions and identify the cooking method for each ingredient\n3.read through the ingredients once more and using their quantites and the identified cooking method for each find the calories for each ingredient\n4.Add all the calories for each ingredient to return the total calories of the dish\n"    
  for x in lst:
    if x["partition"] == partition:
      ingredients = parse_ingredients(x)
      instructions = parse_instructions(x["instructions"])
      calories = (x["nutr_values_per100g"]["energy"]/100) * sum(x['weight_per_ingr'])
      sentence = message + "\nIngredients:\n" + ingredients + ".\n\nInstructions:\n" + instructions 

      dataset_X += [sentence + "\n\n###\n\n"]
      dataset_y += [" total calories: " + str(calories)]
  
  f_data = [{'prompt':i, 'completion':j} for i,j in zip(dataset_X, dataset_y)]
                    
  return f_data


our_model = "ada:ft-personal-2022-12-25-20-56-27"
openai.api_key = "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"


def get_calories(prompt, model):
    """
    calls chatgpt on the prompt and extracts the calories from the answer 
    """
    mx_prmpt = 2049
    try:
        result = openai.Completion.create(
            model=model, prompt=prompt
        )
        txt = result['choices'][0]['text']
    except:
        txt = ""
        raise
    out = re.search(r"\d+\.\d+\s*(end)?", txt)
    try:
        if out is not None:
          out  = float(out[0].replace("end", "").replace(" ", ""))
    except:
        out = out[0].replace(" end", "")
        print(prompt, out)
    return out


def get_calories_for_datset(model, dataset, sleep_time=60):
    """
    This function calculates caloires for everything in the 
    given dataset using the provided model. 
    
    Sleep time is necessary as open ai api times out
    and crashes when too many requests are made
    """
    out_calories = []

    for i, x in enumerate(dataset):
      tmp = get_calories(x["prompt"],  model)
      out_calories.append(tmp)
      # Sleep to not exceed openai rate
      if tmp is None:
        print(f"Failed {i}")
      if len(out_calories)  % 43 == 0:
        time.sleep(sleep_time)
        print(f"Done with {len(out_calories)} prompts")
    return out_calories


# time.sleep(60)

# apply_ft_discriminator(validation[0]["prompt"], our_model)