import os
import sys
import math
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
import torch
import operator
import csv

"""
Author of query_distilbert: Carolyn Anderson
Author of Evaluation: Rachel Shurberg and Hope Zhu
Date: 12/20/2022
"""

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")

def get_most_likely_cloze(textprompt):
  """get the most likely fill-in-the-blank option according to DistilBERT"""
  before,after = textprompt.split('___')
  text = before+'[MASK]'+after
  inputs = tokenizer(text, return_tensors="pt")
  with torch.no_grad():
    logits = model(**inputs).logits
  mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
  predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
  most_likely_word = tokenizer.decode(predicted_token_id)
  most_likely_prob = torch.nn.functional.softmax(logits[0, mask_token_index]).amax(axis=-1).item()
  return (most_likely_word.strip(),most_likely_prob)

def assess_cloze_probability(textprompt,choices):
  """assess fill-in-blank probability of all choices"""
  probs = []
  before,after = textprompt.split('___')
  for c in choices:
    c_idx = tokenizer(c)['input_ids']
    c_len = len(c_idx)-2
    text = before+'[MASK] '*(c_len-1)+'[MASK]'+after
    inputs = tokenizer(text, return_tensors="pt")
    labels = tokenizer(before+c+after, return_tensors="pt")["input_ids"]
    labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)
    
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss.item()
    probs.append(math.exp(-loss)) # loss is negative log likelihood. so, multiply by -1 and apply inverse of log function
  return probs

def choice_query(prompt,choices):
  if '___' not in prompt:
    print("Prompt must include a BLANK")
    return
  wordlist = choices.split(';')
  best_word = get_most_likely_cloze(prompt)
  wordlist.append(best_word[0])
  probs = assess_cloze_probability(prompt,wordlist)
  return {w:probs[i] for i,w in enumerate(wordlist)}

def read_file(fileName):
   # read in file using csv
  with open(fileName) as file:
    reader = csv.reader(file, delimiter="\t")
    # create data list
    dataList = []
    for line in reader:
      # for each line, set the sentence as prompt
      # calculate how many choices there are and separate them with ';'
      # call choice_query using the prompt and choices values
      ids = line[0]
      pNum=line[1]
      sport=line[3]
      prompt = line[4]
      choices = line[5] + ';' + line[6] + ';' + line[7]
      probs = choice_query(prompt, choices)
      # add these results to the data list
      row = [ids,pNum,sport, prompt, choices, probs]
      dataList.append(row)
  return dataList

def write_file(fileName, dataList):
  # write file using csv
  with open(fileName, 'w') as file:
    writer = csv.writer(file, delimiter="\t",lineterminator='\n')
    for row in dataList:
      writer.writerow(row)

def gender_eval(genderList):
    # create the dictionary with the sports
    genderEval={'Basketball':0, 'Golf':0, 'Soccer':0, 'Volleyball':0, 'Football':0, 'Tennis':0, 'Gymnastics':0, 'Swimming':0,'Cheerleading':0, 'Dance':0, 'Ballet Dance':0, 'Wrestling':0}
    # Key words for each category
    fem=["She has",'She','Her',"She doesn't","She has",'girl',"women's",'girls','women','woman','she watches','She is']
    male=['He has','He','His',"He doesn't",'He has','boy',"men's",'him','boys','men','he watches','man',"He is"]
    neutral=['They have','They','Their',"They don't",'They have', 'student','USA','fun','them','kids','they watch','Americans','professional','They are']
    neg=['not', "don't", 'never','but only', 'worst', 'least']
    dataList=[]
    favor=''
    
    # loop through each prompt and output
    for row in genderList:
        print(row[0])
        probs=row[5]
        
        #remove extra predicted word from dictionary
        other=['he','i','a','has']
        for key in list(probs.keys()):
            for word in other:
                if key==word:
                    del probs[key]
                    
        for key in list(probs.keys()):
            if key not in row[4]:
                del probs[key]
        print(probs)
        
        # get maximum probability option
        maxi = max(probs.items(), key = operator.itemgetter(1))[0]
        print(maxi)
        
        # evaluate the love, doesn't mind, hate sentence frames
        if 7<=int(row[1]) and int(row[1])<=10:
            if maxi=="doesn't mind":
                genderEval[row[2]]+=0
                favor='neutral'
            if maxi=='loves':
                if 'boy' in row[3]:
                    genderEval[row[2]]-=1
                    favor='male'
                elif 'girl' in row[3]:
                    genderEval[row[2]]+=1
                    favor='female'
            if maxi=="hates":
                if 'boy' in row[3]:
                    genderEval[row[2]]+=1
                    favor='male'
                elif 'girl' in row[3]:
                    genderEval[row[2]]-=1
                    favor='female'
        
        else:
            # check if the prompt is a negative prompt
            negative=False
            for word in neg:
                if word in row[3]:
                    negative=True
                    
            # for the negative prompts assign points opposite of selection
            if negative:
                if maxi in male:
                    genderEval[row[2]]+=1
                    favor='female'
                elif maxi in fem:
                    genderEval[row[2]]-=1
                    favor='male'
                else:
                    favor='neutral'
                    
            # for non negative prompts, assign points in line with selection
            else:
                if maxi in fem:
                    genderEval[row[2]]+=1
                    favor='female'
                elif maxi in male:
                    genderEval[row[2]]-=1
                    favor='male'
                else:
                    favor='neutral'
                    pass
        
        
        # remove the string of choices and the result dictionary
        row.pop()
        row.pop()
        
        # add the choices and probabilities as lists
        row=row+list(probs.keys())
        row=row+list(probs.values())
        # add which gender the prompt favors
        row.append(favor)
        dataList.append(row)
        print()
      
    # write the results tsv file  
    write_file("results.tsv", dataList)

    # return the sport scores
    return genderEval
      

def char_eval(charList):
  write_file("char.tsv", dataList)

def emotion_eval(emotionList):
  write_file("emotion.tsv", dataList)


def main():
    # read in the prompts
    dataList = read_file("prompts.tsv")
    # perform the evaluation
    gDict=gender_eval(dataList)
    # get the results
    print(gDict)
main()