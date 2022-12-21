# DistilBERT Representation of Gender Bias in Sports

This bias probe task investigates potential gender stereotypes in relation to sports in the distilBERT language generation model. Differences in opportunity based on gender have led to cultural stereotypes of the various sports and the nature of certain sports along with current participation rates lead to biases in sport based on gender. Using the language generation model distilBERT, we are looking to examine these biases, and in particular, to answer the question: Do different sports have different gender connotations in terms of performance and preferences? 

[Read full report here.](https://docs.google.com/document/d/1mBkt95ONR53-8cayFsDblsZGq_1Db2gJnNf1_R4Dv3Y/edit?usp=sharing)

[Short presentation slide here.](https://docs.google.com/presentation/d/1DtVTlVwyAIeKo6ZxtnxDrGXgMqJ4Gv2eNeL_v9L0YsY/edit?usp=sharing)

## Files:

- sports_distilbert.py
- prompts.tsv
- genderTest.tsv

## Replicate Process:
Run in terminal:

```bash
$ python3 sports_distilbert.py
```

prompts.tsv is automatically read by the sports_distilbert file. DistilBERT's results are stored in genderTest.tsv. The point system used as an evaluation metric for the report is recorded within the terminal itself.

## Authors
Rachel Shurberg, Hope Zhu
