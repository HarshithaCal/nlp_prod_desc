First, tried the distilgpt2 model (check distilgpt2.py) but results were bad. 
- This is expected as the distilgpt2 is designed for sentence completion rather than instruction following.
- Also, the model is very small with the context length of 1024 tokens (input + output).  The sample output is saved in the file distilgpt2_output.txt. 

- @inproceedings{sanh2019distilbert,
  title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},
  author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas},
  booktitle={NeurIPS EMC^2 Workshop},
  year={2019}
    }


Then, I tried finetuning distilgpt2:
- To finetune the model, I used the huggingface trainer. The training script is in the file finetune_distilgpt2.py. The training data is in the file data_distilgpt2.json.
- As the context length of distilgpt2 is 1024 tokens, I made sure that the input text length is not  long.