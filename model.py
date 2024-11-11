from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
from openai import OpenAI
import anthropic
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import os

# Eric Bennett 11.11.24
# Data from running this script is in the scores.csv file in this directory
# Would not recommend running again but this is how I did it.



texts = pd.read_csv("yue_wei_texts.csv")
models = ['sikubert', 'bert-base-multilingual-uncased', 'sikuroberta', 'xlm-roberta-base', 'gpt-4o', 'gpt-4o-mini', 'claude-3-opus-20240229', 'claude-3-haiku-20240307']  # Add more models if needed

openai_api_key = "" #paste api key here
anthropic_api_key = ""
aimodel = 'gpt-4o'

class Config:
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        

config = Config()

# establish the two clients for use later
config.openai_client = OpenAI(api_key=openai_api_key)
config.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)


csv_file = "scores.csv"
if os.path.exists(csv_file):
    output_df = pd.read_csv(csv_file)
else:
    output_df = pd.DataFrame(columns=['model', 'text_name', 'index', 'source', 'result_1', 'score_1', 
                                      'result_2', 'score_2', 'result_3', 'score_3', 'result_4', 
                                      'score_4', 'masked_text', 'date_created'])

print("downloading models...")

if 'sikubert' in models:
    tokenizer_siku_bert = AutoTokenizer.from_pretrained("SIKU-BERT/sikubert")
    model_siku_bert = AutoModelForMaskedLM.from_pretrained("SIKU-BERT/sikubert")
    pipe_siku_bert = pipeline("fill-mask", model=model_siku_bert, tokenizer=tokenizer_siku_bert, top_k=4)
if 'bert-base-multilingual-uncased' in models:
    tokenizer_base_bert = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-uncased")
    model_base_bert = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-multilingual-uncased")
    pipe_base_bert = pipeline("fill-mask", model=model_base_bert, tokenizer=tokenizer_base_bert, top_k=4)
if 'sikuroberta' in models:
    tokenizer_siku_roberta = AutoTokenizer.from_pretrained("SIKU-BERT/sikuroberta")
    model_siku_roberta = AutoModelForMaskedLM.from_pretrained("SIKU-BERT/sikuroberta")
    pipe_siku_roberta = pipeline("fill-mask", model=model_siku_roberta, tokenizer=tokenizer_siku_roberta, top_k=4)
if 'xlm-roberta-base' in models:
    tokenizer_base_roberta = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base") #Note: mask must be <mask> for this one for some reason
    model_base_roberta = AutoModelForMaskedLM.from_pretrained("FacebookAI/xlm-roberta-base")
    pipe_base_roberta = pipeline("fill-mask", model=model_base_roberta, tokenizer=tokenizer_base_roberta, top_k=4)



for model_name in models:

    for _, row in texts.iterrows():

        text = row['text']
        textname = row['name']
        print(f"Total Iterations: {len(text)}")
        for i, char in tqdm(enumerate(text)):
            if not (char in '。!?？！「」，()[]{}.（）,：:;\'"'): #only masks characters that aren't punctuation possible marks
                masked_text = text[:i] + "<mask>" + text[i + 1:]
                
                if model_name == 'sikubert':
                    results = pipe_siku_bert(masked_text)
                elif model_name == 'sikuroberta':
                    results = pipe_siku_roberta(masked_text)
                elif model_name == 'xlm-roberta-base':
                    results = pipe_base_roberta(masked_text)
                elif model_name == 'gpt-4o':
                    completion = config.openai_client.chat.completions.create(
                            model=aimodel,
                            messages=[
                                {
                                    "role": "system",
                                    "content": "Given a Classical Chinese text missing one character, replaced with the following string: [MASK]. Respond only with your prediction for the single character (it will ALWAYS be one character)."
                                },
                                {
                                    
                                    "role": "user",
                                    "content": masked_text,
                                },
                            ]
                    )
                elif model_name == 'claude-3-haiku-20240307':
                    message = config.anthropic_client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=1000,
                        temperature=0,
                        system= f"Given a Classical Chinese text missing one character, replaced with the following string: [MASK]. Respond only with your prediction for the single character (it will ALWAYS be one character).",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": masked_text
                                    }
                                ]
                            }
                        ]
                    )
                else:
                    results = pipe_base_bert(masked_text)

                #ONLY EDIT PAST HERE vv
                if 'gpt-4o' in model_name:
                    result_scores = [("N/A", 0)] * 4
                    result_scores[0] = (completion.choices[0].message.content, 0)
                if 'claude-3-haiku-20240307' in model_name:
                    result_scores = [("N/A", 0)] * 4
                    result_scores[0] = (message.content[0].text, 0)
                
                if ('gpt-4o' not in model_name) and ('claude-3-haiku-20240307' not in model_name):
                    result_scores = [(result['token_str'], result['score']) for result in results]
                #print(f"Original char: {char}")
                #for token, score in result_scores:
                #   print(f"Prediction: {token}, Score: {score}")

                entry = pd.DataFrame([{
                    'model': model_name,
                    'text_name': textname,
                    'index': i,
                    'source': char,
                    'result_1': result_scores[0][0],
                    'score_1': result_scores[0][1],
                    'result_2': result_scores[1][0] if len(result_scores) > 1 else None,
                    'score_2': result_scores[1][1] if len(result_scores) > 1 else None,
                    'result_3': result_scores[2][0] if len(result_scores) > 2 else None,
                    'score_3': result_scores[2][1] if len(result_scores) > 2 else None,
                    'result_4': result_scores[3][0] if len(result_scores) > 3 else None,
                    'score_4': result_scores[3][1] if len(result_scores) > 3 else None,
                    'masked_text': masked_text,
                    'date_created': datetime.now().strftime("%Y-%m-%d")
                }])

                output_df = pd.concat([output_df, entry], ignore_index=True)