import json
import os
import random

from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from django.views.decorators.clickjacking import xframe_options_exempt
# from bert_lm import SemanticBERT_LM_predictions
import run_squad_prediction as f

# BLM_Conditoinal_SemanticBERT = SemanticBERT_LM_predictions('conditional-semanticbert')
# BLM_SemanticBERT = SemanticBERT_LM_predictions('semanticbert')
# BLM_BERT_SQuAD = SemanticBERT_LM_predictions('bert(squad)')

@csrf_exempt
@xframe_options_exempt
def show_demo(request):
    context = {
        "alg": "best",
        "model": "bert-pretrained-squad2.0"
    }

    return render(request, 'qa.html', context)


@csrf_exempt
@xframe_options_exempt
def predict_answers(request, sent1, sent2, alg, model):
    if model == 'bert-pretrained-squad2.0':
        predictedTokens = f.run_prediction(sent1, sent2,model)
        if alg == "best":
            print("The best prediction")
            predictedTokens = predictedTokens[0:1]
        elif alg == "topn":
            print("Top N prediction")
            predictedTokens = predictedTokens[0:5]
    # elif model == 'semanticbert':
    #     print('model: SemanticBERT')
    #     if alg == "best":
    #         print("The best prediction")
    #         predictedTokens = BLM_SemanticBERT.calculate_bert_masked_beam_search(sent1, sent2, beam_size=20)
    #         predictedTokens = predictedTokens[0:1]
    #     elif alg == "topn":
    #         print("Top N prediction")
    #         predictedTokens = BLM_SemanticBERT.calculate_bert_masked_beam_search(sent1, sent2, beam_size=20)
    #         predictedTokens = predictedTokens[0:5]
    # elif model == 'bert(squad)':
    #     print('model: BERT pre-trained on SQuAD')
    #     if alg == "best":
    #         print("The best prediction")
    #         predictedTokens = BLM_BERT_SQuAD.calculate_bert_masked_beam_search(sent1, sent2, beam_size=20)
    #         predictedTokens = predictedTokens[0:1]
    #     elif alg == "topn":
    #         print("Top N prediction")
    #         predictedTokens = BLM_BERT_SQuAD.calculate_bert_masked_beam_search(sent1, sent2, beam_size=20)
    #         predictedTokens = predictedTokens[0:5]

    table = []
    for rowId in range(len(predictedTokens)):
        row = [(predictedTokens[rowId]['text'], predictedTokens[rowId]['probability'])]
        table.append(row)

    context = {
        "predictedTokens": predictedTokens,
        "table": table,
        "sent1": sent1,
        "sent2": sent2,
        "alg": alg,
        'model': model
    }
    return render(request, 'qa.html', context)
