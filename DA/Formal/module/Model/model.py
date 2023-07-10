from transformers import BertTokenizer, BertConfig, BertForTokenClassification, AutoTokenizer, AutoModelForSequenceClassification, pipeline
import json
import torch 
import torch.nn

class NER :
    def __init__(self,ckpt_path,conf_path) -> None:
        pretrained_model_name = "beomi/kcbert-base" 
        model_ckpt = torch.load(ckpt_path,map_location=torch.device("cpu"))
        model_config = BertConfig.from_pretrained(
            pretrained_model_name,
            num_labels = model_ckpt['classifier.bias'].shape.numel()
            )
        self.conf_path = conf_path
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name,do_lower_case=False)
        self.model = BertForTokenClassification(model_config) 
        self.label_map = self.load_config()

    def load_config(self) :
        labels = open(self.conf_path,"r").read()
        labels = json.loads(labels)['id2label']
        label_map = {}
        for idx in labels.keys() :
            label_map[int(idx)] = labels[idx]
        return label_map
    
    @staticmethod
    def NERtokenizer(script):
        tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base" ,do_lower_case=False)
        script_list = script.split('/')
        token_list = []
        for sentence in script_list : 
            token_list += tokenizer.tokenize(sentence)
        return token_list

    
    def inference_fn(self,sentence):
        inputs = self.tokenizer([sentence],max_length=128,padding="max_length",truncation=True)
        with torch.no_grad():
            outputs = self.model(**{k: torch.tensor(v) for k, v in inputs.items()})
            probs = outputs.logits[0].softmax(dim=1)
            _, preds = torch.topk(probs, dim=1, k=1)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            predicted_tags = [self.label_map[pred.item()] for pred in preds]
            result = []
            for token, predicted_tag in zip(tokens, predicted_tags):
                if all([token not in [self.tokenizer.pad_token, self.tokenizer.cls_token, self.tokenizer.sep_token]]):
                    predicted_tag
                    result.append([token, predicted_tag])
        return result
    
class Formal :
    def __init__(self) -> None:
        self.pretrained_model = AutoModelForSequenceClassification.from_pretrained("j5ng/kcbert-formal-classifier")
        self.tokenizer = AutoTokenizer.from_pretrained('j5ng/kcbert-formal-classifier')
        self.model = pipeline(task="text-classification", model = self.pretrained_model, tokenizer= self.tokenizer)

    def pred(self,sentence) : 
        return int(self.model(sentence)[0]['label'][-1])