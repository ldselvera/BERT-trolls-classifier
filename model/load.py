import torch
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification

def load_info():
    # Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top. 

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.cuda()
    model.load_state_dict(torch.load("model/model.pt"))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    return model, tokenizer