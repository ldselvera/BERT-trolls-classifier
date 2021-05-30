import torch
import numpy as np
from keras.preprocessing.sequence import pad_sequences
  
def prediction(sentence, model, tokenizer):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  sent = "[CLS] " + sentence + " [SEP]"

  tokenized_text = tokenizer.tokenize(sent)
  input_id = tokenizer.convert_tokens_to_ids(tokenized_text)

  MAX_LEN = 128
  # Create attention masks
  attention_masks = []
  input_id = pad_sequences([input_id], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

  for seq in input_id:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask) 
      
  prediction_inputs = torch.tensor(input_id)
  prediction_masks = torch.tensor(attention_masks)

  prediction_inputs = prediction_inputs.to(device)
  prediction_masks = prediction_masks.to(device)

  prediction_inputs = torch.tensor(prediction_inputs).to(device).long()

  with torch.no_grad():
      # Forward pass, calculate logit predictions
      logits = model(prediction_inputs, token_type_ids=None, attention_mask=prediction_masks)
      
  logits = logits.detach().cpu().numpy()   

  if np.argmax(logits, axis=1):
      result = "Troll"
  else:
      result = "Not Troll"

  return result