import torch
import torch.nn as nn
# small dataset
eng_hin_pairs = {
    "Hello.": "नमस्ते।",
    "How are you?": "आप कैसे हैं?",
    #"I am fine.": "मैं ठीक हूँ।",
    #"What is your name?": "आपका नाम क्या है?",
    #"My name is Arpit.": "मेरा नाम अर्पित है।",
    #"Nice to meet you.": "आपसे मिलकर खुशी हुई।",
    #"Where are you from?": "आप कहाँ से हैं?",
   # "I am from India.": "मैं भारत से हूँ।",
   # "What do you do?": "आप क्या करते हैं?",
   # "I am a student.": "मैं छात्र हूँ।",
   # "Do you speak Hindi?": "क्या आप हिंदी बोलते हैं?",
   # "Yes, a little.": "हाँ, थोड़ा सा।",
   # "Thank you.": "धन्यवाद।",
   # "You're welcome.": "आपका स्वागत है।",
   # "Good morning.": "सुप्रभात।",
   #"Good night.": "शुभ रात्रि।",
   # "See you later.": "फिर मिलेंगे।",
   # "I love programming.": "मुझे प्रोग्रामिंग पसंद है।",
    #"Can you help me?": "क्या आप मेरी मदद कर सकते हैं?",
    #"This is very difficult.": "यह बहुत कठिन है।"
}
#max len of english sentence = 5
#max len of hindi sentence = 8


eng_vocab={'<UNK>': 0,
    '<PAD>': 1,
    '<SOS>': 2,
    '<EOS>': 3}
for sentence in eng_hin_pairs.keys():
    for word in sentence.split():
        if word not in eng_vocab:
            eng_vocab[word] = len(eng_vocab)
 
hindi_vocab={'<UNK>': 0,
    '<PAD>': 1,
    '<SOS>': 2,
    '<EOS>': 3}    
for sentence in eng_hin_pairs.values():
    for word in sentence.split():
        if word not in hindi_vocab:
            hindi_vocab[word] = len(hindi_vocab)    

def tokenize(sentence,vocab):
    return [vocab.get(word,vocab['<UNK>']) for word in sentence.split()]

dataset = []
for en, hi in eng_hin_pairs.items():
    en_ids = [eng_vocab['<SOS>']] + tokenize(en, eng_vocab) + [eng_vocab['<EOS>']]
    hi_ids = [hindi_vocab['<SOS>']] + tokenize(hi, hindi_vocab) + [hindi_vocab['<EOS>']]
    dataset.append((en_ids, hi_ids))

def apply_padding(sentence,max_len, pad_id):
    return sentence + [pad_id] * (max_len - len(sentence))
padded_dataset = []
for pair in dataset:
    eng_sen=apply_padding(pair[0], 10, eng_vocab['<PAD>'])
    hin_sen=apply_padding(pair[1], 10, hindi_vocab['<PAD>'])
    padded_dataset.append((eng_sen, hin_sen))
    
decoder_input = [pair[1][:-1] for pair in padded_dataset]  # Exclude the last token
decoder_target = [pair[1][1:] for pair in padded_dataset]  # Exclude the first token

src_batch=torch.tensor([pair[0] for pair in padded_dataset],dtype=torch.long)  
tgt_batch=torch.tensor(decoder_input, dtype=torch.long)
label_batch=torch.tensor(decoder_target, dtype=torch.long)
#print('shapes of - ','src_batch:',src_batch.shape,' tgt_batch:', tgt_batch.shape,' label_batch:', labels_batch.shape)

src_mask= (src_batch == eng_vocab['<PAD>']).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, src_seq_len)
print('src_mask shape',src_mask.shape)
tgt_len = tgt_batch.size(1)
tgt_pad_mask= (tgt_batch == hindi_vocab['<PAD>'] ).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, tgt_seq_len)
# Create a no look-ahead mask for the target sequence
tgt_look_ahead_mask =  torch.triu(torch.ones((tgt_len, tgt_len),dtype=torch.bool), diagonal=1)  # ( tgt_seq_len, tgt_seq_len)
tgt_look_ahead_mask = tgt_look_ahead_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, tgt_seq_len, tgt_seq_len)
tgt_mask = tgt_pad_mask | tgt_look_ahead_mask  # Combine padding and no look-ahead masks

#print('tgt_mask shape',tgt_mask.shape)
#print(tgt_mask[0][0])

