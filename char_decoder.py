#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.

        super(CharDecoder, self).__init__()
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(target_vocab.char2id), bias=True)
        self.pad_token_idx = target_vocab.char2id['<pad>']
        self.decoderCharEmb = nn.Embedding(len(target_vocab.char2id), char_embedding_size, padding_idx=self.pad_token_idx)
        self.target_vocab = target_vocab

        ### END YOUR CODE

    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        x_embed = self.decoderCharEmb(input) # (length, batch, embed_size)
        output, dec_hidden = self.charDecoder(x_embed, dec_hidden) # output is (length, batch, hidden_size)
        output = output.transpose(1,0) # now (batch, length, hidden_size)
        scores = self.char_output_projection(output) # (batch, length, vocab_size)
        return scores.permute(1,0,2), dec_hidden

        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch, for every character in the sequence.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).

        self.charDecoder.train()
        cross_entropy = nn.CrossEntropyLoss(ignore_index=self.pad_token_idx, reduction='sum')
        char_sequence_input = char_sequence.narrow(0, 0, char_sequence.shape[0]-1) # (length-1, batch)
        char_sequence_output = char_sequence.narrow(0, 1, char_sequence.shape[0] - 1) # (length-1, batch)
        scores, dec_hidden = self.forward(char_sequence_input, dec_hidden)
        loss = cross_entropy(scores.permute(0, 2, 1), char_sequence_output)
        return loss

        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

        output_words = [[] for i in range(initialStates[0].shape[1])]
        current_chars = [['{'] for i in range(initialStates[0].shape[1])]
        sm = nn.Softmax(dim=2)
        self.charDecoder.eval()
        dec_hidden = initialStates

        for i in range(max_length):
            ids = [self.target_vocab.char2id[char[0]] for char in current_chars]
            input = torch.tensor(ids).long().unsqueeze(0).to(device=device)
            scores, dec_hidden = self.forward(input, dec_hidden=dec_hidden)
            probs = sm(scores)
            current_chars = [[self.target_vocab.id2char[torch.argmax(prob.squeeze(0)).item()]] for prob in probs.squeeze(0).split(1)]
            output_words = [x[0] + x[1] for x in zip(output_words, current_chars)]
        output_words = [''.join(lst) for lst in output_words]
        for word_idx, word in enumerate(output_words):
            for char_idx, char in enumerate(word):
                if char == '}':
                    output_words[word_idx] = word[0:char_idx]
                    break
        return output_words
        
        ### END YOUR CODE

