#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        dropout_rate = 0.3
        pad_token_idx = vocab.char2id['<pad>']
        self.embed_size = embed_size
        self.embeddings = nn.Embedding(len(vocab.char2id), embed_size, padding_idx=pad_token_idx)
        self.CNN = CNN(embed_size, embed_size, 5)
        self.highway = Highway(embed_size)
        self.drop = nn.Dropout(dropout_rate)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1f

        embed = self.embeddings(input)
        x_reshaped= embed.permute(0, 1, 3, 2)

        word_embeds = []
        for t in x_reshaped.split(1):
            x_conv_out = self.CNN(t.squeeze(0))
            x_highway = self.highway(x_conv_out)
            x_word_emb = self.drop(x_highway)
            word_embeds.append(x_word_emb.unsqueeze(0))
        x_embeds = torch.cat(word_embeds)

        return x_embeds
        ### END YOUR CODE

