import torch
import torch.nn as nn

from Core.functions.utils import generate_square_subsequent_mask, sample_predictions


class MultiClassification_head(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MultiClassification_head, self).__init__()
        self.layer = nn.Linear(input_size, input_size)
        self.activation = nn.ReLU()
        self.classification_layer = nn.Linear(input_size, num_classes)
        self.head = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)

    def forward(self, x):
        x = self.activation(self.layer(x))
        pkt_repr = self.classification_layer(x)
        x = self.head(pkt_repr)
        return x, pkt_repr

class MultiClassification_head_flow(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate=0.3):
        super(MultiClassification_head_flow, self).__init__()
        hidden_size1 = input_size
        hidden_size2 = input_size // 2
        
        # First hidden layer block
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.BatchNorm1d(hidden_size1),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Second hidden layer block
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size1, hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.classification_layer = nn.Linear(hidden_size2, num_classes)
        self.head = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
        
    def forward(self, x):
        # Forward pass through hidden layers
        x = self.layer1(x)
        x = self.layer2(x)
        pkt_repr = self.classification_layer(x)
        x = self.head(pkt_repr)
        
        return x, pkt_repr

class Attention_Luong(nn.Module):
    def __init__(self, attention_size):
        super(Attention_Luong, self).__init__()
        self.attention_size = attention_size
        self.linearLayer = torch.nn.Linear(attention_size, 1, bias = False)

    def forward(self, last_state, attention_mask):
        encoding_mask = torch.zeros(attention_mask.size())
        encoding_mask[:, 0] = 1
        mask_expanded = attention_mask.float().unsqueeze(-1).expand(last_state.shape)
        mask_expanded = mask_expanded.to(last_state.device)
        last_state = last_state * mask_expanded
        
        dot_prod = self.linearLayer(last_state).squeeze(2)              
        weights = torch.softmax(dot_prod, dim = 1).unsqueeze(1)   
        encoder_hidden_states = torch.matmul(weights, last_state).squeeze(1)          
        encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states.expand(encoder_hidden_states.shape[0], attention_mask.shape[1], encoder_hidden_states.shape[2])
        return encoder_hidden_states, encoding_mask

class ModelWithBottleneck(nn.Module):
    def __init__(self, model, type, pkt_dim, linear_layers=False, decoder=None, bottleneck=None):
        super(ModelWithBottleneck, self).__init__()
        self.encoder = model.encoder
        self.type_bottleneck = type
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.decoder_start_token_id = model.config.decoder_start_token_id
        self.pad_token_id = model.config.pad_token_id
        self.eos_token_id = model.config.eos_token_id
        self.lm_head = model.lm_head
        self.model_dim = model.model_dim
        self.pkt_repr_dim = pkt_dim
        self.linear_layers = linear_layers
        if self.linear_layers:
            self.compression_layer = nn.Linear(self.model_dim, pkt_dim)
            self.decompression_layer = nn.Linear(pkt_dim, self.model_dim)

    def forward(self, input_ids, attention_mask, decoder_input_ids=None, answer_len=None):
        encoder_output = self.encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                )
        if self.bottleneck is not None:
            self.encoder_hidden_states, attention_mask = self.bottleneck(
                encoder_output.last_hidden_state, attention_mask
            )
        else:
            self.encoder_hidden_states, attention_mask = self.select_fix_bottleneck(
                encoder_output.last_hidden_state, attention_mask
            )
        if self.linear_layers:
            self.encoder_hidden_states = self.compression_layer(self.encoder_hidden_states)
        if self.decoder is not None:
            decoder_mask = generate_square_subsequent_mask(
                decoder_input_ids.shape[1]
            )
            if self.linear_layers:
                self.encoder_hidden_states = self.decompression_layer(self.encoder_hidden_states)

            decoder_mask = (
                decoder_mask.unsqueeze(0)
                .expand(
                    decoder_input_ids.shape[0],
                    decoder_mask.shape[0],
                    decoder_mask.shape[0],
                )
                .to(self.encoder.device)
            )
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_mask,
                encoder_hidden_states=self.encoder_hidden_states,
                encoder_attention_mask=attention_mask.to(self.encoder.device),
            )
            sequence_output = decoder_outputs[0]
            sequence_output = sequence_output * (self.model_dim**-0.5)
            lm_logits = self.lm_head(sequence_output)
            lm_logits = lm_logits.reshape(-1, self.decoder.config.vocab_size)
            if answer_len != None:
                output_ids = self.generate_output(answer_len, attention_mask)
                return lm_logits, output_ids
            return lm_logits, None
        return self.encoder_hidden_states
    
    def remove_decoder(self):
        self.decoder = None
        self.decompression_layer = None
    
    def generate_output(self, t_len, attention_mask):
        """
        generate_output
        ---------------
        The method is used to generate the ids of the output vector.
        Args:
            - encoding_mask, encoder_hidden_states (Tensors)

        Output:
            - generated_so_far (Tensor): it is the representation of the output.
        """
        decoding_step = 1
        generated_so_far = (
            torch.ones(attention_mask.shape[0], 1)
            .fill_(self.decoder_start_token_id)
            .type(torch.long)
            .to(self.encoder_hidden_states.device)
        )
        unfinished_sents = generated_so_far.new(generated_so_far.shape[0]).fill_(1)
        sent_lengths = generated_so_far.new(generated_so_far.shape[0]).fill_(t_len)

        while decoding_step < t_len:
            tgt_mask = generate_square_subsequent_mask(
                generated_so_far.shape[1]
            ).to(self.encoder_hidden_states.device)
            tgt_mask = (
                tgt_mask.unsqueeze(0)
                .expand(attention_mask.shape[0], tgt_mask.shape[0], tgt_mask.shape[0])
                .to(self.encoder_hidden_states.device)
            )
            decoder_outputs = self.decoder(
                input_ids=generated_so_far,
                attention_mask=tgt_mask,
                encoder_hidden_states=self.encoder_hidden_states,
                encoder_attention_mask=attention_mask.to(self.encoder.device),
            )
            sequence_output = decoder_outputs[0]
            # consider only last predicted word, B x V
            next_token_logits = self.lm_head(sequence_output)[:, -1, :]
            predictions = sample_predictions(
                next_token_logits, top_k=50, top_p=0.9
            )
            # pad finished sentences if eos_token_id exists
            tokens_to_add = (
                predictions * unfinished_sents
                + self.pad_token_id * (1 - unfinished_sents)
            )
            generated_so_far = torch.cat(
                [generated_so_far, tokens_to_add.unsqueeze(-1)], dim=-1
            )
            decoding_step += 1

            eos_in_sents = tokens_to_add == self.eos_token_id
            # if sentence is unfinished and the token to add is eos, sent lengths is filled with current length
            is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(
                eos_in_sents.long()
            ).bool()
            sent_lengths.masked_fill_(
                is_sents_unfinished_and_token_to_add_is_eos, decoding_step
            )
            # unfinished_sents is set to zero if eos in sentence
            unfinished_sents.mul_((~eos_in_sents).long())

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break
        return generated_so_far
    

    def select_fix_bottleneck(self, last_state, attention_mask):
        """
        select_fix_bottleneck
        ---------------------
        The method is used to identify the strategy to obtain the bottleneck
        of the model (between encoder and decoder).

        Args:
            - last_state (Tensor)
            - attention_mask (Tensor)

        Output:
            - encoder_hidden_states (Tensor): it is the last state modified by the
            selected strategy
            - encoding_mask (Tensor): it is the attention mask modified by the
            selected strategy. If it is necessary the first embedding of the
            encoder_hidden_states, it has 1 followed by all zeros.
        """
        encoding_mask = torch.zeros(attention_mask.shape)
        encoding_mask[:, 0] = 1  # Only the first is considered
        if self.type_bottleneck == "none":
            return last_state, attention_mask

        elif self.type_bottleneck == "first":
            emb_first_position = last_state[:, 0]
            encoder_hidden_states = emb_first_position.unsqueeze(1).expand(
                emb_first_position.shape[0],
                attention_mask.shape[1],
                emb_first_position.shape[1],
            )
            return encoder_hidden_states, encoding_mask

        elif self.type_bottleneck == "mean":
            mask_expanded = (
                attention_mask.float().unsqueeze(-1).expand(last_state.shape)
            )
            mask_expanded = mask_expanded.to(last_state.device)
            emb_mean = torch.sum(last_state * mask_expanded, 1) / torch.clamp(
                mask_expanded.sum(1), min=1e-9
            )
            encoder_hidden_states = emb_mean.unsqueeze(1).expand(
                emb_mean.shape[0], attention_mask.shape[1], emb_mean.shape[1]
            )
            return encoder_hidden_states, encoding_mask

        else:
            # Default strategy is None
            return last_state, attention_mask
    

    
class EncoderWithBottleneck(nn.Module):
    def __init__(self, model, type, bottleneck=None):
        super(EncoderWithBottleneck, self).__init__()
        self.encoder = model.encoder
        self.type_bottleneck = type
        self.bottleneck = bottleneck

    def forward(self, input_ids, attention_mask, decoder_input_ids=None):
        self.attention_mask = attention_mask
        encoder_output = self.encoder(
                    input_ids=input_ids, attention_mask=self.attention_mask
                )
        if self.bottleneck is not None:
             self.encoder_hidden_states, self.attention_mask = self.bottleneck(
                encoder_output.last_hidden_state, self.attention_mask
            )
        else:
             self.encoder_hidden_states, self.attention_mask = self.select_fix_bottleneck(
                encoder_output.last_hidden_state, self.attention_mask
            )

        return self.encoder_hidden_states, self.attention_mask
    
    def select_fix_bottleneck(self, last_state, attention_mask):
        """
        select_fix_bottleneck
        ---------------------
        The method is used to identify the strategy to obtain the bottleneck
        of the model (between encoder and decoder).

        Args:
            - last_state (Tensor)
            - attention_mask (Tensor)

        Output:
            - encoder_hidden_states (Tensor): it is the last state modified by the
            selected strategy
            - encoding_mask (Tensor): it is the attention mask modified by the
            selected strategy. If it is necessary the first embedding of the
            encoder_hidden_states, it has 1 followed by all zeros.
        """
        encoding_mask = torch.zeros(attention_mask.shape)
        encoding_mask[:, 0] = 1  # Only the first is considered
        if self.type_bottleneck == "none":
            return last_state, attention_mask

        elif self.type_bottleneck == "first":
            emb_first_position = last_state[:, 0]
            encoder_hidden_states = emb_first_position.unsqueeze(1).expand(
                emb_first_position.shape[0],
                attention_mask.shape[1],
                emb_first_position.shape[1],
            )
            return encoder_hidden_states, encoding_mask

        elif self.type_bottleneck == "mean":
            mask_expanded = (
                attention_mask.float().unsqueeze(-1).expand(last_state.shape)
            )
            mask_expanded = mask_expanded.to(last_state.device)
            emb_mean = torch.sum(last_state * mask_expanded, 1) / torch.clamp(
                mask_expanded.sum(1), min=1e-9
            )
            encoder_hidden_states = emb_mean.unsqueeze(1).expand(
                emb_mean.shape[0], attention_mask.shape[1], emb_mean.shape[1]
            )
            return encoder_hidden_states, encoding_mask

        else:
            # Default strategy is None
            return last_state, attention_mask