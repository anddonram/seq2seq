import dnnspell as dnn
from keras.models import Model,Sequential,load_model
from keras.layers import Dense,Dropout, RepeatVector, Activation, LSTM,\
        TimeDistributed, Input,recurrent,Masking,Embedding

import numpy as np

#Hidden layer size
latent_dim=1024
batch_size=64
LSTM_LAYERS=2
DROPOUT=0.2
EMB_DIM=32
model_filename="spell.h5"

def text_to_embedding(wrong_text, corrected_text,num_texts,
    max_encoder_seq_length, num_encoder_tokens,
    max_decoder_seq_length, num_decoder_tokens,
    input_token_index,target_token_index):
    encoder_input_data = np.zeros(\
        (num_texts, max_encoder_seq_length),
        dtype='float32')
    decoder_target_data = np.zeros(\
        (num_texts, max_decoder_seq_length),
    dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(wrong_text, corrected_text)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t] = input_token_index[char]

            decoder_target_data[i, t] = target_token_index[char]

    return encoder_input_data, decoder_target_data


def create_model(params_filename,read_filename,samples):
    """Generate the model"""
    corrected_text=dnn.read_file(read_filename,samples)
    wrong_text=[dnn.noise(text,2) for text in corrected_text]
    corrected_text=['\t'+text+'\n' for text in corrected_text]
    wrong_text=['\t'+text+'\n' for text in wrong_text]

    input_token_index,target_token_index,\
        input_characters,target_characters,\
        max_encoder_seq_length,num_encoder_tokens,\
        max_decoder_seq_length,num_decoder_tokens=dnn.get_parameters_from_file(params_filename)

    for k in target_token_index.keys():
        target_token_index[k]=target_token_index[k]+1
    aux_char=['']
    aux_char.extend(target_characters)
    target_characters=aux_char

    input_token_index=target_token_index
    input_characters=target_characters
    num_encoder_tokens=num_decoder_tokens

    encoder_input_data, decoder_target_data = text_to_embedding(wrong_text, corrected_text,samples,
    max_encoder_seq_length, num_encoder_tokens,
    max_decoder_seq_length, num_decoder_tokens,
    input_token_index,target_token_index)

    initialization = "he_normal"
    model = Sequential()
    # "Encode" the input sequence using an RNN, producing an output of hidden_size
    # note: in a situation where your input sequences have a variable length,
    # use input_shape=(None, nb_feature).
    model.add(Embedding(num_encoder_tokens+1,EMB_DIM,input_length=max_encoder_seq_length,mask_zero=True))

    for layer_number in range(LSTM_LAYERS):
        model.add(recurrent.LSTM(latent_dim,  kernel_initializer=initialization,
        return_sequences=layer_number + 1 < LSTM_LAYERS))
        model.add(Dropout(DROPOUT))
    # For the decoder's input, we repeat the encoded input for each time step
    model.add(RepeatVector(max_encoder_seq_length))
    # The decoder RNN could be multiple layers stacked or a single layer
    for _ in range(LSTM_LAYERS):
        model.add(recurrent.LSTM(latent_dim, return_sequences=True, kernel_initializer=initialization))
        model.add(Dropout(DROPOUT))

    # For each of step of the output sequence, decide which character should be chosen
    model.add(TimeDistributed(Dense(num_decoder_tokens, kernel_initializer=initialization)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.save(model_filename)


def load_and_train_model(params_filename,read_filename,samples,train,epochs):
    corrected_text=dnn.read_file(read_filename,samples)
    wrong_text=[dnn.noise(text,2) for text in corrected_text]
    corrected_text=['\t'+text+'\n' for text in corrected_text]
    wrong_text=['\t'+text+'\n' for text in wrong_text]

    input_token_index,target_token_index,\
        input_characters,target_characters,\
        max_encoder_seq_length,num_encoder_tokens,\
        max_decoder_seq_length,num_decoder_tokens=dnn.get_parameters_from_file(params_filename)


    for k in target_token_index.keys():
        target_token_index[k]=target_token_index[k]+1
    aux_char=['']
    aux_char.extend(target_characters)
    target_characters=aux_char

    input_token_index=target_token_index
    input_characters=target_characters
    num_encoder_tokens=num_decoder_tokens

    encoder_input_data, decoder_target_data = text_to_embedding(wrong_text, corrected_text,samples,
    max_encoder_seq_length, num_encoder_tokens,
    max_decoder_seq_length, num_decoder_tokens,
    input_token_index,target_token_index)


    decoder_target_data=np.expand_dims(decoder_target_data,axis=0)
    print(encoder_input_data.shape)
    print(decoder_target_data.shape)
    model=load_model(model_filename)
    if train:
        model.fit(encoder_input_data, decoder_target_data,
          batch_size=batch_size,
          epochs=epochs)
        model.save(model_filename)
    for i in range(10):
        input_seq = encoder_input_data[i:i+1]
        target_sequence=model.predict(input_seq)
        decoded_sentence = "".join([target_characters[np.argmax(token)] for token in target_sequence[0,:]])
        if decoded_sentence.find("\n")!=-1:
            decoded_sentence=decoded_sentence[0:decoded_sentence.find("\n")]
        print('-')
        print('Input sentence:', wrong_text[i])
        print('Decoded sentence:', decoded_sentence)
        print('Expected sentence:', corrected_text[i])

if __name__=="__main__":
    samples=256
    samples_filename="bigrams_format.txt"
    params_filename='dnnspell_params.txt'
    create_model(params_filename,samples_filename,samples)
    load_and_train_model(params_filename,samples_filename,samples,True,10)
