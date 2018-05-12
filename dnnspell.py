
from keras.models import Model,Sequential,load_model
from keras.layers import Dense,Dropout, RepeatVector, Activation, LSTM, TimeDistributed, Input,recurrent,Masking
from keras.utils.vis_utils import plot_model
import numpy as np
import random
from nltk import FreqDist
import seq2seq
from seq2seq.models import AttentionSeq2Seq,Seq2Seq

chars="qwertyuiopasdfghjklñzxcvbnm"
import ast
#Hidden layer size
latent_dim=256
batch_size=64
LSTM_LAYERS=2
DROPOUT=0.2


def noise(text,changes=1):
	res=text
	length=len(text)
	for i in range (changes):
		mod=random.randint(0,3)
		if length>1:
			if mod==0:
				#Insert
				idx=random.randint(0,length-1)
				char=random.choice(chars)
				res=text[:idx]+char+text[idx:]
			elif mod==1:
				#Delete
				idx=random.randint(0,length-1)
				res=text[:idx]+text[idx+1:]
			elif mod==2:
				#Replace
				idx=random.randint(0,length-1)
				char=random.choice(chars)
				res=text[:idx]+char+text[idx+1:]
			elif mod==3:
				#Swap
				if length>2:
					idx=random.randint(0,length-2)
					idx2=random.randint(idx+1,length-1)
					res=text[:idx]+text[idx2]+text[idx+1:idx2]+text[idx]+text[idx2+1:]
	return res

def read_file(filename,LIMIT):
	count=0
	corrected_text=[]
	with open(filename) as file:
		for line in file:
			corrected_text.append(line.strip().lstrip())
			count=count+1
			if LIMIT>0 and count==LIMIT:
				break


	return corrected_text

def save_parameters_to_file(read_filename,write_filename):
	corrected_text=read_file(read_filename,0)
	corrected_tokens=[c for word in corrected_text for c in word]

	max_variance=4
	extra_tokens=2
	corrected_freq=FreqDist(corrected_tokens)
	max_decoder_seq_length=max(len(x) for x in corrected_text)+extra_tokens
	num_decoder_tokens=corrected_freq.B()+extra_tokens

	max_encoder_seq_length=max_decoder_seq_length-extra_tokens+max_variance
	num_encoder_tokens=num_decoder_tokens

	input_characters = sorted(set(corrected_tokens).union(chars).union({'\n','\t'}))
	target_characters = sorted(set(corrected_tokens).union({'\n','\t'}))
	# print(input_characters)

	input_token_index = dict(
		[(char, i) for i, char in enumerate(input_characters)])
	target_token_index = dict(
		[(char, i) for i, char in enumerate(target_characters)])

	with open(write_filename,'w') as file:
		file.write(str(max_decoder_seq_length))
		file.write("\n")
		file.write(str(num_decoder_tokens))
		file.write("\n")
		file.write(str(max_encoder_seq_length))
		file.write("\n")
		file.write(str(num_encoder_tokens))
		file.write("\n")
		file.write(str(input_characters))
		file.write("\n")
		file.write(str(target_characters))
		file.write("\n")
		file.write(str(input_token_index))
		file.write("\n")
		file.write(str(target_token_index))
		file.write("\n")


def get_parameters_from_file(params_filename):

	input_token_index=None
	target_token_index=None
	with open(params_filename) as file:
		max_decoder_seq_length=ast.literal_eval(file.readline())
		num_decoder_tokens=ast.literal_eval(file.readline())
		max_encoder_seq_length=ast.literal_eval(file.readline())
		num_encoder_tokens=ast.literal_eval(file.readline())


		input_characters=ast.literal_eval(file.readline())
		target_characters=ast.literal_eval(file.readline())
		input_token_index=ast.literal_eval(file.readline())
		target_token_index=ast.literal_eval(file.readline())

	return input_token_index,target_token_index,\
			input_characters,target_characters,\
			max_encoder_seq_length,num_encoder_tokens,\
			max_decoder_seq_length,num_decoder_tokens

def generate_tokens(wrong_text,corrected_text,most_freq=30,output_len=15):

	wrong_tokens=[c for word in wrong_text for c in word]
	corrected_tokens=[c for word in corrected_text for c in word]

	wrong_freq=FreqDist(wrong_tokens)
	corrected_freq=FreqDist(corrected_tokens)

	wrong_chars=wrong_freq.most_common(most_freq)
	corrected_chars=corrected_freq.most_common(most_freq)
	# print(wrong_chars)
	# print(corrected_freq.N())
	# print(corrected_freq.B())

	input_characters = sorted(set(wrong_tokens))
	target_characters = sorted(set(corrected_tokens))
	# print(input_characters)

	input_token_index = dict(
	    [(char, i) for i, char in enumerate(input_characters)])
	target_token_index = dict(
	    [(char, i) for i, char in enumerate(target_characters)])
	return input_token_index,target_token_index,input_characters,target_characters,wrong_freq,corrected_freq


def text_to_matrix(wrong_text, corrected_text,num_texts,
		max_encoder_seq_length, num_encoder_tokens,
		max_decoder_seq_length, num_decoder_tokens,
		input_token_index,target_token_index):
	encoder_input_data = np.zeros(\
	    (num_texts, max_encoder_seq_length, num_encoder_tokens),
	    dtype='float32')
	decoder_input_data = np.zeros(\
	    (num_texts, max_decoder_seq_length, num_decoder_tokens),
	    dtype='float32')
	decoder_target_data = np.zeros(\
	    (num_texts, max_decoder_seq_length, num_decoder_tokens),
	dtype='float32')

	for i, (input_text, target_text) in enumerate(zip(wrong_text, corrected_text)):
		for t, char in enumerate(input_text):
			encoder_input_data[i, t, input_token_index[char]] = 1.0
		for t, char in enumerate(target_text):
			# decoder_target_data is ahead of decoder_input_data by one timestep
			decoder_input_data[i, t, target_token_index[char]] = 1.0
			if t > 0:
				# decoder_target_data will be ahead by one timestep
				# and will not include the start character.
				decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
	return encoder_input_data, decoder_input_data, decoder_target_data

def create_autoencoder_model(model_filename,params_filename):
	input_token_index,target_token_index,\
			input_characters,target_characters,\
			max_encoder_seq_length,num_encoder_tokens,\
			max_decoder_seq_length,num_decoder_tokens=get_parameters_from_file(params_filename)
	create_training_model(model_filename,num_encoder_tokens,num_decoder_tokens)

def create_training_model(model_filename,num_encoder_tokens,num_decoder_tokens):
	#Input,
	encoder_inputs = Input(shape=(None, num_encoder_tokens))


	encoder = LSTM(latent_dim, return_state=True,dropout=DROPOUT,recurrent_dropout=DROPOUT)
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)
	# We discard `encoder_outputs` and only keep the states.
	encoder_states = [state_h, state_c]

	# Set up the decoder, using `encoder_states` as initial state.
	decoder_inputs = Input(shape=(None, num_decoder_tokens))
	# We set up our decoder to return full output sequences,
	# and to return internal states as well. We don't use the
	# return states in the training model, but we will use them in inference.
	decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,dropout=DROPOUT,recurrent_dropout=DROPOUT)

	decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
	                                     initial_state=encoder_states)
	decoder_dense = Dense(num_decoder_tokens, activation='softmax')
	decoder_outputs = decoder_dense(decoder_outputs)

	# Define the model that will turn
	# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


	# Run training

	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
	model.save(model_filename)
	#model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
	return model,encoder_inputs,encoder_states,decoder_inputs,decoder_lstm,decoder_dense



def recover_model(filename):
    model=load_model(filename)

    encoder_inputs = model.input[0]   # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output   # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]   # input_2
    decoder_state_input_h = Input(shape=(latent_dim,), name='input_3')
    decoder_state_input_c = Input(shape=(latent_dim,), name='input_4')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    return model,encoder_model,decoder_model


def create_inference_model(encoder_inputs,encoder_states,decoder_inputs,decoder_lstm,decoder_dense):
	#Inference model
	encoder_model = Model(encoder_inputs, encoder_states)

	decoder_state_input_h = Input(shape=(latent_dim,))
	decoder_state_input_c = Input(shape=(latent_dim,))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
	decoder_outputs, state_h, state_c = decoder_lstm(\
	    decoder_inputs, initial_state=decoder_states_inputs)
	decoder_states = [state_h, state_c]
	decoder_outputs = decoder_dense(decoder_outputs)
	decoder_model = Model(\
	    [decoder_inputs] + decoder_states_inputs,
	    [decoder_outputs] + decoder_states)

	return encoder_model,decoder_model


def decode_sequence(input_seq,
			reverse_input_char_index,reverse_target_char_index,
			encoder_model,decoder_model,num_decoder_tokens,target_token_index,max_decoder_seq_length):
	# Encode the input as state vectors.
	states_value = encoder_model.predict(input_seq)

	# Generate empty target sequence of length 1.
	target_seq = np.zeros((1, 1, num_decoder_tokens))
	# Populate the first character of target sequence with the start character.
	target_seq[0, 0, target_token_index['\t']] = 1.

	# Sampling loop for a batch of sequences
	# (to simplify, here we assume a batch of size 1).
	stop_condition = False
	decoded_sentence = ''
	while not stop_condition:
		output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

		# Sample a token
		sampled_token_index = np.argmax(output_tokens[0, -1, :])
		sampled_char = reverse_target_char_index[sampled_token_index]
		decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
		if (sampled_char == '\n' or \
		 len(decoded_sentence) > max_decoder_seq_length):
			stop_condition = True

		# Update the target sequence (of length 1).
		target_seq = np.zeros((1, 1, num_decoder_tokens))
		target_seq[0, 0, sampled_token_index] = 1.

		# Update states
		states_value = [h, c]

	return decoded_sentence


def get_parameters(wrong_text,corrected_text,wrong_freq,corrected_freq):
	max_encoder_seq_length=max(len(x) for x in wrong_text)
	num_encoder_tokens=wrong_freq.B()
	max_decoder_seq_length=max(len(x) for x in corrected_text)
	num_decoder_tokens=corrected_freq.B()
	return max_encoder_seq_length,num_encoder_tokens,max_decoder_seq_length,num_decoder_tokens

def validate_autoencoder_model(model_filename,epochs,num_samples,samples_filename,bigrams,params_filename,repeats):

	noise_per_bigram=2 if bigrams else 1
	corrected_text=read_file(samples_filename,num_samples)
	wrong_text=[noise(text,noise_per_bigram)+'\n' for text in corrected_text]

	samples=len(corrected_text)*repeats
	for i in range(repeats-1):
		wrong_text.extend(noise(text,noise_per_bigram)+'\n' for text in corrected_text)

	corrected_text=['\t'+text+'\n' for text in corrected_text]
	corrected_text=corrected_text*repeats


	num_texts=len(wrong_text)

	input_token_index,target_token_index,\
			input_characters,target_characters,\
			max_encoder_seq_length,num_encoder_tokens,\
			max_decoder_seq_length,num_decoder_tokens=get_parameters_from_file(params_filename)
	print('Number of unique input tokens:', num_encoder_tokens)
	print('Number of unique output tokens:', num_decoder_tokens)
	print('Max sequence length for inputs:', max_encoder_seq_length)
	print('Max sequence length for outputs:', max_decoder_seq_length)

	samples_batch=40000
	real_batch=min(num_texts,samples_batch)
	encoder_input_data, decoder_input_data, decoder_target_data =text_to_matrix(wrong_text[:real_batch], corrected_text[:real_batch],real_batch,
			max_encoder_seq_length, num_encoder_tokens,
			max_decoder_seq_length, num_decoder_tokens,
			input_token_index,target_token_index)

	#Read from file
	model,encoder_model,decoder_model=recover_model(model_filename)
	if epochs:
		#Only retrain and save if requested
		for epoch in range(epochs):
			for i in range(0,num_texts,samples_batch):
				real_batch=min(num_texts-i,samples_batch)
				encoder_input_data, decoder_input_data, decoder_target_data = text_to_matrix(wrong_text[i:i+real_batch], corrected_text[i:i+real_batch],real_batch,
						max_encoder_seq_length, num_encoder_tokens,
						max_decoder_seq_length, num_decoder_tokens,
						input_token_index,target_token_index)
				model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
				          batch_size=batch_size,
				          epochs=1,
				          validation_split=1/repeats)
				# Save model
		model.save(model_filename)


	samples=20
	success=0
	for seq_index in range(samples):
		# Take one sequence (part of the training set)
		# for trying out decoding.
		input_seq = encoder_input_data[seq_index: seq_index + 1]
		decoded_sentence = decode_sequence(input_seq,input_characters,target_characters,
					encoder_model,decoder_model,num_decoder_tokens,target_token_index,max_decoder_seq_length)

		if decoded_sentence==corrected_text[i][0:-1]:
			success=success+1
		else:
			print('-')
			print('Input sentence:', wrong_text[seq_index])
			print('Decoded sentence:', decoded_sentence)
			print('Expected sentence:', corrected_text[seq_index])

	print("Accuracy:",success/samples)
def spell_checker(model_filename,params_filename,word):
	model,encoder_model,decoder_model=recover_model(model_filename)

	input_token_index,target_token_index,\
			input_characters,target_characters,\
			max_encoder_seq_length,num_encoder_tokens,\
			max_decoder_seq_length,num_decoder_tokens=get_parameters_from_file(params_filename)

	encoder_input_data, decoder_input_data, decoder_target_data = text_to_matrix([word[::-1]], [word],1,
			max_encoder_seq_length, num_encoder_tokens,
			max_decoder_seq_length, num_decoder_tokens,
			input_token_index,target_token_index)

	input_seq = encoder_input_data[0:1]

	decoded_sentence = decode_sequence(input_seq,input_characters,target_characters,
				encoder_model,decoder_model,num_decoder_tokens,target_token_index,max_decoder_seq_length)
	print('-')
	print('Input sentence:', word)
	print('Decoded sentence:', decoded_sentence)

def plot_dnn(filename):
	model,encoder_model,decoder_model=recover_model(filename)
	plot_model(model, to_file='model.png')
	plot_model(encoder_model, to_file='encoder_model.png')
	plot_model(decoder_model, to_file='decoder_model.png')



#####################################
#### Alternative version of the spellchecker
#####################################
def create_seq2seq_model(model_filename,params_filename):
	"""Create the model"""
	input_token_index,target_token_index,\
			input_characters,target_characters,\
			max_encoder_seq_length,num_encoder_tokens,\
			max_decoder_seq_length,num_decoder_tokens=get_parameters_from_file(params_filename)

	initialization = "he_normal"
	model = Sequential()
	# "Encode" the input sequence using an RNN, producing an output of hidden_size
	# note: in a situation where your input sequences have a variable length,
	# use input_shape=(None, nb_feature).
	model.add(Masking(input_shape=(None, num_encoder_tokens)))
	for layer_number in range(LSTM_LAYERS):

		model.add(recurrent.LSTM(latent_dim,  kernel_initializer=initialization,
			return_sequences=layer_number + 1 < LSTM_LAYERS))
		model.add(Dropout(DROPOUT))
		# For the decoder's input, we repeat the encoded input for each time step
	model.add(RepeatVector(max_decoder_seq_length))
	# The decoder RNN could be multiple layers stacked or a single layer
	for _ in range(LSTM_LAYERS):
		model.add(recurrent.LSTM(latent_dim, return_sequences=True, kernel_initializer=initialization))
		model.add(Dropout(DROPOUT))

	# For each of step of the output sequence, decide which character should be chosen
	model.add(TimeDistributed(Dense(num_decoder_tokens, kernel_initializer=initialization)))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.save(model_filename)

def correct_sentence(model,params_filename,sentence):
	word="\t"+sentence+"\n"
	input_token_index,target_token_index,\
			input_characters,target_characters,\
			max_encoder_seq_length,num_encoder_tokens,\
			max_decoder_seq_length,num_decoder_tokens=get_parameters_from_file(params_filename)

	encoder_input_data, decoder_input_data, decoder_target_data = text_to_matrix([word], [word],1,
			max_encoder_seq_length, num_encoder_tokens,
			max_decoder_seq_length, num_decoder_tokens,
			input_token_index,target_token_index)

	input_seq = encoder_input_data[0:1]

	target_predict=model.predict(encoder_input_data)
	target_sequence=target_predict[0:1]

	decoded_sentence = "".join([target_characters[np.argmax(token)] for token in target_sequence[0,:]])
	eos_index=decoded_sentence.find("\n")
	if eos_index!=-1:
		decoded_sentence=decoded_sentence[0:eos_index]


	print('-')
	print('Input sentence:', sentence)
	print('Decoded sentence:', decoded_sentence)



def load_seq2seq_model(model_filename):
	"""Load generic model"""
	return load_model(model_filename)

def validate_model(corrected_text,wrong_text,encoder_input_data,target_characters):
	samples=20
	success=0
	target_predict=model.predict(encoder_input_data)
	for i in range(samples):
		input_seq = encoder_input_data[i:i+1]
		target_sequence=target_predict[i:i+1]

		decoded_sentence = "".join([target_characters[np.argmax(token)] for token in target_sequence[0,:]])
		eos_index=decoded_sentence.find("\n")
		if eos_index!=-1:
			decoded_sentence=decoded_sentence[0:eos_index]

		if decoded_sentence==corrected_text[i][0:-1]:
			success=success+1
		else:
			print('-')
			print('Input sentence:', wrong_text[i])
			print('Decoded sentence:', decoded_sentence)
			print('Expected sentence:', corrected_text[i])

	print("Accuracy:",success/samples)

def validate_model(model_filename,model,params_filename,samples_filename,num_samples,epochs,repeats):
	corrected_text=read_file(samples_filename,num_samples)
	wrong_text=[noise(text,1)+'\n' for text in corrected_text]

	samples=len(corrected_text)*repeats
	for i in range(repeats-1):
		wrong_text.extend(noise(text,1)+'\n' for text in corrected_text)

	corrected_text=['\t'+text+'\n' for text in corrected_text]
	corrected_text=corrected_text*repeats


	input_token_index,target_token_index,\
			input_characters,target_characters,\
			max_encoder_seq_length,num_encoder_tokens,\
			max_decoder_seq_length,num_decoder_tokens=get_parameters_from_file(params_filename)

	encoder_input_data, decoder_input_data, decoder_target_data = text_to_matrix(wrong_text, corrected_text,samples,
			max_encoder_seq_length, num_decoder_tokens,
			max_decoder_seq_length, num_decoder_tokens,
			target_token_index,target_token_index)

	if epochs:
		model.fit(encoder_input_data, decoder_input_data,
				  batch_size=batch_size,
				  epochs=epochs,
				  validation_split=1/repeats)
		model.save(model_filename)

	validate_model(corrected_text,wrong_text,encoder_input_data,target_characters)

#####################################
#### Alternative version of the spellchecker, with attention
#####################################

def create_attention_model(model_filename,params_filename):
	input_token_index,target_token_index,\
			input_characters,target_characters,\
			max_encoder_seq_length,num_encoder_tokens,\
			max_decoder_seq_length,num_decoder_tokens=get_parameters_from_file(params_filename)

	model = AttentionSeq2Seq(input_dim=num_encoder_tokens, input_length=max_encoder_seq_length,
			hidden_dim=latent_dim,
			output_length=max_decoder_seq_length, output_dim=num_decoder_tokens,
			depth=LSTM_LAYERS,dropout=DROPOUT)

	model.compile(loss='mse', optimizer='rmsprop',metrics=['acc'])
	model.save(model_filename)

def load_attention_model(model_filename,params_filename):
	input_token_index,target_token_index,\
			input_characters,target_characters,\
			max_encoder_seq_length,num_encoder_tokens,\
			max_decoder_seq_length,num_decoder_tokens=get_parameters_from_file(params_filename)

	model = AttentionSeq2Seq(input_dim=num_encoder_tokens, input_length=max_encoder_seq_length,
			hidden_dim=latent_dim,
			output_length=max_decoder_seq_length, output_dim=num_decoder_tokens,
			depth=LSTM_LAYERS,dropout=DROPOUT)

	model.load_weights(model_filename)
	model.compile(loss='mse', optimizer='rmsprop',metrics=['acc'])

	return model


if __name__=="__main__":
	import argparse

	#If num_samples <=0, then take all examples
	num_samples=2000
	samples_filename="words_2_format.txt"

	model_filename='dnnspell.h5'
	params_filename='dnnspell_params.txt'
	from_file=False

	epochs=0
	repeats=8

	parser = argparse.ArgumentParser(description='Create an autoencoder deep learning model.')

	parser.add_argument('--load','-l', dest='model_filename',nargs="?", type=str,const=model_filename,
				help='loads the model from a file')

	parser.add_argument('--samples','-s',dest='samples',type=int,
				help='how many examples to train the model with (multiplied by 2, noisy and original samples)')
	parser.add_argument('--epochs','-e',dest='epochs',type=int,
				help='how many epochs to train the model')
	parser.add_argument('--write','-w',action='store_true',
				help='write parameters of samples to file')

	parser.add_argument('--plot','-p',action='store_true',
				help='plots the neural network into png')
	parser.add_argument('--correct','-c',dest='predict', type=str,
				help='corrects the text')
	parser.add_argument('--bigrams','-b',action='store_true',
				help='whether the training is made with bigrams or words only')
	parser.add_argument('--validate','-v',action='store_true',
				help='check the accuracy of the model')
	parser.add_argument('--attention','-a',action='store_true',
				help='try the attention model')
	parser.add_argument('--seq2seq',action='store_true',
				help='try the seq2seq model')
	args = parser.parse_args()
	if args.model_filename:
		model_filename=args.model_filename
		from_file=True

	bigrams=args.bigrams

	if bigrams:
		samples_filename="bigrams_format.txt"
	if args.samples is not None:
		num_samples=args.samples
	if args.epochs is not None:
		epochs=args.epochs

	if args.predict:
		model=None
		if args.attention:
			model=load_attention_model(model_filename,params_filename)
		elif args.seq2seq:
			model=load_seq2seq_model(model_filename)
		if model:
			correct_sentence(model,params_filename,args.predict)
		else:
			spell_checker(model_filename,params_filename,args.predict)


	elif args.validate:
		model=None
		if args.attention:
			model=load_attention_model(model_filename,params_filename)
		elif args.seq2seq:
			model=load_seq2seq_model(model_filename)
		if model:
			validate_model(model_filename,model,params_filename,samples_filename,num_samples,epochs,repeats)
		else:
			validate_autoencoder_model(model_filename,epochs,num_samples,samples_filename,bigrams,params_filename,repeats)

	elif args.attention:
		create_attention_model(model_filename,params_filename)
	elif args.seq2seq:
		create_seq2seq_model(model_filename,params_filename)
	elif args.plot:
		plot_dnn(model_filename)
	elif args.write:
		save_parameters_to_file(samples_filename,params_filename)
	else:
		create_autoencoder_model(model_filename,params_filename)
