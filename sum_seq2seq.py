from dnnspell import *
#####################################
#### Learn to sum from strings
#####################################
def generate_sum_model(params_filename,read_filename,samples):
	"""Generate the model"""
	ops=read_file("examples_format.txt",samples)
	results=[str(eval(text)) for text in ops]
	ops=[w+'\n' for w in ops]
	results=[w+'\n' for w in results]

	input_token_index={'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'0':10,' ':11,'+':12,'-':13,'\n':14}
	target_token_index=input_token_index
	input_characters=['0','1','2','3','4','5','6','7','8','9','0',' ','+','-','\n']
	target_characters=input_characters
	max_encoder_seq_length=10
	num_encoder_tokens=len(input_characters)
	max_decoder_seq_length=4
	num_decoder_tokens=num_encoder_tokens

	encoder_input_data, decoder_input_data, decoder_target_data = text_to_matrix(ops, results,samples,
			max_encoder_seq_length, num_decoder_tokens,
			max_decoder_seq_length, num_decoder_tokens,
			target_token_index,target_token_index)
	initialization = "he_normal"
	model = Sequential()
	# "Encode" the input sequence using an RNN, producing an output of hidden_size
	# note: in a situation where your input sequences have a variable length,
	# use input_shape=(None, nb_feature).
	for layer_number in range(LSTM_LAYERS):
		model.add(recurrent.LSTM(latent_dim, input_shape=(None, num_decoder_tokens), kernel_initializer=initialization,
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
	model.fit(encoder_input_data, decoder_input_data,
			batch_size=batch_size,
			epochs=epochs)
	model.save("sum.h5")

def load_sum_model(filepath,params_filename,read_filename,samples,train,epochs):
	ops=read_file("examples_format.txt",samples)
	results=[str(eval(text)) for text in ops]
	ops=[w+'\n' for w in ops]
	results=[w+'\n' for w in results]

	input_token_index={'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'0':10,' ':11,'+':12,'-':13,'\n':14}
	target_token_index=input_token_index
	input_characters=['0','1','2','3','4','5','6','7','8','9','0',' ','+','-','\n']
	target_characters=input_characters
	max_encoder_seq_length=10
	num_encoder_tokens=len(input_characters)
	max_decoder_seq_length=4
	num_decoder_tokens=num_encoder_tokens

	encoder_input_data, decoder_input_data, decoder_target_data = text_to_matrix(ops, results,samples,
			max_encoder_seq_length, num_decoder_tokens,
			max_decoder_seq_length, num_decoder_tokens,
			target_token_index,target_token_index)
	model=load_model("sum.h5")
	if train:
		model.fit(encoder_input_data, decoder_input_data,
				  batch_size=batch_size,
				  epochs=epochs)
		model.save("sum.h5")

	for i in range(10):
		input_seq = encoder_input_data[i:i+1]
		target_sequence=model.predict(input_seq)

		decoded_sentence = "".join([target_characters[np.argmax(token)] for token in target_sequence[0,:]])
		if decoded_sentence.find("\n")!=-1:
			decoded_sentence=decoded_sentence[0:decoded_sentence.find("\n")]
		print('-')
		print('Input sentence:', ops[i])
		print('Decoded sentence:', decoded_sentence)
		print('Expected sentence:', results[i])


if __name__=="__main__":
    pass
    # Warning: put the parameters correctly
    # generate_sum_model(params_filename,samples_filename,num_samples)
    # load_sum_model("attention.h5",params_filename,samples_filename,num_samples,train,epochs)
