def format_file(filename):
	import nltk

	corrected_text=[]
	with open(filename) as file:
		for line in file:
			idx=line.find(" ")
			if idx!=-1:
				phrases=[l.strip().lstrip() for l in line[idx:].split(".")]
				corrected_text.extend(phrase for phrase in phrases)
				#corrected_text.extend(b[0]+" "+ b[1] for b in nltk.bigrams(words for phrase in phrases for words in phrase.split(" ") ))
	with open("text_format.txt",'w') as file:
		for w in corrected_text:
			file.write(w)
			file.write("\n")

def format_file_2(filename):
	import csv

	corrected_text=[]
	with open(filename,encoding="latin-1") as file:
		reader=csv.DictReader(file,delimiter="\t")
		for line in reader:
			corrected_text.append(line['Palabra'])
	with open("words_format.txt",'w') as file:
		for w in corrected_text:

			file.write(w)
			file.write("\n")

def format_bigrams(filename):
	import nltk
	corrected_text=[]
	with open(filename) as file:
		for line in file:
			words=line.strip().split(" ")
			corrected_text.extend([b[0]+(" " if len(b[1])>1 or b[1].isalnum() else "")+b[1] for b in nltk.bigrams(words) if b[0]!="@" and b[1]!="@"])
	with open("bigrams_format.txt",'w') as file:
		for w in corrected_text:
			file.write(w)
			file.write("\n")

def preprocess_bigrams(filename):
	import nltk
	corrected_text=[]

	with open(filename) as file:
		freq=nltk.FreqDist([c for bi in file for c in bi])
		print(freq.most_common())
		count=0
		file.seek(0)
		for line in file:
			count=count+1
			if line.count("=")>1:
				for bigram in line.split("="):
					if bigram and all([freq[c]>30 for c in bigram]):
						corrected_text.append(bigram.strip())

			elif line and all([freq[c]>30 for c in line]):
				corrected_text.append(line.strip())
		print(count)
		print(len(corrected_text))

	with open(filename,'w') as file:
		for w in corrected_text:
			file.write(w)
			file.write("\n")



def generate_examples(filename):
	import random
	num_samples=256
	samples=[]
	for i in range(num_samples):
		num1=random.randint(0,100)
		num2=random.randint(0,100)
		op=random.randint(0,1)
		text=str(num1)+(" + " if op==0 else " - ")+str(num2)
		samples.append(text)

	with open(filename,'w') as file:
		for w in samples:
			file.write(w)
			file.write("\n")

def search_for_long(filename,num):
	with open(filename) as file:
		for line in file:
			if len(line)>num:
				print(line)

def get_from_freq(filename):
	import csv
	count=0
	corrected_text=[]
	with open(filename) as file:
		reader=csv.DictReader(file,delimiter="\t")
		with open("words_2_format.txt",'w') as w_file:
			for line in reader:
				if int(line["f_raw"])>500:
					w_file.write(line['token...'])
					w_file.write("\n")


if __name__=="__main__":
	#format_file("text.txt")
	#format_file_2("10000_formas.TXT")
	#format_bigrams("text_format.txt")
	#preprocess_bigrams("bigrams_format.txt")
	#generate_examples("examples_format.txt")
	#search_for_long("bigrams_format.txt",35)
	get_from_freq("escow14ax.freq10.w/escow14ax.freq10.w.tsv")
	preprocess_bigrams("words_2_format.txt")
