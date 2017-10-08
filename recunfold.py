import json
import glob, os
from shutil import copyfile


with open('/home/ttg/tweetpair.json') as data_file:
	data = json.load(data_file)

for topic in data:
	# remove this if statement
	filenamein = '/home/ttg/PairPara/PairTweets'+topic+'.txt'
	# f = open(filename , 'r')
	filenameout = '/home/ttg/classifyParaphrases/input.txt'
	copyfile(filenamein, filenameout)
	print('Copied ' + filenamein)
	os.system('./classifyParaphrases.sh >> acc.txt')
	outname = '/home/ttg/classifyParaphrases/outputs/'+topic+'.txt'
	f = open(outname , 'w+')
	f.close()
	copyfile('hoha.txt' , outname)
	
	

	