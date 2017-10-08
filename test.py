import json
from pprint import pprint
import glob, os
import gzip



# data =[]
os.chdir("./")
for file in glob.glob("*.gz"):
    print(file)
    filename = file[:-3] + '.json'
    f = open(filename,'w+')
    f.write("")
    f.close()
    i=0
    f = open(filename , 'a+')
    # with open(file) as data_file:
    with  gzip.open(file, 'r') as data_file: 
        for line in data_file:  
            if 'delete' in line:
                # i+=1
                continue
            try:
                 data =json.loads(line)
                 f.write("<DOC>\n")
                 f.write("\t<DOCNO>" + data['id_str'] + "</DOCNO>\n")
                 f.write("\t<TIME>" + (data['created_at']) + "</TIME>\n")
                 f.write("\t<USER>" + (data['user']['id_str']) + "</USER>\n")
                 f.write("\t<TEXT>\n" + "\t"+(data['text']).encode('utf-8')  + "\n\t</TEXT>\n")
                 f.write("</DOC>\n")
                 # i+=1gu
            except Exception as e:
                if i%5000==0:
                    print(e)
                    print(i)
                i+=1


             # f.write(str(data[text']) + '\n')
    f.close()

    print("Number of errors: " , i);
