import numpy
import cPickle
import subprocess
import matplotlib.pyplot as plt
import matplotlib as mpl

class Parameter:
    def __init__(self, name, low_param, high_param, num_steps):
        self.name = name
        self.low_param = low_param
        self.high_param = high_param
        self.num_steps = num_steps
        if self.num_steps == 0:
            self.values = [self.low_param]
        if self.num_steps == 1: #exponential type
            self.values= list(10**exp for exp in range(self.low_param,self.high_param))
        elif self.num_steps > 1: #linear type goes from low to just below high
             step_size = (self.high_param-self.low_param)*1.0/ self.num_steps
             self.values = list(self.low_param+step_size*i for i in range(0,self.num_steps))

    def __str__(self):
        string = self.name +','+ str(self.values)
        return str(string)

x = dict()
# Parameter( name, low_param, high_param, num_steps)
# num steps = 0 for constant value
#             1 for exponential increases
#             >1 for number of steps
#finetune_lr,pretraining_epochs,pretrain_lr,training_epochs,batch_size,neurons_per_layer,number_of_layers 

x[0] = Parameter("finetune_lr", 0.0005, .02, 3)
x[1] = Parameter("pretraining_epochs", 5, 20, 3)
x[2] = Parameter("pretrain_lr", 0.0005, .002, 3)
x[3] = Parameter("training_epochs", 500, 2000, 3)
x[4] = Parameter("batch_size", 1, 2, 0)
x[5] = Parameter("neurons_per_layer",50,1550,3)
x[6] = Parameter("number_of_layers",1,4,3)


array = cPickle.load(open("array.p",'rb'))

total = len(array)
print total
s = numpy.zeros((7,3))
p1 = numpy.zeros((7,3))
p2 = numpy.zeros((7,3))
p4=[]

for i in [0,1,2,3,4,5,6]:
    print x[i].name
    for j in range(len(x[i].values)):
        #print x[i].name+" = "+str(x[i].values[j])
        #print (array[:,i] == x[i].values[j]).sum()
        s[i,j]=(array[:,i] == x[i].values[j]).sum()
    for j in range(len(x[i].values)):
        percent = s[i,j]/total*100
        p1[i,j] = percent
        print str(x[i].values[j])+" = "+ str(round(percent,2))+"%"
    print " "    
#run top 10%
#talk = "python top_5p.py"
#subprocess.call(talk, shell = True)
#reload
array = cPickle.load(open("top5_array.p",'rb'))

total = len(array)
print total
st = numpy.zeros((total,3))
for i in [0,1,2,3,4,5,6]:
    print x[i].name
    for j in range(len(x[i].values)):
        #print x[i].name+" = "+str(x[i].values[j])
        #print (array[:,i] == x[i].values[j]).sum()
        s[i,j]=(array[:,i] == x[i].values[j]).sum()
    for j in range(len(x[i].values)):
        percent = s[i,j]/total*100
        p2[i,j] = percent
        print str(x[i].values[j])+" = "+ str(round(percent,2))+"%"
    print " "

#recheck
p3 = p2-p1
count = 0
for i in [0,1,2,3,4,5,6]:
    print x[i].name
    for j in range(len(x[i].values)):
        print str(x[i].values[j])+" = "+ str(round(p3[i,j],2))+"%"

percent_range = [.05,.10,.15,.20,.25,.30,.35,.40,.45,.50,.55,.6,.65,.7,.75,.8,.85,.9,.95,1]
for j in percent_range:
    talk = "python top_5p.py "+str(j)
    subprocess.call(talk, shell = True)
    array = cPickle.load(open("top5_array.p",'rb'))
    total = len(array)
    #print total
    st = numpy.zeros((total,3))
    for i in [0,1,2,3,4,5,6]:
        print x[i].name
        for j in range(len(x[i].values)):
            #print x[i].name+" = "+str(x[i].values[j])
            #print (array[:,i] == x[i].values[j]).sum()
            s[i,j]=(array[:,i] == x[i].values[j]).sum()
        for j in range(len(x[i].values)):
            percent = s[i,j]/total*100
            p2[i,j] = percent
            # print str(x[i].values[j])+" = "+ str(round(percent,2))+"%"
        #print " "

    #recheck
    p3 = p2-p1
    p4.append(p3)
    count +=count
print p4
print percent_range
    #reload

count = 0
y = []
plots = []
for i in range(0,7):
    plt.figure(i)
    for j in range(len(x[i].values)):
        count = 0
        y =[]
        for p in percent_range:
            y.append(p4[count][i,j])
            count = count+1
            print len(percent_range)
            print len(y)
            print y
        x1 = plt.plot(percent_range,y,label = str(x[i].values[j]))
    #plt.scatter(percent_range,y)
    plt.title(x[i].name)
    plt.xlabel("Top (%)")
    plt.ylabel("Change from original (%)")
    plt.legend(loc='upper right', shadow=False,ncol=1)
    plt.savefig("../plots/percent_changes"+str(x[i].name)+".png")

