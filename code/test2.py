import subprocess

for i in range(3):
	talk = 'qsub test.pbs -v input1='+str(i)+',input2='+str(i+1)
	subprocess.call(talk,shell=True)
	print talk

