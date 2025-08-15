import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
from ismember import ismember
import math
import time,os
import numba
import tensorflow as tf
import cProfile,pstats,io
from pstats import SortKey

@numba.jit
def matrix_multiplication_numba(A,B):
	return np.dot(A,B)
	
def splitASTENS(astens):
	num=len(astens)
	astensLen=np.zeros(num).astype(int)
	nodeTypeVectors=np.zeros((num,200))
	i=0
	while i<num:
		vector=np.array(astens[i]).astype(int)
		astensLen[i]=len(vector)
		structCoding=np.trunc(vector/1000)
		typeCoding=abs(vector-structCoding*1000)
		structCoding=structCoding.astype(int)
		typeCoding=typeCoding.astype(int)
		unique,counts=np.unique(typeCoding,return_counts=True)
		unique=unique.astype(int)
		counts=counts.astype(int)
		nodeTypeVectors[i,unique]=counts
		i=i+1
	nodeTypeVectors=preprocessing.normalize(nodeTypeVectors,norm='l2')
	return nodeTypeVectors,astensLen

def getFunctionsID(filenames):
	functionsID=filenames.map(lambda x:x.split('.')[0])
	functionsID=functionsID.astype(int)
	functionsID=np.array(functionsID)
	return functionsID

def filter(AstFP_A,AstFP_B,SrcFP_A,SrcFP_B,NV_A,NV_B,FP_Sim_Threshold,VON_Sim_Threshold):
	AstFP_Sim=matrix_multiplication_numba(AstFP_A,AstFP_B)
	SrcFP_Sim=0
	VON_Sim=0
	label=0
	if(AstFP_Sim>=FP_Sim_Threshold):
		SrcFP_Sim=matrix_multiplication_numba(SrcFP_A,SrcFP_B)
		if(SrcFP_Sim>=FP_Sim_Threshold):
			VON_Sim=matrix_multiplication_numba(NV_A,NV_B)
			if(VON_Sim>=VON_Sim_Threshold):
				label=1
	return label,AstFP_Sim,SrcFP_Sim,VON_Sim
	
def batchFilter(AstFP_A,AstFP_Bs,SrcFP_A,SrcFP_Bs,NV_A,NV_Bs,idx,FP_Sim_Threshold,VON_Sim_Threshold):
	AstFP_Sim=tf.matmul(AstFP_A,tf.transpose(AstFP_Bs))
	AstFP_Sim=AstFP_Sim.numpy()[0]
	idxAstFP=idx[AstFP_Sim>=FP_Sim_Threshold]
	SrcFP_Bs=tf.gather(SrcFP_Bs,idxAstFP)
	SrcFP_Sim=tf.matmul(SrcFP_A,tf.transpose(SrcFP_Bs))
	SrcFP_Sim=SrcFP_Sim.numpy()[0]
	idxSrcFP=idxAstFP[SrcFP_Sim>=FP_Sim_Threshold]
	NV_Bs=tf.gather(NV_Bs,idxSrcFP)
	VON_Sim=tf.matmul(NV_A,tf.transpose(NV_Bs))
	VON_Sim=VON_Sim.numpy()[0]
	idx=idxSrcFP[VON_Sim>=VON_Sim_Threshold]
	idx=idx.reshape((len(idx),1))
	return idx

def TreeSeqCC_filtering(functionsID,predictionVectors_AstFP,predictionVectors_SrcFP,nodeTypeVectors,FP_Sim_Threshold,VON_Sim_Threshold):
	predictionVectors_AstFP=tf.convert_to_tensor(predictionVectors_AstFP,tf.float32,name='t')
	predictionVectors_SrcFP=tf.convert_to_tensor(predictionVectors_SrcFP,tf.float32,name='t')
	nodeTypeVectors=tf.convert_to_tensor(nodeTypeVectors,tf.float32,name='t')
	reports=[]
	counts=0
	output=True
	num=len(functionsID)
	index=np.arange(num)
	batchSize=500000
	for i in range(0,num-1):
		AstFP_A=predictionVectors_AstFP[i,:]
		AstFP_A=tf.reshape(AstFP_A,[1,-1])
		SrcFP_A=predictionVectors_SrcFP[i,:]
		SrcFP_A=tf.reshape(SrcFP_A,[1,-1])
		NV_A=nodeTypeVectors[i,:]
		NV_A=tf.reshape(NV_A,[1,-1])
		comparingIDs=index[slice(i+1,num)]
		comparingNum=num-i
		roundNum=math.floor(comparingNum/batchSize)
		for j in range(0,roundNum+1):
			if(j<roundNum):
				loc=slice(j*batchSize,(j+1)*batchSize)
			else:
				loc=slice(j*batchSize,comparingNum)
			idxB=comparingIDs[loc]
			AstFP_Bs=tf.gather(predictionVectors_AstFP,idxB)
			SrcFP_Bs=tf.gather(predictionVectors_SrcFP,idxB)
			NV_Bs=tf.gather(nodeTypeVectors,idxB)
			reportidx=batchFilter(AstFP_A,AstFP_Bs,SrcFP_A,SrcFP_Bs,NV_A,NV_Bs,index[slice(0,num-i-1)],FP_Sim_Threshold,VON_Sim_Threshold)
			reportB=idxB[reportidx]
			reportNum=len(reportB)
			if(reportNum):
				counts=counts+reportNum
				if(output):
					for k in range(0,reportNum):
						reports.append([functionsID[i],functionsID[reportB[k][0]]])
	return counts,reports
	
def TreeSeqCC_filtering_by_Samples(functionsID,predictionVectors_AstFP,predictionVectors_SrcFP,nodeTypeVectors,samples,FP_Sim_Threshold,VON_Sim_Threshold):
	Iloc,idx1=ismember(samples['id1'].values,functionsID)
	samples=samples[Iloc]
	Iloc,idx2=ismember(samples['id2'].values,functionsID)
	samples=samples[Iloc]
	Iloc,idx1=ismember(samples['id1'].values,functionsID)
	Iloc,idx2=ismember(samples['id2'].values,functionsID)
	num=len(samples)
	reports=np.zeros(num).astype(int)
	AstFP_Sim=np.zeros(num)
	SrcFP_Sim=np.zeros(num)
	VON_Sim=np.zeros(num)
	for i in range(0,num):
		A=idx1[i]
		B=idx2[i]
		AstFP_A=predictionVectors_AstFP[A,:]
		SrcFP_A=predictionVectors_SrcFP[A,:]
		NV_A=nodeTypeVectors[A,:]
		AstFP_B=predictionVectors_AstFP[B,:]
		SrcFP_B=predictionVectors_SrcFP[B,:]
		NV_B=nodeTypeVectors[B,:]
		reports[i],AstFP_Sim[i],SrcFP_Sim[i],VON_Sim[i]=filter(AstFP_A,AstFP_B,SrcFP_A,SrcFP_B,NV_A,NV_B,FP_Sim_Threshold,VON_Sim_Threshold)
	samples.insert(3,'report',reports)
	samples.insert(4,'AstFP_Sim',AstFP_Sim)
	samples.insert(5,'SrcFP_Sim',SrcFP_Sim)
	samples.insert(6,'VON_Sim',VON_Sim)
	return samples

def getUniIndex(functionsID,filenames):
	filenames=getFunctionsID(filenames)
	idx,loc=ismember(filenames,functionsID)
	return idx,loc

def prepareData():
	minLength=5
	maxLength=5000
	data=pd.read_pickle("data/BCB_Full-ASTENS.pkl")
	currentTime=time.strftime("%m-%d_%H_%M_%S",time.localtime())
	print("Preparing Data at: ",currentTime)
	filenames=data['filename']
	functionsID=getFunctionsID(filenames)
	nodeTypeVectors,astensLen=splitASTENS(data['astens'])
	idx=np.logical_and(astensLen>=minLength,astensLen<=maxLength)
	functionsID=functionsID[idx]
	nodeTypeVectors=nodeTypeVectors[idx,:]
	nodeTypeVectors=np.array(nodeTypeVectors)
	predictions_AstFP=pd.read_pickle("functionalityPredictions/ASTSDL/ASTSDL-BiLSTM.pkl")
	predictionVectors_AstFP=np.zeros((len(functionsID),predictions_AstFP.shape[1]-1))
	idx,loc=getUniIndex(functionsID,predictions_AstFP['filename'])
	predictionVectors_AstFP[loc,:]=predictions_AstFP.iloc[idx,1:]
	predictionVectors_AstFP=preprocessing.normalize(predictionVectors_AstFP,norm='l2')
	predictionVectors_AstFP=np.array(predictionVectors_AstFP)
	predictions_SrcFP=pd.read_pickle("functionalityPredictions/CodeSDL/CodeSDL-BiLSTM_basic_SkipGram.pkl")
	predictionVectors_SrcFP=np.zeros((len(functionsID),predictions_SrcFP.shape[1]-1))
	idx,loc=getUniIndex(functionsID,predictions_SrcFP['filename'])
	predictionVectors_SrcFP[loc,:]=predictions_SrcFP.iloc[idx,1:]
	predictionVectors_SrcFP=preprocessing.normalize(predictionVectors_SrcFP,norm='l2')
	predictionVectors_SrcFP=np.array(predictionVectors_SrcFP)
	currentTime=time.strftime("%m-%d_%H_%M_%S",time.localtime())
	print("--End at: ",currentTime)
	return functionsID,predictionVectors_AstFP,predictionVectors_SrcFP,nodeTypeVectors

def reduceData(functionsID,predictionVectors_AstFP,predictionVectors_SrcFP,nodeTypeVectors):
	funcList=pd.read_table("data/labeledSamples.txt",header=None,sep=' ')
	reduceFuncID=funcList[0].values
	idx,loc=ismember(functionsID,reduceFuncID)
	functionsID=functionsID[idx]
	predictionVectors_AstFP=predictionVectors_AstFP[idx]
	predictionVectors_SrcFP=predictionVectors_SrcFP[idx]
	nodeTypeVectors=nodeTypeVectors[idx]
	return functionsID,predictionVectors_AstFP,predictionVectors_SrcFP,nodeTypeVectors

def CloneDetection(detectionType):
	functionsID,predictionVectors_AstFP,predictionVectors_SrcFP,nodeTypeVectors=prepareData()
	FP_Sim_Threshold=0
	VON_Sim_Threshold=0
	if(detectionType=="Sample-based"):
		samples=pd.read_table("samples/BCB-samples.txt",sep=' ')
		currentTime=time.strftime("%m-%d_%H_%M_%S",time.localtime())
		print("Detecting with samples list at: ",currentTime)
		reports=TreeSeqCC_filtering_by_Samples(functionsID,predictionVectors_AstFP,predictionVectors_SrcFP,nodeTypeVectors,samples,FP_Sim_Threshold,VON_Sim_Threshold)
		currentTime=time.strftime("%m-%d_%H_%M_%S",time.localtime())
		print("--End at: ",currentTime)
		reports.to_csv("results/reports-pairwise.txt",sep='\t', index=False)
	if(detectionType=="Free"):
		functionsID,predictionVectors_AstFP,predictionVectors_SrcFP,nodeTypeVectors=reduceData(functionsID,predictionVectors_AstFP,predictionVectors_SrcFP,nodeTypeVectors)
		currentTime=time.strftime("%m-%d_%H_%M_%S",time.localtime())
		print("Detecting code clones at: ",currentTime)
		counts,reports=TreeSeqCC_filtering(functionsID,predictionVectors_AstFP,predictionVectors_SrcFP,nodeTypeVectors,FP_Sim_Threshold,VON_Sim_Threshold)
		currentTime=time.strftime("%m-%d_%H_%M_%S",time.localtime())
		print("--End at: ",currentTime)
		file=open("results/reports.txt",'w')
		for line in reports:
			file.write(str(line)+'\n')
		file.close()
		print("Number of reports: ",counts)

if __name__=="__main__":
    #pr=cProfile.Profile()
    #pr.enable()
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'	
    CloneDetection("Sample-based")
    #CloneDetection("Free")
    #pr.disable()
    #s=io.StringIO()
    #sortby=SortKey.CUMULATIVE
    #ps=pstats.Stats(pr,stream=s).sort_stats(sortby)
    #ps.print_stats()
    #print(s.getvalue())
