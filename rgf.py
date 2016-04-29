"""
__Author__: Triskelion <info@mlwave.com>

A small toy Python wrapper for Regularized Greedy Forests

Limitation of Liability. In no event shall Author be liable to you or any 
party related to you for any indirect, incidental, consequential, special, 
exemplary, or punitive damages or lost profits, even if Author has been advised 
of the possibility of such damages. In any event, Author's total aggregate 
liability to you for all damages of every kind and type (regardless of whether 
based in contract or tort) shall not exceed the purchase price of the product.
"""

## Dependencies ###############################################
import os
import subprocess
from glob import glob
import numpy as np
import pandas as pd

## Edit this ##################################################

#Location of the RGF executable
loc_exec = "c:\\python64\\rgf\\bin\\rgf.exe"

#Location of a temporary directory (has to exist)
loc_temp = "rgf\\temp3"

class RegularizedGreedyForestClassifier:
	def __init__(self, verbose=0, max_leaf=500, test_interval=100, loc_exec=loc_exec, loc_temp=loc_temp, algorithm="RGF", loss="LS", l2="1", prefix="model"):
		self.verbose = verbose
		self.max_leaf = max_leaf
		self.algorithm = algorithm
		self.loss = loss
		self.test_interval = test_interval
		self.prefix = prefix
		self.l2 = l2
		if os.path.exists(loc_exec):
			self.loc_exec = loc_exec
		else:
			print("Warning: Location to RGF executable not found or not correctly set:\n\t%s\n"%loc_exec)
		if os.path.exists(loc_temp):
			self.loc_temp = loc_temp
		else:
			print("Warning: Location to a temporary directory does not exist:\n\t%s\n"%loc_temp)
	
	#Fitting/training the model to target variables
	def fit(self,X,y):
		#Store the train set into RGF format
		np.savetxt(os.path.join(loc_temp, "train.data.x"), X, delimiter=' ', fmt="%s")
		y = ["+1" if f == "1" else "-1" for f in map(str, list(y))]
		#Store the targets into RGF format
		np.savetxt(os.path.join(loc_temp, "train.data.y"), y, delimiter=' ', fmt="%s")
		
		#format train command
		params = []
		if self.verbose > 0:
			params.append("Verbose")
		params.append("train_x_fn=%s"%os.path.join(loc_temp, "train.data.x"))
		params.append("train_y_fn=%s"%os.path.join(loc_temp, "train.data.y"))
		params.append("algorithm=%s"%self.algorithm)
		params.append("loss=%s"%self.loss)
		params.append("max_leaf_forest=%s"%self.max_leaf)
		params.append("test_interval=%s"%self.test_interval)
		params.append("reg_L2=%s"%self.l2)
		params.append("model_fn_prefix=%s"%os.path.join(loc_temp, self.prefix))
		
		cmd = "%s train %s 2>&1"%(self.loc_exec,",".join(params))
		
		#train
		output = subprocess.Popen(cmd.split(),stdout=subprocess.PIPE,shell=True).communicate()
		
		for k in output:
			print k
			
	def predict_proba(self,X, clean=True):
		#Store the test set into RGF format
		np.savetxt(os.path.join(loc_temp, "test.data.x"), X, delimiter=' ', fmt="%s")
	
		#Find latest model location
		model_glob = self.loc_temp + os.sep + self.prefix + "*"
		latest_model_loc = sorted(glob(model_glob),reverse=True)[0]
		
		#Format test command
		params = []
		params.append("test_x_fn=%s"%os.path.join(loc_temp, "test.data.x"))
		params.append("prediction_fn=%s"%os.path.join(loc_temp, "predictions.txt"))
		params.append("model_fn=%s"%latest_model_loc)
		cmd = "%s predict %s 2>&1"%(self.loc_exec,",".join(params))
		
		output = subprocess.Popen(cmd.split(),stdout=subprocess.PIPE,shell=True).communicate()
		
		for k in output:
			print k
		
		y_pred = np.array([sigmoid(x) for x in np.loadtxt(os.path.join(loc_temp, "predictions.txt"))])
		y_pred = np.array([[1-x, x] for x in y_pred])
		#Clean temp directory
		if clean:
			model_glob = self.loc_temp + os.sep + "*"
			
			for fn in glob(model_glob):
				if "predictions.txt" in fn or "model-" in fn or "train.data." in fn or "test.data." in fn:
					os.remove(fn)
			
		return y_pred
		
	def get_params(self):
		params = {}
		params["verbose"] = self.verbose
		params["max_leaf"] = self.max_leaf
		params["algorithm"] = self.algorithm
		params["loss"] = self.loss
		params["test_interval"] = self.test_interval
		params["prefix"] = self.prefix
		params["l2"] = self.l2
		return params
		
class RegularizedGreedyForestRegressor:
	def __init__(self, verbose=0, max_leaf=500, test_interval=100, loc_exec=loc_exec, loc_temp=loc_temp, algorithm="RGF", loss="LS", l2="1", prefix="model"):
		self.verbose = verbose
		self.max_leaf = max_leaf
		self.algorithm = algorithm
		self.loss = loss
		self.test_interval = test_interval
		self.prefix = prefix
		self.l2 = l2
		if os.path.exists(loc_exec):
			self.loc_exec = loc_exec
		else:
			print("Warning: Location to RGF executable not found or not correctly set:\n\t%s\n"%loc_exec)
		if os.path.exists(loc_temp):
			self.loc_temp = loc_temp
		else:
			print("Warning: Location to a temporary directory does not exist:\n\t%s\n"%loc_temp)
	
	#Fitting/training the model to target variables
	def fit(self,X,y):
		#Store the train set into RGF format
		np.savetxt(os.path.join(loc_temp, "train.data.x"), X, delimiter=' ', fmt="%s")
		#Store the targets into RGF format
		np.savetxt(os.path.join(loc_temp, "train.data.y"), y, delimiter=' ', fmt="%s")
		
		#format train command
		params = []
		if self.verbose > 0:
			params.append("Verbose")
		params.append("NormalizeTarget")
		params.append("train_x_fn=%s"%os.path.join(loc_temp, "train.data.x"))
		params.append("train_y_fn=%s"%os.path.join(loc_temp, "train.data.y"))
		params.append("algorithm=%s"%self.algorithm)
		params.append("loss=%s"%self.loss)
		params.append("max_leaf_forest=%s"%self.max_leaf)
		params.append("test_interval=%s"%self.test_interval)
		params.append("reg_L2=%s"%self.l2)
		params.append("model_fn_prefix=%s"%os.path.join(loc_temp, self.prefix))
		
		cmd = "%s train %s 2>&1"%(self.loc_exec,",".join(params))
		
		#train
		output = subprocess.Popen(cmd.split(),stdout=subprocess.PIPE,shell=True).communicate()
		
		for k in output:
			print k
			
	def predict(self,X, clean=True):
		#Store the test set into RGF format
		np.savetxt(os.path.join(loc_temp, "test.data.x"), X, delimiter=' ', fmt="%s")
	
		#Find latest model location
		model_glob = self.loc_temp + os.sep + self.prefix + "*"
		latest_model_loc = sorted(glob(model_glob),reverse=True)[0]
		
		#Format test command
		params = []
		params.append("test_x_fn=%s"%os.path.join(loc_temp, "test.data.x"))
		params.append("prediction_fn=%s"%os.path.join(loc_temp, "predictions.txt"))
		params.append("model_fn=%s"%latest_model_loc)
		cmd = "%s predict %s"%(self.loc_exec,",".join(params)) # 2>&1
		
		output = subprocess.Popen(cmd.split(),stdout=subprocess.PIPE,shell=True).communicate()
		
		for k in output:
			print k
		
		y_pred = np.loadtxt(os.path.join(loc_temp, "predictions.txt"))
		
		#Clean temp directory
		if clean:
			model_glob = self.loc_temp + os.sep + "*"
			
			for fn in glob(model_glob):
				if "predictions.txt" in fn or "model-" in fn or "train.data." in fn or "test.data." in fn:
					os.remove(fn)
		print X.shape
		return y_pred
