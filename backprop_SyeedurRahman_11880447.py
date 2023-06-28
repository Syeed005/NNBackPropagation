import numpy as np

class NeuralNetwork(object):
		def __init__(self,acType, biasType, initial_W):
			#parameters
			self.inputSize = 2
			self.outputSize = 2
			self.hiddenSize = 2

			#input & output
			self.I1 = 0.45
			self.I2 = 0.1
			self.T1 = .99
			self.T2 = .01
			if biasType == "givenBias":
				self.B = 1
			else:
				self.B = 0
			self.LR = 0.08
			#weights
			self.W1 = initial_W[0]
			self.W2 = initial_W[1]
			self.W3 = initial_W[2]
			self.W4 = initial_W[3]
			self.W5 = initial_W[4]
			self.W6 = initial_W[5]
			self.W7 = initial_W[6]
			self.W8 = initial_W[7]
			self.W9 = initial_W[8]
			self.W10 = initial_W[9]
			self.W11 = initial_W[10]
			self.W12 = initial_W[11]
				
		def feedForward(self, acType):
			#forward propogation through the network
			self.InH1 = (self.I1 * self.W1) + (self.I2 * self.W4) + (self.B * self.W9)
			self.InH2 = (self.I1 * self.W3) + (self.I2 * self.W2) + (self.B * self.W10)
			
			if acType == "sigmoid": 
				self.OutH1 = self.sigmoid(self.InH1)
				self.OutH2 = self.sigmoid(self.InH2)
			elif acType == "relu":
				self.OutH1 = self.relu(self.InH1)
				self.OutH2 = self.relu(self.InH2)
			else:
				self.OutH1 = 1
				self.OutH2 = 1

			self.InO1 = (self.OutH1 * self.W5) + (self.OutH2 * self.W7) + (self.B * self.W11)
			self.InO2 = (self.OutH1 * self.W6) + (self.OutH2 * self.W8) + (self.B * self.W12)

			if acType == "sigmoid": 
				self.OutO1 = self.sigmoid(self.InO1)
				self.OutO2 = self.sigmoid(self.InO2)
			elif acType == "relu":
				self.OutO1 = self.relu(self.InO1)
				self.OutO2 = self.relu(self.InO2)
			else:
				self.OutO1 = 1
				self.OutO2 = 1

		def backward(self, acType):
			#backward propogate through the network
			self.output_error1 = self.OutO1 - self.T1
			self.output_error2 = self.OutO2 - self.T2
			if acType == "sigmoid":
				self.output_delta_O1 = self.output_error1 * self.sigmoid(self.OutO1,True)
				self.output_delta_O2 = self.output_error2 * self.sigmoid(self.OutO2,True)
				self.output_delta_O1H1 = self.output_delta_O1 * self.W5 * self.sigmoid(self.OutH1,True)
				self.output_delta_O1H2 = self.output_delta_O1 * self.W7 * self.sigmoid(self.OutH2,True)				
				self.output_delta_O2H1 = self.output_delta_O2 * self.W6 * self.sigmoid(self.OutH1,True)				
				self.output_delta_O2H2 = self.output_delta_O2 * self.W8 * self.sigmoid(self.OutH2,True)					
			else:
				self.output_delta_O1 = self.output_error1 * self.relu(self.OutO1,True)
				self.output_delta_O2 = self.output_error2 * self.relu(self.OutO2,True)
				self.output_delta_O1H1 = self.output_delta_O1 * self.W5 * self.relu(self.OutH1,True)
				self.output_delta_O1H2 = self.output_delta_O1 * self.W7 * self.relu(self.OutH2,True)				
				self.output_delta_O2H1 = self.output_delta_O2 * self.W6 * self.relu(self.OutH1,True)				
				self.output_delta_O2H2 = self.output_delta_O2 * self.W8 * self.relu(self.OutH2,True)		

			self.output_delta_W5 = self.output_delta_O1 * self.OutH1
			self.output_delta_W7 = self.output_delta_O1 * self.OutH2				
			self.output_delta_W11 = self.output_delta_O1 * self.B
			self.output_delta_W8 = self.output_delta_O2 * self.OutH2
			self.output_delta_W6 = self.output_delta_O2 * self.OutH1
			self.output_delta_W12 = self.output_delta_O2 * self.B
			self.output_delta_W1 = (self.output_delta_O1H1 * self.I1) + (self.output_delta_O2H1 * self.I1) 
			self.output_delta_W4 = (self.output_delta_O1H1 * self.I2) + (self.output_delta_O2H1 * self.I2) 
			self.output_delta_W9 = (self.output_delta_O1H1 * self.B) + (self.output_delta_O2H1 * self.B) 
			self.output_delta_W2 = (self.output_delta_O2H2 * self.I2) + (self.output_delta_O1H2 * self.I2) 
			self.output_delta_W3 = (self.output_delta_O2H2 * self.I1) + (self.output_delta_O1H2 * self.I1) 
			self.output_delta_W10 = (self.output_delta_O2H2 * self.B) + (self.output_delta_O1H2 * self.B) 
			
			self.W1 -= self.LR * self.output_delta_W1
			self.W2 -= self.LR * self.output_delta_W2
			self.W3 -= self.LR * self.output_delta_W3
			self.W4 -= self.LR * self.output_delta_W4
			self.W5 -= self.LR * self.output_delta_W5				
			self.W6 -= self.LR * self.output_delta_W6
			self.W7 -= self.LR * self.output_delta_W7
			self.W8 -= self.LR * self.output_delta_W8
			self.W9 -= self.LR * self.output_delta_W9				
			self.W10 -= self.LR * self.output_delta_W10
			self.W11 -= self.LR * self.output_delta_W11
			self.W12 -= self.LR * self.output_delta_W12

			return [self.W1, self.W2, self.W3, self.W4, self.W5, self.W6, self.W7, self.W8, self.W9, self.W10, self.W11, self.W12]	

		def calError(self):
			try:
				Err = .5*pow((self.T1-self.OutO1),2)+.5*pow((self.T2-self.OutO2),2)
				return Err
			except OverflowError as e:
				print("Overflow error happened")

		def sigmoid(self, s, deriv=False):
			if (deriv == True):
				return s * (1 - s)
			return 1/(1 + np.exp(-s))
		
		def relu(self, s, deriv=False):
			if (deriv == True):
				return 1 if s > 0 else 0
			return max(0.0, s)
		
def RunAnalysis(acType, biasType,initial_W, targetErr):

	NN = NeuralNetwork(acType, biasType, initial_W)
	count = 0
				
	while True:
		NN.feedForward(acType)
		Error = NN.calError()
		count = count + 1
		if Error == None:
			break
		if (Error < targetErr):
			break
		else:
			updated_W = NN.backward(acType)

	p1 = np.array(initial_W)
	p2 = np.array(updated_W)
	distance = np.linalg.norm(p2-p1)
	
	print(f"{acType} ErrorData: {targetErr} ; Iteration: {count} ; Error: {Error} ; Euclidean Distance: {distance}")
	#print(f"{acType};{targetErr};{count};{distance};{Error}")

def RunProgram(acType, biasType, weightType):
	ErList = [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
	if weightType == "givenWeight":
		initial_W = [0.1, 0.2, 0.25, 0.3, 0.8, 0.75, 0.7, 0.8, 0.6, 0.65, 0.8, 0.2]
	else:
			initial_W = np.random.rand(12)
	print(initial_W)
	for Er in ErList:
		RunAnalysis(acType, biasType, initial_W, Er)


#start of the program
#use keyword "sigmoid" or "relu "for difernt activation funtion
#use keyword "givenBias" or "noBias" for different Bias value
#use keyword "givenWeight" or "randomWeight" for fixed or random weight value

RunProgram("relu","givenBias","givenWeight")