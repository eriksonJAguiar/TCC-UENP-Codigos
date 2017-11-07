class Roc():

	def __init__(self):

		self.fpr = None
		self.tpr = None
		self.auc = 0.0

	def get_fpr(self):
		return self.fpr

	def set_fpr(self,value):
		self.fpr = value

	def get_tpr(self):
		return self.tpr

	def set_tpr(self,value):
		self.tpr = value

	def get_auc(self):
		return self.auc

	def set_auc(self,value):
		self.auc = value
