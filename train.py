from __future__ import division
import os
import time
import copy
import torch
import operator
import torchvision
import pandas as pd
import numpy as np
from skimage import io, transform
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler as lsched
from torch.autograd import Variable
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import torch.nn.init as init
import json
# from data_conv import QuestionDataset, GetDataLoader, sort_batch
import warnings
warnings.filterwarnings("ignore")

EMBEDDING_DIM = 300
HIDDEN_DIM = 512
root_data_dir = ''
K =2000

def getDataLoader(input):
	file = open(root_data_dir+input,'r')
	datastore = json.load(file)
	return datastore

def do_once():
	freq_ans ={}
	data = getDataLoader('v2_mscoco_val2014_annotations.json')
	for items in data['annotations']:
		answer = items['multiple_choice_answer'].lower()
		if(answer in freq_ans):
			freq_ans[answer] = freq_ans[answer] + 1
		else:
			freq_ans[answer] = 1

	data = getDataLoader('v2_mscoco_train2014_annotations.json')
	
	for items in data['annotations']:
		answer = items['multiple_choice_answer'].lower()
		if(answer in freq_ans):
			freq_ans[answer] = freq_ans[answer] + 1
		else:
			freq_ans[answer] = 1

	answer_ids ={}
	count =0
	sorted_freq_ans = sorted(freq_ans.items(), key=operator.itemgetter(1),reverse=True)
	sorted_freq_ans = sorted_freq_ans[0:K]
	for key, value in sorted_freq_ans:
		answer_ids[key] = count
		count = count + 1

	maxi =0
	train_data = []
	Qdata = getDataLoader('v2_OpenEnded_mscoco_train2014_questions.json')
	for items in Qdata['questions']:
		if(maxi<len(items['question'].split())):
			maxi = len(items['question'].split())
		train_data.append(items['question'].lower().split())

	val_data = []
	Qdata = getDataLoader('v2_OpenEnded_mscoco_val2014_questions.json')
	for items in Qdata['questions']:
		if(maxi<len(items['question'].split())):
			maxi = len(items['question'].split())
		val_data.append(items['question'].lower().split())

	word_to_ix = {}

	for line in train_data:
		for word in line:
			if word not in word_to_ix:
				word_to_ix[word] = len(word_to_ix)+1

	for line in val_data:
		for word in line:
			if word not in word_to_ix:
				word_to_ix[word] = len(word_to_ix)+1
	return answer_ids,word_to_ix,maxi


def prepare_sequence(seq, to_ix):
	idxs =[]
	for w in seq:
		if w not in to_ix:
			idxs.append(0)
		else:
			idxs.append(to_ix[w])
	return np.array(idxs)


class QuestionDataset(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self,image_mapping,answer_ids,word_to_ix,maxi, answer_json,question_json, transform=None):
		#image_dir will contain the whole path before 0's
		dataload = getDataLoader(answer_json)
		self.data =[]
		self.word_to_ix = word_to_ix
		self.maxi = maxi
		for items in dataload['annotations']:
			tmp =[]
			tmp.append(items['image_id'])
			tmp.append(items['question_id'])
			answer = items['multiple_choice_answer'].lower()
			if(answer in answer_ids):
				tmp.append(answer_ids[answer])
			else:
				tmp.append(K)
			self.data.append(tmp)
		self.Questions ={}
		Qdata = getDataLoader(question_json)
		for items in Qdata['questions']:
			self.Questions[items['question_id']] = items['question'].lower()
		self.image_mapping = image_mapping
		self.transform = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		image = np.array(self.image_mapping[str(self.data[idx][0])])
		question = prepare_sequence(self.Questions[self.data[idx][1]].split(),self.word_to_ix)
		question = np.pad(question, (0, self.maxi-len(question)), 'constant')
		sample = {'image': image, 'question':question ,
		'answer': np.array([self.data[idx][2]])}

		if self.transform:
			sample = self.transform(sample)

		return sample

class Rescale(object):
	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size

	def __call__(self, sample):
		image, question, answer = sample['image'], sample['question'],sample['answer']

		h, w = image.shape[:2]
		if isinstance(self.output_size, int):
			if h > w:
				new_h, new_w = self.output_size * h / w, self.output_size
			else:
				new_h, new_w = self.output_size, self.output_size * w / h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		img = transform.resize(image, (new_h, new_w))

		return {'image': img, 'question': question,'answer':answer}

class RandomCrop(object):
	"""Crop randomly the image in a sample.

	Args:
		output_size (tuple or int): Desired output size. If int, square crop
			is made.
	"""

	def __init__(self, output_size):
		assert isinstance(output_size, int)
		self.output_size = (output_size, output_size)


	def __call__(self, sample):
		image, question, answer = sample['image'], sample['question'],sample['answer']

		h, w = image.shape[:2]
		new_h, new_w = self.output_size

		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)

		image = image[top: top + new_h,
					  left: left + new_w]
		return {'image': image, 'question': question,'answer': answer}

class CenterCrop(object):
	"""Crop randomly the image in a sample.

	Args:
		output_size (tuple or int): Desired output size. If int, square crop
			is made.
	"""

	def __init__(self, output_size):
		assert isinstance(output_size, int)
		self.output_size = (output_size, output_size)


	def __call__(self, sample):
		image, question, answer = sample['image'], sample['question'],sample['answer']

		h, w = image.shape[:2]
		new_h, new_w = self.output_size

		top = (h - new_h)//2
		left = (w - new_w)//2

		image = image[top: top + new_h,
					  left: left + new_w]
		return {'image': image, 'question': question,'answer': answer}


class ToTensor(object):
	def __call__(self, sample):
		image, question,answer = sample['image'], sample['question'], sample['answer']
		
		return {'image': torch.from_numpy(image),
				'question': torch.from_numpy(question),'answer': torch.from_numpy(answer)}

class Normalize(object):
	def __init__(self,mean,std):
		self.mean = mean
		self.std = std

	def __call__(self, sample):
		image, question,answer = sample['image'], sample['question'],sample['answer']

		norm = transforms.Normalize(mean=self.mean,
							 std=self.std)
		image = norm(image)
		

		return {'image': image,
				'question':question,
				'answer':answer}

data_transform = transforms.Compose([
		ToTensor()
	])

data_transform1 = transforms.Compose([
		ToTensor()
	])

# train_dataset = QuestionDataset(answer_json = 'v2_mscoco_train2014_annotations.json',
# 	question_json ='v2_OpenEnded_mscoco_train2014_questions.json',
# 	image_dir=root_data_dir+'train2014/COCO_train2014_', transform=data_transform)
# val_dataset = QuestionDataset(answer_json = 'v2_mscoco_val2014_annotations.json',
# 	question_json ='v2_OpenEnded_mscoco_val2014_questions.json',
# 	image_dir=root_data_dir+'val2014/COCO_val2014_', transform=data_transform1)

# trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=128,shuffle =False,num_workers=4)


def sort_batch(data, seq_len):
	sorted_idx = sorted(enumerate(seq_len), key=lambda x:x[1],reverse = True)
	sorted_seq_len = [i[1] for i in sorted_idx]
	sorted_idx = [i[0] for i in sorted_idx]
	sorted_idx = torch.LongTensor(sorted_idx)
	# sorted_seq_len = seq_len[sorted_idx]
	data['question'] = data['question'][sorted_idx]
	data['image'] = data['image'][sorted_idx]
	data['answer'] = data['answer'][sorted_idx]
	return data, sorted_seq_len

def GetDataLoader(image_mapping,answer_ids,word_to_ix,maxi):
	train_dataset = QuestionDataset(image_mapping = image_mapping,answer_ids = answer_ids,word_to_ix = word_to_ix,maxi = maxi,
	answer_json = 'v2_mscoco_train2014_annotations.json',
	question_json ='v2_OpenEnded_mscoco_train2014_questions.json', transform=data_transform)
	val_dataset = QuestionDataset(image_mapping = image_mapping,answer_ids = answer_ids,word_to_ix = word_to_ix,maxi = maxi,
	answer_json = 'v2_mscoco_val2014_annotations.json',
	question_json ='v2_OpenEnded_mscoco_val2014_questions.json', transform=data_transform1)

	trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=128,shuffle =True,num_workers=1)
	valloader = torch.utils.data.DataLoader(val_dataset,batch_size=128,shuffle =False,num_workers=1,drop_last=True)
	return trainloader, valloader, len(word_to_ix)




def validation(model,dataLoader):
	model.eval()
	correct = 0
	total = 0
	for i,data in enumerate(dataLoader,0):
		# val_start_time = time.time()
		seq_len =[]
		for q in data['question']:
			seq_len.append(maxi-len(q[q==0]))
		data,seq_len = sort_batch(data,seq_len)
		
		# get the inputs
		img, query, ans = data['image'].type(torch.FloatTensor), data['question'].type(torch.LongTensor), data['answer'].type(torch.LongTensor)
		# get the inputs
		# img, query, ans = data

		# wrap them in Variable
		img, query,ans = Variable(img.cuda()),Variable(query.cuda()), ans.cuda()
		outputs = model(img,query,seq_len)
		# print(outputs.data)
		
		_, predicted = torch.max(outputs.data[:,0:K-1], 1)
		# print(predicted)
		# print (predicted.size())
		total += ans.size(0)
		correct += (predicted == ans[:,0]).sum()
		# print (i, (time.time()-val_start_time))
		# if(i%10==0):
		# 	print(i)
	return (correct/total)


class LSTM_RNN(nn.Module):
	def __init__(self, embedding_dim, hidden_dim,num_hidden_layers, vocab_size):
		super(LSTM_RNN, self).__init__()
		self.hidden_dim = hidden_dim
		self.num_hidden_layers = num_hidden_layers
		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim,num_hidden_layers,batch_first = True)
		self.linear = nn.Linear(2*self.hidden_dim*self.num_hidden_layers,1024)
		# self.hidden = self.init_hidden()

	def init_hidden(self,batch_size):
		return (Variable(torch.zeros(self.num_hidden_layers,batch_size,self.hidden_dim).cuda()),
				Variable(torch.zeros(self.num_hidden_layers,batch_size,self.hidden_dim).cuda()))

	def forward(self, sentence,seq_length):
		embeds = self.word_embeddings(sentence)
		n = (embeds.size())[0]
		embeds = pack_padded_sequence(embeds,seq_length,batch_first=True)
		output = Variable(torch.zeros(1,2*self.hidden_dim*self.num_hidden_layers))
		h0 = Variable(torch.zeros(self.num_hidden_layers,n,self.hidden_dim).cuda())
		c0 = h0
		x,(hn,cn) = self.lstm(embeds,(h0,c0))
		# length = embeds.size()[1]
		# for i in range(length):
		# 	self.hidden,(hn,cn) = self.lstm((embeds[:,i,:]).view(-1,EMBEDDING_DIM), self.hidden)
		# 	if(i==length-1):
		output = hn[0,:,:]
		for i in range(self.num_hidden_layers-1):
			output = torch.cat((output,hn[i+1,:,:]),1)
		for i in range(self.num_hidden_layers):
			output = torch.cat((output,cn[i,:,:]),1)
		output = self.linear(output)
		return output

class VQA_model(nn.Module):
	def __init__(self, embedding_dim, hidden_dim,num_hidden_layers, vocab_size):
		super(VQA_model, self).__init__()
		
		
		# self.CNN =models.vgg16_bn(pretrained=False)
		# self.CNN.load_state_dict(torch.load('vgg16_bn-6c64b313.pth'))
		# for param in self.CNN.parameters():
		# 	param.requires_grad = False
		# Making the last fc layer to be finetunable
		### If I is to be taken
		# self.CNN.classifier[6] = nn.Linear(4096, 1024)	

		### If norm-I is to be taken 
		# self.CNN.classifier[6] = nn.BatchNorm1d(4096)
		# self.fc_cnn = nn.Linear(4096,1024)

		# self.CNN.classifier = nn.Sequential(
  #           self.CNN.classifier[0],
  #           self.CNN.classifier[1],
  #           self.CNN.classifier[2],
  #           self.CNN.classifier[3],
  #           self.CNN.classifier[4],
  #           self.CNN.classifier[5],
  #           # nn.BatchNorm1d(4096),
  #           nn.Linear(4096,1024),
  #           )
		# self.CNN.classifier = nn.Sequential(
  #           *(self.CNN.classifier[i] for i in range(6)))
		self.fci = nn.Sequential(
			# nn.BatchNorm1d(4096, affine=False),
			nn.Linear(4096,1024))
		self.lstm = LSTM_RNN(embedding_dim, hidden_dim,num_hidden_layers,vocab_size)
		self.MLP = nn.Sequential(
			nn.Dropout(),
			nn.Linear(1024, 1024),
			# nn.ReLU(inplace=True),
			nn.Tanh(),
			nn.Dropout(),
			nn.Linear(1024, 1024),
			# nn.ReLU(inplace=True),
			nn.Tanh(),
			nn.Linear(1024, K+1),
		)


	def forward(self, image,sentence,seq_len):
		
		# print(img_features)
		# norm = img_features.norm(p=2, dim=0)
		# img_features = img_features.div(norm.expand_as(img_features))
		# print(img_features)
		# print(self.fci.weight)
		img_features = self.fci(image)
		# print(img_features)
		## For norm-I
		# img_features = self.fc_cnn(img_features)
		# batch_size = (sentences.size())[1]
		# self.LSTM.hidden = self.LSTM.init_hidden(batch_size)
		query_features = self.lstm(sentence,seq_len)
		# print(query_features)
		# print(img_features.size())
		# print(query_features.size())
		final_emb = img_features*query_features
		output = self.MLP(final_emb)
		return output

def train(trainloader,valloader, num_hidden_layers,len_data_vocab):
	model = VQA_model(EMBEDDING_DIM, HIDDEN_DIM,num_hidden_layers,
								 len_data_vocab)
	model_optimal = VQA_model(EMBEDDING_DIM, HIDDEN_DIM,num_hidden_layers,
										 len_data_vocab)
	model = model.cuda()
	model_optimal = model_optimal.cuda()

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(),lr=0.0001,weight_decay = 0.0005)
	scheduler = lsched.ReduceLROnPlateau(optimizer,  mode='min',verbose=True, factor=0.1, patience=2)

	#stopping criteria parameters
	best_acc = 0.0
	min_delta = 1e-5
	p = 10
	epoch = 0

	train_error = []
	# for epoch in range(1):
	while epoch < p:
	# for epoch in range(p):
		val_start_time = time.time()
		model.train()	
		for i, data in enumerate(trainloader, 0):
			# val_start_time = time.time()
			seq_len =[]
			for q in data['question']:
				seq_len.append(maxi-len(q[q==0]))
			data,seq_len = sort_batch(data,seq_len)
			# get the inputs
			train_img, train_query, train_ans = data['image'].type(torch.FloatTensor), data['question'].type(torch.LongTensor), data['answer'].type(torch.LongTensor)

			# wrap them in Variable
			train_img, train_query, train_ans = Variable(train_img.cuda()),	Variable(train_query.cuda()),Variable(train_ans.cuda()) 

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			# val_start_time = time.time()
			# acc = validation(model,valloader)
			train_output = model(train_img, train_query,seq_len)
			# print(train_output)
			# total_val = time.time()- val_start_time
			# print(total_val)
			# print (train_ans.size())
			loss = criterion(train_output, train_ans[:,0])
			# val_start_time = time.time()
			loss.backward()
			# total_val = time.time()- val_start_time
			# print(total_val)
			# val_start_time = time.time()
			optimizer.step()
			# total_val = time.time()- val_start_time
			# print(total_val)
			# print (i, (time.time()-val_start_time))
			# if(i%100==0):
			# 	print(i)

		print('One epoch finished')
		val_acc = validation(model,valloader)
		# print('Val accuracy done')
		# train_acc = validation(model,trainloader)
		# val_acc = 0.8
		# train_acc = 0.6
		scheduler.step(1-val_acc)
		print (i, (time.time()-val_start_time))
		print('Accuracy of the network on the validation set: %.5f %%' %(100*val_acc))
		
		# train_error.append(train_acc)
		 

		if (val_acc - best_acc) >= min_delta:
			best_acc = val_acc
			model_optimal.load_state_dict(model.state_dict())
			torch.save(model.state_dict(),'./Models/lstm_modelADAMBatch2000.pth')
			epoch = 0
		else:
			epoch = epoch +1

	# print (train_error)
	# return model
	return model_optimal


if __name__ == '__main__':
	print "Doing Once"
	answer_ids,word_to_ix,maxi = do_once()
	print "Done"
	with open('answer_id.txt', 'w') as file:
		file.write(json.dumps(answer_ids))
	with open('word_to_ix.txt', 'w') as file:
		file.write(json.dumps(word_to_ix))
	print("data processing begin")
	with open(root_data_dir+'mapping.txt', 'r') as f:
		image_mapping = json.load(f)
	trainloader,valloader, data_vocab_length = GetDataLoader(image_mapping,answer_ids,word_to_ix,maxi)
	print ("data processes")
	model_start_time  = time.time()
	model = train(trainloader,valloader,2,data_vocab_length)
	total_train = time.time()-model_start_time

	# val_start_time = time.time()
	# acc = validation(model,valloader)
	# total_val = time.time()- val_start_time
	# print('Accuracy of the network on the validation set: %f %%' % (100*acc))


	# torch.save(model.state_dict(),'./Models/lstm_model.pth')
	print('Train time %0.7f ' % (total_train))

		