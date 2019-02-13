import tensorflow as tf
import numpy as np
import gym
import sys
import os
from dqn_helper import HelperClass
from fourrooms import Fourrooms
from collections import deque
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from copy import copy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
class A2T():

	def __init__(self):

		self.goal = 20
		self.exp_type = 'expert'
		self.file_name = 'ex_new_cyclic_goal_20_dense_reward_relu_'

		self.random_seed = 20002
		self.batch_size = 32
		tf.set_random_seed(self.random_seed)

		self.episodes = 300000
		self.action_space = 4
		self.num_experts = 4
		self.past_num = 1
		self.autoencoder_len = 9

		self.input_image = tf.placeholder(tf.float32,[None,13,13,self.past_num])

		self.env = gym.make('Fourrooms-v0')
		self.goal_list = ['0','15','27','42']
		self.memory = deque(maxlen=60000)		
		self.stateQ = deque(maxlen=self.past_num)						

		self.preFillLimit = 32
		self.learning_rate = 0.0001
		self.epsilon_decay_steps = 2000000
		self.target_updation_period = 10000

		self.sess = tf.Session()
		self.update_frequency = 60
		self.helperObj = HelperClass(self.autoencoder_len,self.past_num)
		self.action_mask = tf.placeholder('float32',shape=(None,4))
		self.gamma = 0.99
		self.expert_q = np.zeros((104,self.num_experts,self.action_space))
		self.expert_q_pl = tf.placeholder(tf.float32,[None,self.num_experts,self.action_space])

		self.expert_q = np.load('../hidden_policies/q_0_4_41_45.npy')# Loading the expert policies. Right now q(s,a) of all states and actions are being stored here. 		       	

		self.state_configure = np.load('state_coordinates_4_room.npy').item()# Is used  to give the coordinates (x,y) of the cell location in the gridworld given just the cell number.
        
		self.epsilon = 1
		self.eval_steps = 100
		self.test_steps = []
		self.testenv = gym.make('Fourrooms-v0')
		
		self.build_network()		
		self.run_network()


	def generateState(self):

		state = np.zeros((13,13,self.past_num))
		for i in range(self.past_num):
			state[:,:,i] = self.stateQ[i]

		return state

	def generateFrame(self,state):

		#-----------STATE REPRESENTATION IS A GRID OF ZEROS AND THE LOCATION OF THE AGENT ALONE BEING 255.0-------#	
			
		temp = np.zeros((13,13))
		temp[self.state_configure[state]] = 255.0
		
		return temp

	def chooseEvalAction(self,state,cell):
		
		
		return np.argmax(self.sess.run(self.expert_output_list,{self.input_image:state,self.expert_q_pl:self.expert_q[cell,:,:][np.newaxis,:,:]}))


	# Used to evaluate the greedy policy to check how well the agent is performing.

	def evaluatePolicy(self):
		
		steps = 0
		for i in range(self.eval_steps):

			done = False
			state = self.testenv.reset()
			cell = state
			while not done:
		
				frame = self.generateFrame(state)[np.newaxis,:,:,np.newaxis]
				action = self.chooseEvalAction(frame,cell)
				ns,reward,done,__ = self.testenv.step(action)
				steps += 1
				state = ns
				cell = ns
		self.test_steps.append(steps)
		np.save('step_expert/less_exploration_no_dense_reward_entropy_relu_'+str(self.goal)+'.npy',self.test_steps)

	def baseNetwork(self,name):

		conv1 = tf.nn.relu(tf.layers.conv2d(inputs=self.input_image,filters=4,kernel_size=[2,2],strides=(1,1),use_bias=True,
			kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.zeros_initializer(),name=name+'c1'))

		
		conv2 = tf.nn.relu(tf.layers.conv2d(inputs=conv1,filters=3,kernel_size=[2,2],strides=(2,2),use_bias=True,
			kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.zeros_initializer(),name=name+'c2'))
		
		conv2_flat = tf.layers.flatten(conv2,name=name+'f1')
		
		dense1 = tf.nn.relu(tf.layers.dense(inputs=conv2_flat, units=16, use_bias=True,
		kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.keras.initializers.he_normal(),name=name+'d1'))
		
		dense2 = tf.nn.relu(tf.layers.dense(inputs=dense1, units=16, use_bias=True,
		kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.keras.initializers.he_normal(),name=name+'d2'))

		dense3 = tf.nn.relu(tf.layers.dense(inputs=dense2, units=8, use_bias=True,
		kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.keras.initializers.he_normal(),name=name+'d3'))

		output =tf.layers.dense(inputs=dense3, units=4,activation=None,use_bias=True,
		kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.keras.initializers.he_normal(),name=name+'o1')


		return output

	
	def attention_network(self,name):

		conv1 = tf.nn.relu(tf.layers.conv2d(inputs=self.input_image,filters=4,kernel_size=[2,2],strides=(1,1),use_bias=True,
			kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.keras.initializers.he_normal(),name=name+'c1'))
	
		conv2 = tf.nn.relu(tf.layers.conv2d(inputs=conv1,filters=3,kernel_size=[2,2],strides=(2,2),use_bias=True,
			kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.keras.initializers.he_normal(),name=name+'c2'))
		
		conv2_flat = tf.layers.flatten(conv2,name=name+'f1')
		
		dense1 = tf.nn.relu(tf.layers.dense(inputs=conv2_flat, units=16, use_bias=True,
		kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.keras.initializers.he_normal(),name=name+'d1'))
		
		
		dense2 = tf.nn.relu(tf.layers.dense(inputs=dense1, units=16, use_bias=True,
		kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.keras.initializers.he_normal(),name=name+'d2'))
		
		dense3 = tf.nn.relu(tf.layers.dense(inputs=dense2, units=8, use_bias=True,
		kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.keras.initializers.he_normal(),name=name+'d3'))
		
		output = tf.nn.softmax(tf.nn.relu(tf.layers.dense(inputs=dense3, units=self.num_experts+1,activation=None,use_bias=True,
		kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.keras.initializers.he_normal(),name=name+'o1')))

		return output

	# Used to pull a mini batch from the replay structure.

	def createBatch(self):

		count=0

		if self.past_num != 0:
			current_array = np.zeros((self.batch_size,13,13,self.past_num),dtype='float32')
			next_array = np.zeros((self.batch_size,13,13,self.past_num),dtype='float32')

		else:
			current_array = np.zeros((self.batch_size,self.autoencoder_len),dtype='float32')
			next_array = np.zeros((self.batch_size,self.autoencoder_len),dtype='float32')

		reward_array = np.zeros(self.batch_size)
		done_array = np.zeros(self.batch_size,dtype='bool')
		action_array_base = np.zeros((self.batch_size,4),dtype='uint8')
		current_cell_array = np.zeros(self.batch_size,dtype='uint8')
		next_cell_array = np.zeros(self.batch_size,dtype='uint8')
		mini_batch = random.sample(self.memory,self.batch_size)
		expert_actions_array = np.zeros((self.batch_size,self.num_experts,4))
		

		for current_state ,action_attention ,reward ,next_state ,done, current_cell,next_cell  in mini_batch:

			current_array[count] = current_state
			next_array[count] = next_state
			reward_array[count] = reward
			action_array_base[count] = action_attention
			done_array[count] = done	
			current_cell_array[count] = current_cell # Gives you the actual cell number of the agent's location eg [41].
			next_cell_array[count] = next_cell
			
			count+=1
			
		return current_array,action_array_base,reward_array,next_array,done_array,current_cell_array,next_cell_array

	def build_network(self):

		#Creating the target and online attention networks.

		self.state_weight_attention = self.attention_network('attention')		
		self.state_weight_attention_target = self.attention_network('att_tar')

		#Creating the target and online base networks.

		self.model = self.baseNetwork('base_original')
		self.target_model = self.baseNetwork('target_base')

		expert_output_list_target = [self.target_model*self.state_weight_attention_target[:,0][:,tf.newaxis]]
		expert_output_list = [self.model*self.state_weight_attention[:,0][:,tf.newaxis]]
		
		#--------------------USED TO ZERO OUT Q'S OF OTHER ACTIONS WHICH WERE NOT TAKEN-------------------#

		self.model_mask = tf.multiply(self.model,self.action_mask)
		
		#----------MULTIPLYING OUTPUT OF EACH EXPERT WITH THE CORRESPONDING WEIGHT FROM THE ATTENTION MECHANISM--------------#

		for i in range(self.num_experts):

			expert_output_list.append(self.expert_q_pl[:,i,:]*self.state_weight_attention[:,i+1][:,tf.newaxis])
			expert_output_list_target.append(self.expert_q_pl[:,i,:]*self.state_weight_attention_target[:,i+1][:,tf.newaxis])


		self.expert_output_list = tf.reduce_sum(tf.transpose(tf.convert_to_tensor(expert_output_list),[1,0,2]),1)   # The final output of the online a2t agent
		self.expert_output_list_target = tf.reduce_sum(tf.transpose(tf.convert_to_tensor(expert_output_list_target),[1,0,2]),1) # The final output of the target a2t agent.

		self.expert_output_list_mask = tf.multiply(self.expert_output_list,self.action_mask)

		attention_var = []
		target_var = []
		base_var = []
		att_tar_var = []


		#--------------------------SEPERATING VARIABLES OF THE GRAPH FOR SYNCING THE ONLINE AND TARGET MODELS--------------#

		for v in tf.trainable_variables():

			if 'attention' in v.name:
				attention_var.append(v)
			if 'target' in v.name:
				target_var.append(v)
			if 'base_original' in v.name:
				base_var.append(v)
			if 'att_tar' in v.name:
				att_tar_var.append(v)


		opt = tf.train.AdamOptimizer(self.learning_rate)
	
		#-----------THIS IS THE PLACEHOLDER WHICH GETS THE TARGET VALUES FOR THE UPDATE-------------------------#

		self.target_update = tf.placeholder('float32',shape=(None,4))
		
		#-------------------------------------------------------------------------------------------------------#

		
		#------------------ADDED REGULARIZER TO MAKE THE ATTENTION SMOOTH--------------------------------------#

		self.regularizer = tf.reduce_sum((self.state_weight_attention+1e-30)*tf.log(self.state_weight_attention+1e-30))


		#---------------BLOCK OF CODE TO CALCULATE THE LOSS DUE TO ATTENTION MECHANISM----------------------------------#

		self.attention_loss = tf.losses.huber_loss(labels=self.target_update,predictions=self.expert_output_list_mask) +0.005*self.regularizer
		self.grad_attention = opt.compute_gradients(self.attention_loss,var_list=attention_var)				
		self.optimizer_attention = opt.apply_gradients(self.grad_attention)		

		#-------------------BLOCK OF CODE TO CALCULATE THE LOSS DUE TO BASE NETWORK------------------------------------#

		self.base_loss = tf.losses.huber_loss(labels = self.target_update,predictions=self.model_mask)
		self.grad_base = opt.compute_gradients(self.base_loss,var_list=base_var)		
		self.optimizer_base = opt.apply_gradients(self.grad_base)		

		# Used for pushing values from online to target network.

		self.sync_1 = tf.group(*[v1.assign(v2) for v1, v2 in zip(target_var, base_var)])		
		self.sync_2 = tf.group(*[v1.assign(v2) for v1, v2 in zip(att_tar_var, attention_var)])		
		
		self.sess.run(tf.global_variables_initializer())

		saver_list = []

		# ---- CODE TO LOAD THE EXPERT MODELS COMMENTED OUT FOR SIMPLICITY -------#

		# for i in range(self.num_experts):

		# 	temp_var = []			

		# 	for j in tf.trainable_variables():
		# 		name = 'current_goal_'+self.goal_list[i]
		# 		if  name == j.name.split('/')[0][:-2]:
			
		# 			temp_var.append(j)
				
			
		# 	saver = tf.train.Saver(temp_var)
		# 	saver_list.append(saver)
		
		# for i in range(self.num_experts):

		# 	saver_list[i].restore(self.sess,'/home/arjun/Documents/Research/Project/Distillation/expert/DQN/phase_1/goal_'+ self.goal_list[i]+'/check')
		# 	print('loaded model for goal ',self.goal_list[i])

		# saver = tf.train.Saver(attention_var)
		# saver.restore(self.sess,'/home/arjun/Documents/Research/Project/Distillation/A2T/models_hidden/model_500/check')
		# saver = tf.train.Saver(base_var)
		# saver.restore(self.sess,'/home/arjun/Documents/Research/Project/Distillation/models_hidden/model_500/check')
		

		self.order = []
		self.order.append('b')
		for i in range(self.num_experts):
			self.order.append('h')

	def rollout(self):


		for i in range(104):

			current_cell = self.env.reset()
			frame = self.generateFrame(current_cell)
			self.stateQ.append(frame)
			state = self.generateState()
			d = False
			st = 0

			while not d:

				action = self.choose_action(state,current_cell)
				ns,r,d,__ = self.env.step(action)
				current_cell = ns
				frame = self.generateFrame(current_cell)
				self.stateQ.append(frame)
				state = self.generateState()
				st += 1

			print(' Steps taken from ', i ," = ", st)

	def plotAttention( self,env,traj, obs_dim,sequence_length, name):
		i = 0
		mat = -1*np.ones([obs_dim, obs_dim])

		for coord in env.tostate.keys():
			mat[coord[0], coord[1]] = -1

		for (s, a, r, l) in traj:

			cell = env.tocell[s]             
			mat[cell[0],cell[1]] = a

			if i == sequence_length:
				break

			i += 1           

		plt.imshow(mat)
		plt.clim(-0.1, 1.1)
		plt.colorbar()  		
		plt.savefig(name)
		plt.clf()
		
	#-------USED TO GET THE ATTENTION OF EACH EXPERT ON THE WHOLE STATE SPACE OF THE GRID WORLD. THIS IS THEN PLOTTED -------------#

	def examineAttention(self,episode_num):

		episode = np.zeros((self.num_experts+1,104,4))
		ones = np.ones((self.num_experts+1))
		for i in range(104):

			frame = self.generateFrame(i)
			self.stateQ.append(frame)
			state = self.generateState()

			attention_values = self.sess.run(self.state_weight_attention,{self.input_image:state[np.newaxis]})			
			episode[:,i,1] = attention_values[0]
			episode[:,i,0] = i*ones

		
		direc = 'attention_Visualization/'+self.exp_type+'/'+self.file_name+str(self.goal)+'/'+str(episode_num)

		if not os.path.exists(direc):

			os.makedirs(direc)

		np.save(direc+'/attention_values.npy',episode[:,:,1])

	#-------EPSILON GREEDY ACTION SELECTION USING THE OUTPUT OF THE A2T AGENT ----------------#

	def choose_action(self,state,cell):

		prob=np.random.uniform(low=0,high=1)		

		if prob < self.epsilon:
			return self.env.action_space.sample()
		else:
			return np.argmax(self.sess.run(self.expert_output_list,{self.input_image:state[np.newaxis,:,:,:],self.expert_q_pl:self.expert_q[cell,:,:][np.newaxis,:,:]}))

	def replay(self,index):
		
		current_array,action_array,reward_array,next_array,done_array,current_cell_array,next_cell_array =self.createBatch()		
		
		target_next_qvalues = self.sess.run(self.expert_output_list_target,feed_dict={self.input_image:next_array,self.action_mask:np.ones(action_array.shape),self.expert_q_pl:self.expert_q[next_cell_array]})		
		
		target_next_qvalues[done_array] = 0		

		#--------------R + GAMMA*MAX_A`(Q(S`,A`))----------#
		
		#-------------------------THE TARGET FOR UPDATE IS GOT BY USING THE TARGET NETWORKS OF THE ATTENTION AND THE BASE NETWORK--------------------#


		target = reward_array+self.gamma*np.max(target_next_qvalues,axis=1)

		feed_dict = {					
				self.target_update : action_array * target[:,None],
				self.input_image : current_array,
				self.action_mask : action_array,
				self.expert_q_pl:self.expert_q[current_cell_array]				
				}

		
		self.sess.run([self.optimizer_base,self.optimizer_attention],feed_dict)
		
		
	def run_network(self):

		steps = 0
		count = 0	
		saver = tf.train.Saver()	
		epsilons=np.linspace(self.epsilon,0.1,self.epsilon_decay_steps)
		t_ep_steps = 0
		selector = 0
		t_ep = []
		t_reward_l = []
		t_reward_steps = 0

		for i in range(self.episodes):

			done = False
			current_cell = self.env.reset()
			current_frame = self.generateFrame(current_cell)
			start_cell = copy(current_cell)

			self.stateQ.append(current_frame)

			current_state = self.generateState()
			t_reward = 0
			ep_steps = 0

			while not done:

				action = self.choose_action(current_state,current_cell)

				next_cell, reward, done, __ = self.env.step(action)
				
				#------------I UNDERSTAND THIS BLOCK CAN BE WRITTEN BETTER. THIS WILL BE USEFUL ONLY WHEN FRAMES ARE CONCATENATED.IN THIS CODE WE ARE CONSIDERING 
				#------------	ONLY THE CURRENT CELL AND HENCE CAN BE BETTER BUT IT IS BEING RETAINED FOR MAKING THINGS MORE GENERIC. FRAMES CAN BE CONCATENATED BY
				#-------------SETTING THE self.past_num TO A SUITABLE VALUE.

				next_frame = self.generateFrame(next_cell)
				self.stateQ.append(next_frame)
				next_state = self.generateState()

				#-----------------EVERY EPISODE LASTS FOR 1000 STEPS. EXTRA REWARD WHEN AGENT DRAGGES THE EPISODE TILL THE END TO REACH THE GOAL-------------#

				if ep_steps == 999:
					reward = -10

				action_one=np.eye(4)[action]

				self.memory.append([current_state,action_one,reward,next_state,done,current_cell,next_cell])

				if steps >= self.preFillLimit:

					if count==self.update_frequency:
						count = 0
						self.replay(selector%2) #-------------SELECTOR WAS USED TO ALTERNATE TRAINING BETWEEN ATTENTION AND BASE NETWORK---------------#
						selector += 1
				else:
					i = 0
					count=0

				if steps % self.target_updation_period == 0:
				
					self.sess.run([self.sync_1,self.sync_2])#--------UPDATING TARGET MODEL------------------#		

				current_state = next_state				
				current_cell = next_cell

				t_reward += reward
				count += 1
				steps += 1
				ep_steps += 1
			t_ep_steps +=ep_steps
			t_reward_steps += t_reward

			# --------------- LOGGING FEW VALUES ----------------------#


			if (i+1)%100 == 0:
				t_ep.append(t_ep_steps)
				t_reward_l.append(t_reward_steps)
				np.save('step_'+self.exp_type+'/'+self.file_name+str(self.goal)+'_'+str(self.random_seed)+'.npy',t_ep)	
				
				
				t_ep_steps = 0			
				t_reward_steps = 0
				
			self.epsilon=epsilons[min(steps,self.epsilon_decay_steps-1)]
			print('Episode:-',i,' Reward :- ',t_reward,'Epsilon:- ',self.epsilon,'Steps taken :- ',ep_steps)

			if i%1000 == 0:

				saver.save(self.sess,'models_'+self.exp_type+'/'+self.file_name+str(self.goal)+'_'+str(self.random_seed)+'/check')

			if i % 1000 == 0:

				self.examineAttention(i)
				
				
if __name__ == '__main__':

	obj = A2T()
