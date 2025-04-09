
"""
LlmMediumLevelAgent use llm generate medium level actions.
"""
import os
from overcook_mdp import ComplexOvercookedGridworld
from overcook_gym_env import OvercookPygameEnv
import json
import openai
from openai.types.chat.chat_completion import ChatCompletion
import itertools, os, json, re
from collections import defaultdict
import numpy as np
from overcook_gym_class import Action
import copy

PROMPT_DIR = '/alpha/ComplexOvercooked/prompts'

client = openai.OpenAI(api_key="EMPTY", base_url="http://202.117.43.44:8099/v1")

class LlmMediumLevelAgent():
	def __init__(
			self,
			mdp:ComplexOvercookedGridworld,
			env:OvercookPygameEnv,
			layout='supereasy',
			model='Qwen2.5-72B-Int4-Chat',
			prompt_level='l2-ap', # ['l1-p', 'l2-ap', 'l3-aip']
			belief_revision=False,
			retrival_method="recent_k",
			K=1, 
			auto_unstuck=False,
			controller_mode='new', # the default overcooked-ai Greedy controller
			debug_mode='N', 
			agent_index=None,
			outdir = None 
	):
		self.model = model
		self.trace = True 
		self.debug_mode = 'Y' 
		self.controller_mode = controller_mode 
		self.layout = layout
		self.mdp = mdp
		self.env = env 
		self.out_dir = outdir 
		self.agent_index = agent_index

		self.prompt_level = prompt_level
		self.belief_revision = belief_revision

		self.retrival_method = retrival_method
		self.K = K
		
		self.prev_state = None
		self.auto_unstuck = auto_unstuck

		self.current_ml_action = None
		self.current_ml_action_steps = 0
		self.time_to_wait = 0
		self.possible_motion_goals = None
		self.pot_id_to_pos = []

		self.layout_prompt = self.generate_layout_prompt()
		with open("/alpha/ComplexOvercooked/prompts/supereasy.txt", "r") as f:
			self.instruction_message = [{"role": "system", "content": f.read()}]


	def generate_layout_prompt(self):
		layout_prompt_dict = {
			"lemon_dispenser": " <lemon Dispenser {id}>",
			"tomato_dispenser": " <tomato Dispenser {id}>",
			"rawfish_dispenser": " <Rawfish Dispenser {id}>",
			"rawbeef_dispenser": " <rawbeef Dispenser {id}>",
			"hamburger_dispenser": " <hamburger Dispenser {id}>",
			"dish_dispenser": " <Dish Dispenser {id}>",
			"serving": " <Serving Loc {id}>",
			"pot": " <Pot {id}>",
			"cutting_table": " <Cutting table {id}>",

		}
		layout_prompt = "Here's the layout of the kitchen:"
		for obj_type, prompt_template in layout_prompt_dict.items():
			locations = getattr(self.mdp, f"get_{obj_type}_locations")()
			for obj_id, obj_pos in enumerate(locations):
				layout_prompt += prompt_template.format(id=obj_id) + ","
				if obj_type == "pot":
					self.pot_id_to_pos.append(obj_pos)
		layout_prompt = layout_prompt[:-1] + ".\n"
		print("layout_prompt:", layout_prompt)
		return layout_prompt
	  
	def action(self, state):

		start_pos_and_or = state["players_pos_and_or"][self.agent_index]

		# only use to record the teammate ml_action, 
		# if teammate finish ml_action in t-1, it will record in s_t, 
		# otherwise, s_t will just record None,
		# and we here check this information and store it into proagent
		self.current_timestep = state["timestep"]
		# if state.ml_actions[1-self.agent_index] != None:
		# 	self.teammate_ml_actions_dict[str(self.current_timestep-1)] = state.ml_actions[1-self.agent_index]

		# if current ml action does not exist, generate a new one
		if self.current_ml_action is None:
			self.current_ml_action = self.generate_ml_action(state)

		# if the current ml action is in process, Player{self.agent_index} done, else generate a new one
		if self.current_ml_action_steps > 0:
			current_ml_action_done = self.check_current_ml_action_done(state)  # 判断当前的ml_action是否执行结束了
			if current_ml_action_done:
				# generate a new ml action
				self.generate_success_feedback(state)  # 在大模型上下文中加入ml_action成功的反馈
				self.current_ml_action = self.generate_ml_action(state)

		count = 0
		# while not self.validate_current_ml_action(state):

		# 	self.trace = False
		# 	self.generate_failure_feedback(state)
		# 	self.current_ml_action = self.generate_ml_action(state)
			
		# 	count += 1
		# 	if count > 3:
		# 		self.current_ml_action = "wait(1)"
		# 		self.time_to_wait = 1

		
		self.trace = True 
		if "wait" in self.current_ml_action:
			self.current_ml_action_steps += 1
			self.time_to_wait -= 1
			
			# lis_actions = self.mdp.get_valid_actions(state.players[self.agent_index])
			lis_actions = self.get_avail_agent_actions(agent_id=self.agent_index).index(1)
			lis_actions = [i for i, value in enumerate(lis_actions) if value == 1]
			chosen_action =lis_actions[np.random.randint(0,len(lis_actions))]
			self.prev_state = state
			return chosen_action
		else:
			possible_motion_goals = self.find_motion_goals(state)   

			current_motion_goal, chosen_action = self.choose_motion_goal(
				start_pos_and_or, 
				possible_motion_goals, 
				state
			)

		# if self.auto_unstuck and chosen_action != Action.INTERACT:
		# 	if (
		# 			self.prev_state is not None
		# 			and state.players
		# 			== self.prev_state.players
		# 	):
		# 		if self.agent_index == 0:
		# 			joint_actions = list(
		# 				itertools.product(Action.ALL_ACTIONS, [Action.STAY])
		# 			)
		# 		elif self.agent_index == 1:
		# 			joint_actions = list(
		# 				itertools.product([Action.STAY], Action.ALL_ACTIONS)
		# 			)
		# 		else:
		# 			raise ValueError("Player index not recognized")

		# 		unblocking_joint_actions = []
		# 		for j_a in joint_actions:
		# 			if j_a != [Action.INTERACT,Action.STAY] and  j_a != [Action.STAY,Action.INTERACT]:
		# 				if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
		# 					new_state, _ = self.mlam.mdp.get_state_transition(state, j_a)
		# 				elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
		# 					new_state, _, _ = self.mlam.mdp.get_state_transition(state, j_a)		
		# 				if (
		# 						new_state.players_pos_and_or
		# 						!= self.prev_state.players_pos_and_or
		# 					):
		# 					unblocking_joint_actions.append(j_a)
		# 		unblocking_joint_actions.append([Action.STAY, Action.STAY])
		# 		chosen_action = unblocking_joint_actions[
		# 			np.random.choice(len(unblocking_joint_actions))
		# 		][self.agent_index]

		self.prev_state = state
		if chosen_action is None:
			self.current_ml_action = "wait(1)"
			self.time_to_wait = 1
			chosen_action = Action.STAY
		self.current_ml_action_steps += 1

		# print(f'ml_action = {self.current_ml_action}') 
		# print(f'P{self.agent_index} : {Action.to_char(chosen_action)}')
		if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
			return chosen_action, {}
		elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
			return chosen_action
		
		
	def generate_state_prompt(self, state):
		print(state)
		ego = f"player_{self.agent_index}"
		teammate = f"player_{1-self.agent_index}"
		time_prompt = f"Scene {state['timestep']}: "
		ego_object = state["player"][self.agent_index]
		teammate_object = state["player"][1-self.agent_index]
		ego_state_prompt = f"<Player {self.agent_index}> holds {ego_object}"

		teammate_state_prompt = f"<Player {1-self.agent_index}> holds {teammate_object}"
		
		kitchen_state_prompt = "Kitchen states: "
		kitchen_state_prompt += ",".join(state['pot'])

		task_state_prompt = f"Task states: Current task is {state['task']} and the remaining time is {state['tasktime']}."
		return (self.layout_prompt + time_prompt + ego_state_prompt +
				teammate_state_prompt + kitchen_state_prompt + task_state_prompt)
	
	def parse_ml_action(self, response, agent_index): 

		if agent_index == 0: 
			pattern = r'layer\s*0: (.+)'
		elif agent_index == 1: 
			pattern = r'layer\s*1: (.+)'
		else:
			raise ValueError("Unsupported agent index.")

		match = re.search(pattern, response)
		if match:
			action_string = match.group(1)
		else:
			# raise Exception("please check the query")
			action_string = response
			# print("please check the query")

		# Parse the response to get the medium level action string
		try: 
			ml_action = action_string.split()[0]
		except: 
			print('failed on 528') 
			action_string = 'wait(1)'
			ml_action = action_string
			# ml_action = 'wait(1)' 

		if "place" in action_string:
			ml_action = "place_obj_on_counter"

		elif "pick" in action_string:
			if "rawfish" in action_string:
				ml_action = "pickup_rawfish"
			elif "BClemon" in action_string:
				ml_action = "pickup_BClemon"
			elif "dish" in action_string:
				ml_action = "pickup_dish"

		elif "put" in action_string:
			if "rawfish" in action_string:
				ml_action = "put_rawfish_in_pot"
			elif "BClemon" in action_string:
				ml_action = "put_BClemon_in_cutting_table"

		elif "fill" in action_string:   
			ml_action = "fill_dish_with_soup"

		elif "deliver" in action_string:
			ml_action = "deliver_soup"

		elif "wait" not in action_string:
			ml_action='wait(1)'  
			action_string = ml_action
		if "wait" in action_string:
			
			def parse_wait_string(s):
				# Check if it's just "wait"
				if s == "wait":
					return 1

				# Remove 'wait' and other characters from the string
				s = s.replace('wait', '').replace('(', '').replace(')', '').replace('"', '').replace('.', '') 

				# If it's a number, return it as an integer
				if s.isdigit():
					return int(s)

				# If it's not a number, return a default value or raise an exception
				return 1
			if self.layout == 'forced_coordination': 
				# 这里可以改一下试试 
				self.time_to_wait = max(3, parse_wait_string(action_string))
			else: 
				self.time_to_wait = parse_wait_string(action_string)    
			# print(ml_action) 
			# print(self.time_to_wait) 
			
			ml_action = f"wait({self.time_to_wait})"

		else:
			pass
		
		# aviod to generate two skill, eg, Plan for Player 0: "deliver_soup(), pickup(onion)".
		if "," in ml_action:
			ml_action = ml_action.split(',')[0].strip()

		            
		return ml_action    

	def generate_ml_action(self, state):
		"""
		Selects a medium level action for the current state.
		Motion goals can be thought of instructions of the form:
			[do X] at location [Y]

		In this method, X (e.g. deliver the soup, pick up an onion, etc) is chosen based on
		a simple set of  heuristics based on the current state.

		Effectively, will return a list of all possible locations Y in which the selected
		medium level action X can be performed.
		"""

		belief_prompt = ''
		state_prompt = belief_prompt + self.generate_state_prompt(state)
		# ipdb.set_trace()

		print(f"\n\n### Observation module to GPT\n")   
		print(f"{state_prompt}")
		

		state_message = [{"role": "user", "content": state_prompt}]
		response = client.chat.completions.create(
							messages=self.instruction_message + state_message,
							model=self.model,
							max_tokens=256,
							temperature=0,
							stop=None,
						)
		response = response.choices[0].message.content
		# if 'wait' not in response:
		# 	self.planner.add_msg_to_dialog_history(state_message) 
		# 	self.planner.add_msg_to_dialog_history({"role": "assistant", "content": response})
		
		print(f"\n\n\n### GPT Planner module\n")   
		print("====== GPT Query ======")
		print(response)  


		print("\n===== Parser =====\n")
		
		ml_action = self.parse_ml_action(response, self.agent_index)

		# if "wait" not in ml_action:
		# 	self.planner.add_msg_to_dialog_history({"role": "assistant", "content": ml_action})
		
		print(f"Player {self.agent_index}: {ml_action}")
		self.current_ml_action_steps = 0
		return ml_action



	def find_motion_goals(self, state):
		"""
		Generates the motion goals for the given medium level action.
		Specifically handles BClemon and rawfish target locations.
		:param state: Current environment state
		:return: List of motion goals (position, direction)
		"""
		motion_goals = []
		# player = state["players_pos_and_or"][self.agent_index]
		current_action = self.current_ml_action
		counter_pickup_objects = self.mdp.get_counter_objects_dict()
		# 获取所有桌子的位置和物品信息
		tables = self.env.game.tables
		table_positions = [(table.rect.x // 80, table.rect.y // 80) for table in tables]

		# 根据当前动作选择目标
		if current_action in ["pickup(BClemon)", "pickup_BClemon"]:
			lemon_dispenser_loc = self.mdp.get_lemon_dispenser_locations()
			# 查找 BClemon 的目标点
			for table, pos in zip(tables, table_positions):
				if table.item == "BClemon":
					motion_goals.append(pos)

		elif current_action in ["pickup(rawfish)", "pickup_rawfish"]:
			# 查找 rawfish 的目标点
			rawfish_dispenser_loc = self.mdp.get_rawfish_dispenser_locations()

			for table, pos in zip(tables, table_positions):
				if table.item == "rawfish":
					motion_goals.append(pos)
		else:
			raise NotImplementedError
		

		# # 如果没有找到目标点，返回默认值
		# if not motion_goals:
		# 	print(f"No valid motion goals found for action: {current_action}")
		# 	motion_goals.append(player[0])  # 默认返回玩家当前位置

		return motion_goals


	def choose_motion_goal(self, start_pos_and_or, motion_goals, state = None):
		"""
		For each motion goal, consider the optimal motion plan that reaches the desired location.
		Based on the plan's cost, the method chooses a motion goal (either boltzmann rationally
		or rationally), and returns the plan and the corresponding first action on that plan.
		"""

		chosen_goal, chosen_goal_action = self.get_lowest_cost_action_and_goal_new(
				start_pos_and_or, motion_goals, state
			)

		return chosen_goal, chosen_goal_action
	
	
	def get_lowest_cost_action_and_goal_new(self, start_pos_and_or, motion_goals, state): 
		"""
		Chooses motion goal that has the lowest cost action plan.
		Returns the motion goal itself and the first action on the plan.
		"""   
		min_cost = np.inf
		best_action, best_goal = None, None
		for goal in motion_goals:   
			action_plan, plan_cost = self.real_time_planner(
				start_pos_and_or, goal, state
			)     
			print("action_plan = {}".format(action_plan))
			print("plan_cost = {}".format(plan_cost))
			if plan_cost < min_cost:
				best_action = action_plan
				min_cost = plan_cost
				best_goal = goal     
		if best_action is None: 
			# print('\n\n\nBlocking Happend, executing default path\n\n\n')
			# print('current position = {}'.format(start_pos_and_or)) 
			# print('goal position = {}'.format(motion_goals))        
			if np.random.rand() < 0.5:  
				return None, Action.STAY
			else: 
				return self.get_lowest_cost_action_and_goal(start_pos_and_or, motion_goals)
		return best_goal, best_action

	def real_time_planner(self, start_pos_and_or, goal, state):   
		terrain_matrix = {
			'matrix': copy.deepcopy(self.mdp.terrain_mtx), 
			'height': len(self.mdp.terrain_mtx), 
			'width' : len(self.mdp.terrain_mtx[0]) 
		}
		other_pos_and_or = state["players_pos_and_or"][1 - self.agent_index]
		
		## 在A*算法路径规划前暂时修正layout矩阵的y轴坐标偏差  TODO： 在overcook_gym_class中永久修复
		start_pos_and_or = [list(pos) for pos in start_pos_and_or]
		other_pos_and_or = [list(pos) for pos in other_pos_and_or]
		goal = list(goal)
		start_pos_and_or[0][1] -= 1
		other_pos_and_or[0][1] -= 1
		goal[1] -= 1

		action_plan, plan_cost = find_path(start_pos_and_or, other_pos_and_or, goal, terrain_matrix) 

		return action_plan, plan_cost


class Node:
    """
        A node class for A* Pathfinding
        parent is parent of the current Node
        position is current position of the Node in the maze
        g is cost from start to current Node
        h is heuristic based estimated cost for current Node to end Node
        f is total cost of present node i.e. :  f = g + h
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position
    
def find_path(start_pos_and_or, other_pos_and_or, goal, terrain_mtx):  
    
    start_node = Node(None, start_pos_and_or)  
    # end_node   = Node(None, goal)

    yet_to_visit_list = [] 
    visited_list = [] 
	
    move = [(-1, 0),    # left 
            (0, -1),    # up 
            (1, 0),     # right 
            (0, 1)]     # down 
    # NORTH = (0, 1)
    # SOUTH = (0, -1)
    # EAST = (1, 0)
    # WEST = (-1, 0)

    n_rows = terrain_mtx['height']  
    n_cols = terrain_mtx['width']    
    mtx = terrain_mtx['matrix'] 

    mtx[other_pos_and_or[0][1]][other_pos_and_or[0][0]] = 'B' 

    yet_to_visit_list.append(start_node)   

    # BFS search 
    while len(yet_to_visit_list) > 0:  
        current_node = yet_to_visit_list[0]    
        yet_to_visit_list.pop(0)  
        visited_list.append(current_node)   

        # reached, no need to search further
        if current_node.position[0] == goal[0]: 
            continue 
        
        for new_position in move:  
            node_position = (
                current_node.position[0][0] + new_position[0], 
                current_node.position[0][1] + new_position[1]
            ) 

            # position out of bound 
            if (node_position[0] > (n_cols) or 
                node_position[0] < 0 or 
                node_position[1] > (n_rows) or 
                node_position[1] < 0): 
                continue 
            
            if mtx[node_position[1]][node_position[0]] != ' ':
                continue 
                
            
            new_node = Node(current_node, (node_position, new_position))  

            if (new_node in visited_list) or (new_node in yet_to_visit_list): 
                continue  

            new_node.f = current_node.f + 1 
            yet_to_visit_list.append(new_node)  

    last_node = None 
    for i in visited_list:  
        if i.position[0] == goal[0]:   
            if last_node is None: 
                if i.position[1] == goal[1]: 
                    last_node = i  
                else: 
                    last_node = Node(i, (goal[0], goal[1])) 
                    last_node.f = i.f + 1    
            else: 
                if i.position[1] == goal[1] and i.f < last_node.f:
                    last_node = i 
                elif i.f + 1 < last_node.f: 
                    last_node = Node(i, (goal[0], goal[1])) 
                    last_node.f = i.f + 1 

    # no available plans. 
    if last_node is None: 
        return None, np.inf 
    else: 
        previous_node = last_node        
        while (previous_node.parent is not None) and (previous_node.parent != start_node): 
            previous_node = previous_node.parent     

        if previous_node == start_node:  
            return Action.INTERACT, 1
        else: 
            # did not move, changed direction 
            if previous_node.position[0] == start_node.position[0]: 
                return previous_node.position[1], last_node.f + 1 
            else: 
                # moved  
                return (
                    previous_node.position[0][0] - start_node.position[0][0], 
                    previous_node.position[0][1] - start_node.position[0][1]
                ), last_node.f + 1 
       


if __name__ == "__main__":
	mdp = ComplexOvercookedGridworld(map_name='supereasy')
	LlmMediumLevelAgent(mdp=mdp)