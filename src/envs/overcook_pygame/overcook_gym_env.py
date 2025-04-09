import numpy as np
import gym
import pygame
import os, sys
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict
from typing import Union, Dict, Tuple, List, Optional, Any
from gymnasium.spaces import flatdim
from overcook_gym_main import MainGame
from overcook_gym_class import ONEBLOCK, Table, Action, Direction, \
    TASK_FINISH_EVENT, OUT_SUPPLY_EVENT, OUT_DISH_EVENT, GET_MATERIAL_EVENT, \
        GET_DISH_EVENT, MADE_NEWTHING_EVENT, BEGINCUTTING_EVENT, CUTTINGDOWN_EVENT, \
            BEGINCOOKING_EVENT, COOKINGDOWN_EVENT, COOKINGOUT_EVENT, TRY_NEWTHING_EVENT,\
                  PUTTHING_DISH_EVENT, TRASH_EVENT

# 获取当前文件的目录
current_dir = os.path.dirname(__file__)
maps_path = os.path.join(current_dir, 'maps.json')
with open(maps_path, 'r', encoding='utf-8') as file:
    maps = json.load(file)

## 合成菜品的得分
TASK_VALUE = {'AClemoncookedfish': defaultdict(float,{'AClemoncookedfish': 5, 
                                                      'AClemon': 0, 
                                                      'cookedfish': 0, 
                                                      'BClemon': 0,
                                                      'rawfish': 0}),
              'cookedfish': defaultdict(float, {'cookedfish': 2, 
                                                'rawfish': 0}),
              'ACtomatocookedbeefhamburger': defaultdict(float, {'ACtomatocookedbeefhamburger': 6.0,
                                                                 'ACtomatohamburger': 0,
                                                                 'BCtomato': 0,
                                                                 'ACtomato': 0,
                                                                 'cookedbeefhamburger': 0,
                                                                 'cookedbeef': 0,
                                                                 'rawbeef': 0,
                                                                 'hamburger': 0}),
              'cookedbeefhamburger': defaultdict(float, {'cookedbeefhamburger': 3.0, 
                                                         'cookedbeef': 0, 
                                                         'rawbeef': 0,
                                                         'hamburger': 0})}

# 为了解决前两个问题，需要新的状态，记录场上各个食材的数量，玩家lastpick的标志位
# 计算reward时考虑，如果是【当前需要的食材，且场上食材数量不够2）则reward+1】
# 通过标识位来判断用户是拿起了东西，如果这个东西是需要的则加分，不需要的减分，并且当它放下后是减分的，除非用于处理或合成
# 如果当前盘子不够则加1（可以计算离盘子距离等）

def digitize(num):
    # 将数字转换为字符串并填充为4位
    num_str = str(num).zfill(4)
    # 取出每一位
    thousands = int(num_str[0])
    hundreds = int(num_str[1])
    tens = int(num_str[2])
    ones = int(num_str[3])
    # 返回每一位数字组成的元组
    return thousands, hundreds, tens, ones


def get_cleaned_matrix_size(matrix):
    # Find the height of the cleaned matrix
    height = next((i for i, s in enumerate(reversed(matrix)) if s.strip('_')), len(matrix))
    # Find the width of the cleaned matrix (max length of the trimmed strings)
    width = max(len(s.rstrip('_')) for s in matrix[:len(matrix) - height])
    # Size of the cleaned matrix (width, height)
    return len(matrix)-height, width


class OvercookPygameEnv(gym.Env):
    metadata = {'name': 'MyEnv-v0', 'render.modes': ['human']}

    # 玩家1：蓝色； 玩家2：红色
    # 玩家动作： stay：0，下：1，右：2，上：3，左：4，interact：5
    # X: 桌子， F：生鱼供应处，B：生牛肉供应处，H：汉堡包供应处，M：番茄供应处，D：盘子供应处，L：柠檬供应处，T：垃圾桶，E：送菜口，C：锅，U：案板
    def __init__(self, 
                 map_name, 
                 seed=1, 
                 ifrender=False, 
                 debug=False,
                 pltheatmap=False,
                 fps=60):
        # 初始化 pygame
        self.reward_shaping_params = {
            'one_step': 0, # 每走一步给一个负奖励
            'pick_dish': 0,
            'out_dish': 0,  # 从仓库拿盘子
            'pick_need_material': 0,  #
            'out_need_material': 0,  # 从仓库里拿东西
            'made_newthing': 0,  # hjh合成的奖励要多一点
            'process_cutting': 0,  # 只要在切，就加分
            'get_need_cutting': 3,  # 切出有意义的东西了
            'process_cooking': 0,  # 只要放东西进去煮，就加分/或放进去是need的东西才加分
            'get_need_cooking': 3,  # 煮出有意义的东西了
            # 'process_cutting': 0.15,  # 只要在切，就加分
            # 'get_need_cutting': 0.8,  # 切出有意义的东西了
            # 'process_cooking': 0.15,  # 只要放东西进去煮，就加分/或放进去是need的东西才加分
            # 'get_need_cooking': 0.8,  # 煮出有意义的东西了
            'try_new_thing': 0,  # hjh 只要去尝试合成了（手上有东西去碰另外的东西）
            'putting_dish': 0  # hjh 只要用盘子收东西，或者把做好的东西放到盘子上就给奖励
        }
        super(OvercookPygameEnv, self).__init__()
        self.map_name:str = map_name
        self.TASK_MENU:Dict[str,int] = maps[map_name]['task']
        self.n_agents:int = maps[map_name]['players']
        self.TASKNUM:int = maps[map_name]['tasknum']
        self.ITEMS:List[str] = maps[map_name]['items']

        self.debug = debug
        self.episode_limit = 600
        self.fps = fps
        self.game_over = False
        self.timercount = 0  # 自定义的计时器
        self.showkey()
        self.ifrender = ifrender
        self.pltheatmap = pltheatmap
        if self.pltheatmap:
            self.heatmapsize:Tuple = get_cleaned_matrix_size(maps[map_name]['layout'])

        # wanghm
        # self.observation_space= gym.spaces.Tuple(tuple(self._setup_observation_space() for _ in range(self.n_agents))) # wanghm
        # self.observation_space = self._setup_observation_space()
        # self.share_observation_space = [self.observation_space, self.observation_space]
        
        self.obs_shape = self.dummy_reset()[0].shape[0]
        self.reset_featurize_type(obs_shape=self.obs_shape)

        # 初始化游戏
        self.initialize_game()
        self.t = 0
        self._max_episode_steps = 600

    # def _setup_observation_space(self):
    #     obs_shape = self.reset()[0].shape[0]
    #     low = np.ones(obs_shape, dtype=np.float32) * -np.inf
    #     high = np.ones(obs_shape, dtype=np.float32) * np.inf
    #     return gym.spaces.Box(low, high, dtype=np.float32)

    def get_share_observation(self, nobs:List[np.ndarray]) -> np.ndarray:
        """
        share observation 和 state 没什么区别
        """
        share_obs = np.stack(nobs, axis=0)
        return share_obs

    def initialize_game(self):
        self.get_need_cutting = 0
        self.get_need_cooking = 0
        self.get_need_synthesis = 0

        self.clock = pygame.time.Clock()

        self.game = MainGame(map_name=self.map_name, 
                             ifrender=self.ifrender)
        
        self.taskcount = [{key:0 for key in self.TASK_MENU} for i in range(self.n_agents)]
        self.alltaskcount = {key:0 for key in self.TASK_MENU} #用来计算总的任务变更
        for taskname in self.game.taskmenu:
            self.alltaskcount[taskname]+=1
        if self.pltheatmap:
            self.heatmap = [np.zeros((self.heatmapsize[0], self.heatmapsize[1])) for i in range(self.n_agents)]
        else:
            self.heatmap = None
        self.matiral_count = {key: 0 for key in self.itemdict.keys()}
        self.playitem = [self.game.playergroup[i].item for i in range(self.n_agents)]
        self.playdish = [self.game.playergroup[i].dish for i in range(self.n_agents)]
        self.playerdic = {'a':0,'b':1,'c':2,'d':3}
        
    def change_shapereward(self,reward_shaping_params): #for the llm to change the reward_shaping_params
        self.reward_shaping_params = reward_shaping_params
        
    def dummy_reset(self):
        self.initialize_game()
        self.timercount = 0
        self.state = {}  

        nobs = self.get_obs()
        return nobs
    
    def reset(self) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]:  # wanghm
        self.initialize_game()
        self.timercount = 0
        self.state = {}  # self.state is used in LLM
        nobs = self.get_obs() 
        
        available_actions = self.get_avail_actions()
        share_obs = self.get_share_observation(nobs)
        self.episode_reward_dict = {
            "cumulative_sparse_rewards": np.array([0.0]),
            "cumulative_shaped_rewards": np.array([0.0]),
            "cumulative_rewards":np.array([0.0])
        }
        return nobs, share_obs, available_actions

    def reset_featurize_type(self, obs_shape: int)->None:
        # reset observation_space, share_observation_space and action_space
        self.observation_space = []
        self.share_observation_space = []
        self.action_space = []
        # self._setup_observation_space()
        high = np.ones(obs_shape) * float("inf")
        low = np.ones(obs_shape) * -float("inf")
        for i in range(self.n_agents):
            self.observation_space.append(gym.spaces.Box(np.float32(low), np.float32(high), dtype=np.float32))
            self.action_space.append(gym.spaces.Discrete(6))
            # self.share_observation_space.append(self._setup_share_observation_space(obs_shape * self.n_agents))
            self.share_observation_space.append(self._setup_share_observation_space(obs_shape))

    def _setup_share_observation_space(self, share_obs_shape: int)->gym.spaces.Box:
        high = np.ones(share_obs_shape) * float("inf")
        low = np.ones(share_obs_shape) * -float("inf")
        return gym.spaces.Box(np.float32(low), np.float32(high), dtype=np.float32)

    def get_avail_actions(self)->List[np.ndarray]:
        avaliable_actions = []
        for i in range(self.n_agents):
            avaliable_actions.append(self.get_avail_agent_actions(i))
        return avaliable_actions
        
    def get_avail_agent_actions(self, agent_id:int)->np.ndarray:
        availaction = [Action.STAY]
        player = self.game.playergroup[agent_id] #python中理论上改动这个player原先的也会变
        if player.cutting:#切菜时只有不动和交互合法
            availaction.append(Action.INTERACT)
            avaliable_actions = [0] * 6
            for index in availaction:
                avaliable_actions[index] = 1
            return avaliable_actions

        #检测移动是否会阻塞
        tempplayergroup = []
        for j in range(self.n_agents):
            if j != agent_id:
                tempplayergroup.append(self.game.playergroup[j])

        for action in range(1,5):
            if player.availaction(action, self.game.walls, tempplayergroup):#直接调用移动函数判断是否合法,转向是合法的
                availaction.append(action)

        #检测交互是否合法（比如前面是空气，桌子上也没有菜或者任何可取的，就不合法）
        for table in self.game.tables:
            if table.availbeinter(player):
                availaction.append(Action.INTERACT)
                break
        if self.game.Cointable.availbeinter(player):
                availaction.append(Action.INTERACT)
        # availaction.append(5)

        avaliable_actions = [0] * 6
        for index in availaction:
            avaliable_actions[index] = 1
        return avaliable_actions

    def step(self, action_n: List[int]) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], np.ndarray]:
        if self.pltheatmap:
            for i in range(self.n_agents):
                self.heatmap[i][self.game.playergroup[i].rect.y / 80-1][self.game.playergroup[i].rect.x / 80 ] += 1

        self.timercount += 1

        for i in range(self.n_agents):
            if action_n[i] == Action.INTERACT:
                self.game.tables.update(self.game.playergroup[i], 
                                        True, 
                                        self.timercount)#有可能会post not finished
                
                self.game.Cointable.update(self.game.playergroup[i], 
                                           True, 
                                           self.game.taskmenu)#有可能会post success
            else:
                self.game.tables.update(self.game.playergroup[i], 
                                        False, 
                                        self.timercount)#有可能会post not finished
                
                self.game.Cointable.update(self.game.playergroup[i], 
                                           False, 
                                           self.game.taskmenu)#有可能会post success  
                tempplayergroup = []
                for j in range(self.n_agents):
                    if j != i:
                        tempplayergroup.append(self.game.playergroup[j])

                self.game.playergroup[i].update(action=action_n[i], 
                                                pengzhuang=self.game.walls, 
                                                player=tempplayergroup)
                
        self.game.timercount.update(self.timercount)
        
        sparse_reward, shaped_reward = self.calculate_reward()
        reward = sparse_reward + shaped_reward
        if self.debug:
            print('sparse_reward:', sparse_reward)
            print('shaped_reward:', shaped_reward)

        done = self.is_done()


        if self.ifrender:
            self.render()
            
        available_actions = self.get_avail_actions()
        tasksequence = []
        self.t += 1  # 训练总时间

        infos = {'shaped_r': shaped_reward,
                 'sparse_r': sparse_reward,
                 'reward': sparse_reward + shaped_reward,
                #  'tasksequence': tasksequence,
                #  'heatmap':self.heatmap
                }
        
        self._update_reward_dict(infos)

        if done:  
            if float(self.episode_reward_dict["cumulative_rewards"]) > 800: 
                print('异常分数')
                print(self.taskcount)
                print(self.timercount)     
            infos["episode"] = {
                "ep_reward": self.episode_reward_dict["cumulative_rewards"],
                "ep_sparse_r": self.episode_reward_dict["cumulative_sparse_rewards"],
                "ep_shaped_r": self.episode_reward_dict["cumulative_shaped_rewards"],
                "ep_length": self.timercount,
                "cooked_count": self.get_need_cooking,
                "cutted_count": self.get_need_cutting,
                "synthesis_count": self.get_need_synthesis,
                "success_count": self.taskcount,
            }

        nobs = self.get_obs()
        share_obs = self.get_share_observation(nobs)
        dones = [done] * self.n_agents
        rewards = [reward] * self.n_agents
        
        # 返回新的状态、奖励、是否结束和其他信息
        return nobs, share_obs, rewards, dones, infos, available_actions

    def _update_reward_dict(self, infos):
        self.episode_reward_dict["cumulative_sparse_rewards"] += np.array(infos["sparse_r"])
        self.episode_reward_dict["cumulative_shaped_rewards"] += np.array(infos["shaped_r"])
        self.episode_reward_dict["cumulative_rewards"] += np.array(infos["reward"])

    def pick_random_state_or_goal(self):  # 返回一个随机的合理的状态
        pass

    def showkey(self):
        # 可以让环境反馈一个itempiclist，就不用自己设定了
        # itempicslist = self.showitemlist()
        '''
        itempicslist = ['dish', 'BClemon', 'AClemon', 'rawfish',
                        'AClemoncookedfish', 'cookedfish', 'pot',
                        'ACtomatocookedbeefhamburger', 'cookedbeefhamburger',
                        'hamburger', 'ACtomato', 'BCtomato', 'rawbeef', 'cookedbeef', "ACtomatohamburger"]
        '''
        self.itemdict:Dict[str, int] = {char: index + 1 for index, char in enumerate(self.ITEMS)}
        self.taskdict:Dict[str, int] = {char: index for index, char in enumerate(self.TASK_MENU)}

    def calculate_reward(self) -> Tuple[float, float]:
        tasksequence = []
        finished_count = 0
        sparse_reward = 0.0
        shaped_reward = 0.0
        
        self.game.task_sprites.update(self.timercount)
        taskfinished = [False for _ in range(len(self.game.task_sprites))]  # 当前这个时间步，有没有任务被完成了
        for event in pygame.event.get():
            if event.type == pygame.USEREVENT:
                if event.action == "countdown_finished":
                    self.game_over = True

                elif event.action == "notfinished":
                    # 如果不同，代表已经经过前面的事件更新了，那此时我不用更新了
                    if self.game.taskmenu[self.game.task_dict[event.taskclass]] != event.oldtask:
                        print("任务冲突，以新任务为准,这条不应该会出现")
                    else:
                        self.game.NOWCOIN -= 1
                        self.game.taskmenu[self.game.task_dict[event.taskclass]] = event.newtask
                        tasksequence.append("Failed to complete required food within the time")
                        self.alltaskcount[event.newtask]+=1
                    # print("未能在任务时间内完成菜品制作")
                    # reward-=10

            elif event.type == TASK_FINISH_EVENT:  # hjh这里是任务完成的奖励
            
                self.game.NOWCOIN += self.TASK_MENU[event.action][0] if isinstance(self.TASK_MENU[event.action], list) else self.TASK_MENU[event.action]
                # finished_count += self.TASK_MENU[event.action]
                self.taskcount[self.playerdic[event.player]][event.action] += 1  # 用于展示任务完成次数
                sparse_reward += self.TASK_MENU[event.action][0] if isinstance(self.TASK_MENU[event.action], list) else self.TASK_MENU[event.action]
                # print(f"成功送出菜品 {event.action}")

                named_tasks = []
                for task in self.game.task_sprites:
                    if task.task == event.action:  # 去找到agent完成了哪个任务优先筛检剩余时间少的任务
                        named_tasks.append(task)  # 有可能为空，为空是因为前面倒计时重新更新了
                if named_tasks:
                    min_task = min(named_tasks, key=lambda task: task.remaining_time)
                    min_task.newtask(self.timercount)
                    self.game.taskmenu[self.game.task_dict[min_task]] = min_task.task
                    self.alltaskcount[min_task.task] += 1
                    # taskfinished[self.game.task_dict[min_task]]=True#对应的任务被完成了，此时对应的task被更新了一次
                else:
                    self.game.NOWCOIN += 1  # 代表之前倒计时重新了
                    #tasksequence.pop()
                tasksequence.append("Successfully delivered the required food")
            elif event.type == OUT_SUPPLY_EVENT:
                self.matiral_count[event.item] += 1
                if self.matiral_count[event.item] <= 2:
                    for task in self.game.task_sprites:
                        # if event.item in task.task:
                        if event.item in TASK_VALUE[task.task]:
                            shaped_reward += self.reward_shaping_params['out_need_material']
                            # self.get_reward_counts['out_need_material']+=1
                            # tasksequence.append("Take out the required materials from the supplys")
                            if self.debug:
                                print("从仓库中拿到东西")

            elif event.type == OUT_DISH_EVENT:
                self.matiral_count['dish'] += 1
                if self.matiral_count['dish'] <= 2:
                    shaped_reward += self.reward_shaping_params['out_dish']
                    tasksequence.append("Take out the dish from the supplys")
                    if self.debug:
                        print("从仓库中拿到盘子")
                    # self.get_reward_counts['out_dish']+=1

            elif event.type == GET_MATERIAL_EVENT:
                # 如果拿到手里的东西对比原来是更需要的，那么加分，否则扣分
                for task in self.game.task_sprites:
                    if TASK_VALUE[task.task][event.newitem] > TASK_VALUE[task.task][event.olditem]:
                        shaped_reward += self.reward_shaping_params['pick_need_material']
                        tasksequence.append("Exchange a thing more meaningful for the tasks")
                        if self.debug:
                            print("换过来的东西是更有价值的")
                        # self.get_reward_counts['pick_need_material']+=1
                    # else:
                    #     reward -=self.reward_shaping_params['pick_need_material']
                    #     print("未能从仓库中拿到菜品制作所需要的材料")
                    #     self.get_reward_counts['pick_need_material']-=1
            elif event.type == TRY_NEWTHING_EVENT:
                shaped_reward += self.reward_shaping_params['try_new_thing']
                # self.get_reward_counts['try_new_thing']+=1
                tasksequence.append("Trying to synthesize new things")
                if self.debug:
                    print("尝试合成")
            elif event.type == MADE_NEWTHING_EVENT:
                # 如果合成的是当前需要的任务的东西,则加分
                tempreward = 0
                for idx, task in enumerate(self.game.task_sprites):
                    tempreward += TASK_VALUE[task.task][event.newitem]
                    self.get_need_synthesis += 1
                # if tempreward==0:
                #     shaped_reward-=self.reward_shaping_params['made_newthing'] # 合成了废品 会扣分
                #     if self.debug:
                #         print("合成了废品")
                # #     self.get_reward_counts['made_newthing']-=1
                # else:
                shaped_reward += tempreward
                # tasksequence.append(f"Synthesized a {event.newitem}")
                if self.debug:
                    print(f"合成了{event.newitem}，获得奖励{tempreward}")

            elif event.type == BEGINCUTTING_EVENT:
                shaped_reward += self.reward_shaping_params['process_cutting']
                tasksequence.append(f"begin cutting up material")
                if self.debug:
                    print("正在切菜")

            elif event.type == CUTTINGDOWN_EVENT:
                for task in self.game.task_sprites:
                    if event.item in task.task:
                        shaped_reward += self.reward_shaping_params['get_need_cutting']
                        self.get_need_cutting += 1
                        tasksequence.append(f"cut the {event.item} completely")
                        if self.debug:
                            print(f"把{event.item}切好了，奖励{shaped_reward}")

            elif event.type == BEGINCOOKING_EVENT:
                shaped_reward += self.reward_shaping_params['process_cooking']
                tasksequence.append(f"begin cooking")
                if self.debug:
                    print("正在煮菜")

            elif event.type == COOKINGDOWN_EVENT:
                for task in self.game.task_sprites:
                    if event.item in task.task:
                        shaped_reward += self.reward_shaping_params['get_need_cooking']
                        self.get_need_cooking += 1
                        tasksequence.append(f"cook up the required material{event.item}")
                        if self.debug:
                            print(f"把{event.item}煮好了，奖励{self.reward_shaping_params['get_need_cooking']}")
            # elif event.type == COOKINGOUT_EVENT:
            #     if 'raw' in event.item:
            #         reward-=10
            #     else:
            #         for task in self.game.task_sprites:
            #             reward+=TASK_VALUE[task.task][event.item]

            elif event.type == PUTTHING_DISH_EVENT:
                if 'raw' or 'BC' in event.item:
                    tasksequence.append("Carrying unprocessed products on a plate")
                    if self.debug:
                        print('用盘子端未加工品')
                    shaped_reward -= self.reward_shaping_params['putting_dish']
                else:
                    for task in self.game.task_sprites:
                        if event.item in TASK_VALUE[task.task]:
                            tasksequence.append("Carrying required processed products on a plate")
                            if self.debug:
                                print('端到的东西是加工过的')
                            shaped_reward += self.reward_shaping_params['putting_dish']

            elif event.type == TRASH_EVENT:
                for task in self.game.task_sprites:
                    if event.item in TASK_VALUE[task.task].keys():  # wanghm
                        tasksequence.append("Pour something into the trash can")
                        # if TASK_VALUE[task.task][event.item] >=3:
                        #     if self.print_rew_log:
                        #     print(f"倒掉了{event.item},得到负奖励-1")

                        # 这里如果考虑如果把第二级的合成品放进去（如牛肉番茄汉堡）有马上要合成的牛肉汉堡，把这种东西倒掉就很可惜，就扣分。先不考虑为场上空出来位置这种、因为要计算的量太大了
                        # shaped_reward -= 1
                        break

            shaped_reward += self.reward_shaping_params['one_step']

            th, hu, te, on = digitize(self.game.NOWCOIN)
            self.game.num1.set_num(th)
            self.game.num2.set_num(hu)
            self.game.num3.set_num(te)
            self.game.num4.set_num(on)

        # return sparse_reward, shaped_reward, tasksequence
        return sparse_reward, shaped_reward
    
    def get_state(self, nobs):
        tasks, tasktime = self.cal_tasktime()


        game_state = {
                "player1_pos": (nobs[0][0], nobs[0][1]),  # 玩家位置
                "player1_item": nobs[0][3],  # 手持物品
                "player1_has_dish": nobs[0][4],  # 是否持有盘子
                "player1_is_cutting": nobs[0][5],  # 是否在切菜
                
                "player2_pos": (nobs[1][0], nobs[1][1]),
                "player2_item": nobs[1][3],
                "player2_has_dish": nobs[1][4],
                "player2_is_cutting": nobs[1][5],
                
                "order": tasks,  # 当前订单列表
                "tasktime": tasktime,  # 当前得分
                "time_left":  self.timercount  # 剩余时间
            }
        
        return game_state
    
    
    

    def get_obs(self) -> List[List[np.ndarray]]:
        """
        Encode state with some manually designed features. Works for arbitrary number of players

        Returns:
            ordered_features (list[np.Array]): The ith element contains a player-centric featurized view for the ith player

            The encoding for player i is as follows:

                [player_features[i], other_player_features, player_i_dist_to_other_players, player_i_position]

                player_{i}_features:
                    pi_orientation: length 4 one-hot-encoding of direction currently facing
                    pi_obj: length 4 one-hot-encoding of object currently being held (all 0s if no object held)
                    pi_closest_{onion|tomato|dish|soup|serving|empty_counter}: (dx, dy) where dx = x dist to item, dy = y dist to item. (0, 0) if item is currently held                    
                    pi_closest_pot_{j}_{is_empty|is_cooking|is_ready}: {0, 1} depending on boolean value for jth closest pot # TODO
                    pi_closest_pot_{j}_cook_time: int value for seconds remaining on soup. -1 if no soup is cooking # TODO
                    pi_closest_pot_{j}: (dx, dy) to jth closest pot from player i location # TODO
                    pi_closest_cutting_{j}_{is_empty|is_cutting}: {0, 1} depending on boolean value for jth closest cutting table # TODO
                    pi_closest_cutting_{j}: (dx, dy) to jth closest cutting table from player i location # TODO
                    pi_wall: length 4 boolean value of whether player i has wall in each direction # TODO

                other_player_features:
                    ordered concatenation of player_{j}_features for j != i

                player_i_dist_to_other_players (length (num_players - 1)*2):
                    [player_j.pos - player_i.pos for j != i]

                player_i_position (length 2)
        """

        def concat_dicts(a:Dict, b:Dict)->Dict:
            return {**a, **b}
        
        def ManhattanDis(relativepos:Tuple[int, int])->int:
            return abs(relativepos[0]) + abs(relativepos[1])
        
        def rel_xy(p:pygame.sprite.Sprite, 
                   obj:pygame.sprite.Sprite)-> List[int]:
            return [(p.rect.x-obj.rect.x)/80, 
                    (p.rect.y-obj.rect.y)/80]  
        
        assert(isinstance(self.game.task_sprites, pygame.sprite.Group))
        assert(isinstance(self.game.playergroup, list))
        assert(isinstance(self.game.pots, pygame.sprite.Group))
        assert(isinstance(self.game.cuttingtables, pygame.sprite.Group))
        assert(isinstance(self.game.tables, pygame.sprite.Group))

        # sparse_r, shaped_r, tasksequence = self._calculate_rew()
        players:List[pygame.sprite.Sprite] = self.game.playergroup

        playab_cointables = [[] for _ in range(self.n_agents)]  # cointable的位置，不需要（最近）因为coin只会有一个
        playab_closet_item_pos = [[10e9] * (len(self.itemdict)) * 2 for _ in range(self.n_agents)]  # 所有最近物品的位置
        for i, player in enumerate(players):
            playab_cointables[i] = rel_xy(player, self.game.Cointable)
            # 计算一下收银台的位置
            if player.item:
                itemindex = (self.itemdict[player.item] - 1) * 2
                playab_closet_item_pos[i][itemindex:itemindex + 2] = [0, 0]
            elif player.dish:
                dishindex = (self.itemdict[player.dish] - 1) * 2
                playab_closet_item_pos[i][dishindex:dishindex + 2] = [0, 0]

        emptypalcenum = 0  # 空桌子的数量
        tablenum = 0
        playab_tables = [[] for _ in range(self.n_agents)]  # 所有可交互物的位置
        playab_closet_empty_table_pos = [[10e9, 10e9] for _ in range(self.n_agents)]  # 最近的空桌子
        for temp in self.game.tables:  # 目前的tables中不包含coin
            if temp.item:
                # supply的item属性也被改了，现在所有tables都有item属性，如果true说明有这个东西，需要计算最近距离
                # 在别的玩家身上还要算
                for i, player in enumerate(players):
                    itemindex = (self.itemdict[temp.item] - 1) * 2  # 如番茄的存储位置
                    temppos = rel_xy(player, temp)

                    if ManhattanDis(temppos) < ManhattanDis(playab_closet_item_pos[i][itemindex:itemindex + 2]):
                        playab_closet_item_pos[i][itemindex:itemindex + 2] = temppos
            
            if isinstance(temp, Table):
                if temp.dish:  # 在考虑dish距离，和上一段是一致的，只是dish单独有个属性
                    for i, player in enumerate(players):
                        dishindex = (self.itemdict[temp.dish] - 1) * 2
                        temppos = rel_xy(player, temp)
                        if ManhattanDis(temppos) < ManhattanDis(playab_closet_item_pos[i][dishindex:dishindex + 2]):
                            playab_closet_item_pos[i][dishindex:dishindex + 2] = temppos
                if not temp.item:
                    emptypalcenum += 1
                    # 空桌子就计算一下最近的空桌子
                    for i, player in enumerate(players):
                        temppos = rel_xy(player, temp)
                        if ManhattanDis(temppos) < ManhattanDis(playab_closet_empty_table_pos[i][0:2]):
                            playab_closet_empty_table_pos[i][0:2] = temppos
                continue
                
            for i, player in enumerate(players):
                tablenum += 1
                playab_tables[i] += rel_xy(player, temp)
                
        # print(tablenum)
        # print(list(self.game.tables))
        # print(len(self.game.tables))
        # print("*"*10)
        for i, player in enumerate(players):
            if playab_closet_empty_table_pos[i][0] > 100000:  # 没有空桌子了
                playab_closet_empty_table_pos[i] = [0, 0]

            for j in range(len(playab_closet_item_pos[i])):  # 如果没有找到这个物品说明还没出现，也赋值0
                if playab_closet_item_pos[i][j] > 100000:
                    playab_closet_item_pos[i][j] = 0

        # 所有桌子和所有案板、锅和每个玩家的相对距离 num_pots * n_agents * 2 + num_cuttingtables * n_agents * 2
        playab_pots_pos = [[10e9, 10e9]*len(self.game.pots) for _ in range(self.n_agents)]  #所有锅的相对距离
        pots_state = [0, 0]*len(self.game.pots)
        pots_remaining_time = [-1]*len(self.game.pots)
        pots_lang = []
        for p, pot in enumerate(self.game.pots):
            for i, player in enumerate(players):
                potpos = rel_xy(player, pot)
                playab_pots_pos[i][p*2:(p*2)+2] = potpos  # 修正索引计算
            # (0,0) - 空锅 (is_empty)
            # (0,1) - 正在烹饪 (is_cooking)  
            # (1,0) - 食物准备好 (is_ready)
            if pot.is_empty:
                pots_state[p*2:(p+1)*2] = [0, 0]
                pots_lang.append(f"pot{p} is empty")
            elif pot.is_cooking:
                pots_state[p*2:(p+1)*2] = [0, 1]
                pots_lang.append(f"pot{p} is cooking, the {pot.item} will be ready in {pot.remaining_time} seconds")

            elif pot.is_ready:
                pots_state[p*2:(p+1)*2] = [1, 0]
                pots_lang.append(f"{pot.item} in pot{p} is ready")

            pots_remaining_time[p] = pot.remaining_time / 10 if pot.remaining_time >=0 else -1

        self.state.update({"pot": pots_lang})

        
        playab_cuttingtables_pos = [[10e9, 10e9]*len(self.game.cuttingtables) for _ in range(self.n_agents)]  #所有案板的相对距离
        cuttingtables_state = [0,0]*len(self.game.cuttingtables)
        for p,cuttingtable in enumerate(self.game.cuttingtables):
            for i, player in enumerate(players):
                cuttingtablepos = rel_xy(player, cuttingtable)
                playab_cuttingtables_pos[i][p*2:(p*2)+2] = cuttingtablepos
            if cuttingtable.is_empty:
                cuttingtables_state[p*2:(p+1)*2] = [0, 0]
            elif cuttingtable.is_cutting:#非空又没切这种情况理论上没有？
                cuttingtables_state[p*2:(p+1)*2] = [0, 1]
            elif pot.is_ready:
                cuttingtables_state[p*2:(p+1)*2] = [1, 0]


        tasktime = []
        nowtime = self.timercount
        self.state.update({"timestep": nowtime})

        tasks = []
        task_name = []
        for task in self.game.task_sprites:
            tasks.append(task)
            task_name.append(task.task)
            tasktime.append((task.timer - (nowtime - task.start_time)))
        self.state.update({"task": task_name, 
                           "tasktime": tasktime})

        current_goal = list(map(lambda x: (np.eye(len(self.taskdict))[self.taskdict[x.task]]).tolist(), tasks))  # 这里为什么+1
        # current_goal = list(map(lambda x: (np.eye(len(self.taskdict)+1)[self.taskdict[x.task]]).tolist(), tasks))  # 这里为什么+1
        
        # 在任务数多于一个的环境中考虑各任务名称和任务剩余时长
        if len(tasks) > 1:
            flatencurrent_goal = []
            for taskonehot in current_goal:
                flatencurrent_goal += taskonehot
            # task = np.array([remaintime/10, self.game.NOWCOIN, iscook, emptypalcenum ] + tasktime + taskreward + flatencurrent_goal)
            task_feature = [i / 100 for i in tasktime] + flatencurrent_goal
        else:
            task_feature = []

        # TODO
        # 增加最近的空桌子，最近的几个操作台，最近的所有物资，如果在手上，则(0,0)
        # 如果这个环境中不会出现这个物资，该怎么办？不加，或者还没产生，怎么赋值？(0,0)

        # TODO整理成feature字典
        # all_features = {}
        # for i, player in enumerate(players):
        #     # player orientation
        #     orientation_idx = [self.directiondict[str(player.direction)]]
        #     all_features[f"p{i}_orientation"] = np.eye(4)[orientation_idx]
            
        #     # player holding object
        #     if player.item:
        #         hold_obj = np.eye(len(self.itemdict))[self.itemdict[player.item]].tolist()
        #     else:
        #         hold_obj = np.zeros(len(self.itemdict)).tolist()
        #     all_features[f"p{i}_obj"] = hold_obj

        player_absolute_positions = []
        player_relative_positions = []
        player_features = []
        if self.debug:
            print(f"当前时间步: {self.timercount}")
        hold_objects = []
        ori = []
        pos = []
        
        for i, player in enumerate(players):
            # 玩家朝向
            orientation_idx = Direction.DIRECTION2INDEX[player.direction]
            orientation = np.eye(len(Direction.DIRECTION2INDEX))[orientation_idx].tolist()
            ori.append(tuple(player.direction))
            # 玩家手持物品
            if player.item:
                hold_obj = np.eye(len(self.itemdict))[self.itemdict[player.item]-1].tolist()
                hold_objects.append(player.item)
            else:
                hold_obj = np.zeros(len(self.itemdict)).tolist()
                hold_objects.append("nothing")

            # 玩家四周是否有墙壁
            collide = []
            rect_sprite = pygame.sprite.Sprite()
            for direction in list(Direction.directions.values()): # 下右上左
                rect_sprite.rect = player.rect.move(direction[0] * ONEBLOCK / 2,
                                                    direction[1] * ONEBLOCK / 2)
                is_block = pygame.sprite.spritecollide(rect_sprite, self.game.tables, False)
                collide.append(int(bool(is_block)))

            player_features.append(
                np.array(orientation
                         +hold_obj
                         +collide
                         +playab_cointables[i]
                         +playab_pots_pos[i]
                        +playab_cuttingtables_pos[i]
                        +playab_closet_item_pos[i]
                        # +playab_tables[i] 
                        # +playab_closet_empty_table_pos[i]
                         ))
            
            # if self.debug:
            #     print('*' * 10)
            #     print(f'玩家{i} player_feature:')
            #     print(f'面向：{orientation} {self.directionzhongwen[str(player.direction)]}')
            #     print(f'手持物品: {hold_obj}', player.item if player.item else '空手')
            #     print(f'四面是否有墙壁：{collide}')
            #     print(f'和coin table的相对距离：{playab_cointables[i]}')
            #     print(f'和pots的相对距离：{playab_pots_pos[i]}')
            #     print(f'和cuttingtables的相对距离：{playab_pots_pos[i]}')
            #     print(f'和每个功能桌子（非table、cointable）的相对距离：{playab_tables[i]}')
            #     print("\n")

            player_absolute_positions.append(np.array([player.rect.x / 80, 
                                                        player.rect.y / 80]))
            pos.append((player.rect.x//80, player.rect.y//80))
            tempdis = []
            for j in range(self.n_agents):
                if j != i:
                    tempdis += rel_xy(players[i], self.game.playergroup[j])
            player_relative_positions.append(np.array(tempdis))

        self.state.update({"player": hold_objects})
        self.state.update({"players_pos_and_or": tuple(zip(*[pos, ori]))})
        ordered_features = []
        for i in range(self.n_agents):
            player_i_features = player_features[i]
            player_i_abs_pos = player_absolute_positions[i]
            player_i_rel_pos = player_relative_positions[i]
            other_player_features = np.concatenate(
                [player_features[j] for j in range(self.n_agents) if j != i]
            )
            player_i_ordered_features = np.squeeze(
                np.concatenate(
                    [
                        player_i_features,
                        other_player_features,
                        player_i_rel_pos,
                        player_i_abs_pos,
                        pots_state,
                        pots_remaining_time,
                        cuttingtables_state,
                        task_feature
                    ]
                )
            )

            # if pots_remaining_time[0] > 0:
            #     print(pots_remaining_time)
            ordered_features.append(player_i_ordered_features)
            # if self.debug:
            #     print(f'玩家{i} 的相对距离:{player_i_rel_pos}')
            #     print(f'玩家{i} 的绝对位置:{player_i_abs_pos}')

        if self.debug and len(tasks) > 1:
            print(f'当前任务特征: 任务{flatencurrent_goal},任务剩余时间{tasktime}')

        # return sparse_r, shaped_r, tasksequence, self.timercount > self.episode_limit, ordered_features, avaliable_actions
        return ordered_features

    def is_done(self):
        return self.timercount >= self.episode_limit
    
    def render(self, mode='human', close=False):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        self.game.window.fill((255, 250, 205))
        self.game.all_sprites.draw(self.game.window)
        
        
        # 绘制reward
        self.game.draw_reward()
        
        pygame.display.update()
        pygame.display.flip()
        ## 控制游戏帧率：一秒跑多少timestep
        self.clock.tick(self.fps)
        
    def pltsuccessrate(self,filename='successrate'):
        import matplotlib.pyplot as plt
        # 计算每个人的成功率
        success_rates = []
        for person_data in self.taskcount:
            success_rate = {key: (person_data[key] / self.alltaskcount[key]) if self.alltaskcount[key] != 0 else 0 for key in self.alltaskcount}
            success_rates.append(success_rate)

        # 绘制柱状图
        fig, axs = plt.subplots(1, len(success_rates), figsize=(18, 8), sharey=True)

        for i, ax in enumerate(axs):
            keys = list(success_rates[i].keys())
            values = list(success_rates[i].values())
            ax.bar(keys, values)
            ax.set_title(f'Person {i + 1} Success Rate')
            ax.set_xticklabels(keys, rotation=45, ha='right')
        plt.savefig(f'results/{filename}.png', dpi=300)  # 'dpi'参数设置图像的分辨率
        # 关闭图表以避免在保存后再次显示
        plt.close()
        
    def pltheatmappng(self,fiename='heatmap'):
        import matplotlib.pyplot as plt
        for i in range(self.n_agents):
            # 使用imshow()函数制热力图
            plt.imshow(self.heatmap[i], cmap='hot', interpolation='nearest')
            # 添加颜色条
            plt.colorbar()
            plt.savefig(f'results/{fiename}{i}.png', dpi=300)  # 'dpi'参数设置图像的分辨率
            # 关闭图表以避免在保存后再次显示
            plt.close()
            
    def close(self):
        # 关闭 Pygame
        pygame.quit()

    def seed(self, seed=None):
        # 设置随机数生成器的种子
        return np.random(seed=seed)


