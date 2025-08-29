import numpy as np
import gym
import pygame
import os, sys
import json
import threading
import time
from queue import Queue, Empty
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict
from typing import Union, Dict, Tuple, List, Optional, Any
from gymnasium.spaces import flatdim
from envs.overcook_main import MainGame
from envs.overcook_mdp import ComplexOvercookedGridworld, TASK_VALUE
from envs.overcook_class import ONEBLOCK, Table, Action, Direction, \
    TASK_FINISH_EVENT, OUT_SUPPLY_EVENT, OUT_DISH_EVENT, GET_MATERIAL_EVENT, \
        GET_DISH_EVENT, MADE_NEWTHING_EVENT, BEGINCUTTING_EVENT, CUTTINGDOWN_EVENT, \
            BEGINCOOKING_EVENT, COOKINGDOWN_EVENT, COOKINGOUT_EVENT, TRY_NEWTHING_EVENT,\
                  PUTTHING_DISH_EVENT, TRASH_EVENT
from envs.overcook_class import TrashTable, Pot, SupplyTable
import ipdb

# 获取当前文件的目录
current_dir = os.path.dirname(__file__)
maps_path = os.path.join(current_dir, 'maps.json')
with open(maps_path, 'r', encoding='utf-8') as file:
    maps = json.load(file)

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
                 lossless_obs=False,
                 fps=60):

        # 安全初始化pygame
        if not pygame.get_init():
            pygame.init()
        
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
        self.lossless_obs = lossless_obs
        self.debug = debug
        self.episode_limit = 600
        self.fps = fps
        self.game_over = False
        self.timercount = 0  # 自定义的计时器
        self.showkey()
        self.ifrender = ifrender
        
        # 初始化MDP
        self.mdp = ComplexOvercookedGridworld(map_name=map_name)
        self.terrain = maps[self.map_name]['layout']
        self.terrain_mtx = self.mdp.terrain_mtx
        self.height = self.mdp.height
        self.width = self.mdp.width
        
        # 初始化游戏实例
        self.game = MainGame(map_name=self.map_name, 
                             ifrender=self.ifrender)
        # 初始化状态字典，用于MDP处理 - 直接引用游戏对象
        self.state = {
            "tables": self.game.tables,
            "pots": self.game.pots,
            "cuttingtables": self.game.cuttingtables,
            "players": self.game.playergroup,
            "cointable": self.game.Cointable,
            "timercount": 0
        }

        self.obs_shape = self.dummy_reset()[0].shape[0]
        self.reset_featurize_type(obs_shape=self.obs_shape)
        
        if self.ifrender:
            self.clock = pygame.time.Clock()
        
        self.t = 0
        self._max_episode_steps = 600

    # def get_share_observation(self, nobs:List[np.ndarray]) -> np.ndarray:
    #     """
    #     share observation 和 state 没什么区别
    #     """
    #     share_obs = np.stack(nobs, axis=0)
    #     return share_obs

    # def initialize_game(self):
    #     """初始化游戏环境和MDP状态"""
    #     self.get_need_cutting = 0
    #     self.get_need_cooking = 0
    #     self.get_need_synthesis = 0

    #     # 只在需要渲染时初始化时钟
    #     if self.ifrender:
    #         self.clock = pygame.time.Clock()

    #     # 初始化游戏实例
    #     self.game = MainGame(map_name=self.map_name, 
    #                          ifrender=self.ifrender)
        
    #     # 初始化任务计数器 - 使用字典推导式优化
    #     self.taskcount = [{key:0 for key in self.TASK_MENU} for _ in range(self.n_agents)]
    #     self.alltaskcount = {key:0 for key in self.TASK_MENU} #用来计算总的任务变更
        
    #     # 批量更新任务计数
    #     for taskname in self.game.taskmenu:
    #         self.alltaskcount[taskname] += 1
            
    #     # 初始化物品计数器 - 使用字典推导式优化
    #     self.matiral_count = {key: 0 for key in self.itemdict.keys()}
        
    #     # 预先计算玩家字典，避免重复创建
    #     self.playerdic = {'a':0,'b':1,'c':2,'d':3}
        
    def dummy_reset(self):
        self.timercount = 0
        # 确保状态字典正确设置
        self.state["timercount"] = self.timercount
        
        nobs = self.get_obs_grid() if self.lossless_obs else self.get_obs()
        return nobs
    
    def reset(self) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
        
        self.game.init_maps()
        self.game.init_tasks()
        self.game.init_all_sprites()
        
        
        """初始化游戏环境和MDP状态"""
        self.get_need_cutting = 0
        self.get_need_cooking = 0
        self.get_need_synthesis = 0
        
        # 初始化任务计数器 - 使用字典推导式优化
        self.taskcount = [{key:0 for key in self.TASK_MENU} for _ in range(self.n_agents)]
        self.alltaskcount = {key:0 for key in self.TASK_MENU}  #用来计算总的任务变更
        
        # 批量更新任务计数
        for taskname in self.game.taskmenu:
            self.alltaskcount[taskname] += 1
            
        # 初始化物品计数器 - 使用字典推导式优化
        self.matiral_count = {key: 0 for key in self.itemdict.keys()}
        
        # 预先计算玩家字典，避免重复创建
        self.playerdic = {'a':0,'b':1,'c':2,'d':3}
        
        self.timercount = 0
        
        # 状态字典已在initialize_game中设置，只需更新timercount
        self.state["timercount"] = self.timercount
        
        # 获取MDP特征
        self.mdp_features = self.mdp.get_mdp_features(self.state)
        
        # 获取观察值 - 根据配置选择观察方式
        nobs = self.get_obs_grid() if self.lossless_obs else self.get_obs()

        # 获取可用动作和共享观察
        available_actions = self.get_avail_actions()
        # share_obs = self.get_share_observation(nobs)
        
        # 重置奖励字典 - 使用numpy数组直接初始化
        self.episode_reward_dict = {
            "cumulative_sparse_rewards": np.zeros(1, dtype=np.float32),
            "cumulative_shaped_rewards": np.zeros(1, dtype=np.float32),
            "cumulative_rewards": np.zeros(1, dtype=np.float32)
        }
        
        return nobs, None, available_actions

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
        # 使用列表推导式优化循环
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]
        
    def get_avail_agent_actions(self, agent_id:int)->np.ndarray:
        # 初始化可用动作数组 - 默认所有动作不可用
        avaliable_actions = [0] * 6
        player = self.game.playergroup[agent_id]
        
        # 停留动作始终可用
        avaliable_actions[Action.STAY] = 1
        
        # 切菜时特殊处理 - 只有停留和交互可用
        if player.cutting:
            avaliable_actions[Action.INTERACT] = 1
            return avaliable_actions

        # 预先创建其他玩家列表 - 只创建一次
        tempplayergroup = [self.game.playergroup[j] for j in range(self.n_agents) if j != agent_id]

        # 检测移动动作是否可用
        for action in range(1, 5):
            if player.availaction(action, self.game.walls, tempplayergroup):
                avaliable_actions[action] = 1

        # 检测交互是否合法 - 优化循环结构
        interact_available = False
        
        # 检查是否可以与收银台交互
        if self.game.Cointable.availbeinter(player):
            interact_available = True
        else:
            # 检查是否可以与任何桌子交互
            for table in self.game.tables:
                if table.availbeinter(player):
                    interact_available = True
                    break
        
        # 设置交互动作可用性
        if interact_available:
            avaliable_actions[Action.INTERACT] = 1
            
        return avaliable_actions

    def step(self, action_n: List[int]) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], np.ndarray]:

        # 更新时间计数器
        self.timercount += 1
        
        # 预先创建临时玩家组列表，避免在循环中重复创建
        temp_player_groups = []
        for i in range(self.n_agents):
            temp_group = [self.game.playergroup[j] for j in range(self.n_agents) if j != i]
            temp_player_groups.append(temp_group)
        
        # 执行动作并更新游戏状态 - 批量处理以减少函数调用
        for i in range(self.n_agents):
            player = self.game.playergroup[i]
            is_interact = action_n[i] == Action.INTERACT
            
            # 批量更新tables和Cointable
            self.game.tables.update(player, is_interact, self.timercount)
            self.game.Cointable.update(player, is_interact, self.game.taskmenu)
            
            # 非交互动作时才更新玩家位置，减少不必要的计算
            if not is_interact:
                player.update(action=action_n[i], 
                             pengzhuang=self.game.walls, 
                             player=temp_player_groups[i])
        
        # 更新状态字典 - 一次性更新所有状态
        self.state = {
            "tables": self.game.tables,
            "pots": self.game.pots,
            "cuttingtables": self.game.cuttingtables,
            "players": self.game.playergroup,
            "cointable": self.game.Cointable,
            "timercount": self.timercount
        }
        
        # 获取MDP特征
        self.mdp_features = self.mdp.get_mdp_features(self.state)
        self.game.timercount.update(self.timercount)
        
        # 计算奖励
        sparse_reward, shaped_reward = self.calculate_reward()
        reward = sparse_reward + shaped_reward
        if self.debug:
            print('sparse_reward:', sparse_reward)
            print('shaped_reward:', shaped_reward)

        done = self.is_done()

        # 仅在需要时渲染，避免不必要的图形处理
        if self.ifrender:
            self.render()
            
        # 获取可用动作
        available_actions = self.get_avail_actions()
        self.t += 1

        # 构建信息字典 - 只包含必要信息
        infos = {
            'shaped_r': shaped_reward,
            'sparse_r': sparse_reward,
            'reward': reward,
        }
        
        self._update_reward_dict(infos)

        # 仅在回合结束时添加额外信息，减少不必要的字典操作
        if done:  
            if float(self.episode_reward_dict["cumulative_rewards"]) > 800: 
                print('异常分数', self.taskcount, self.timercount)    
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

        # 获取观察和共享观察 - 只在需要时计算
        nobs = self.get_obs()
        # share_obs = self.get_share_observation(nobs)
        dones = [done] * self.n_agents
        rewards = [reward] * self.n_agents
        
        return nobs, None, rewards, dones, infos, available_actions

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
        
    def _handle_task_finish_event(self, event):
        """处理任务完成事件"""
        sparse_reward = 0.0
        
        task_value = self.TASK_MENU[event.action]
        reward_value = task_value[0] if isinstance(task_value, list) else task_value
        
        self.game.NOWCOIN += reward_value
        self.taskcount[self.playerdic[event.player]][event.action] += 1
        sparse_reward += reward_value
        
        # 优化任务匹配逻辑
        self._update_task_after_completion(event.action)
        
        return sparse_reward, 0.0
    
    def _update_task_after_completion(self, action):
        """任务完成后更新任务列表"""
        named_tasks = []
        for task in self.game.task_sprites:
            if task.task == action:  # 去找到agent完成了哪个任务优先筛检剩余时间少的任务
                named_tasks.append(task)  # 有可能为空，为空是因为前面倒计时重新更新了
        if named_tasks:
            min_task = min(named_tasks, key=lambda task: task.remaining_time)
            min_task.newtask(self.timercount)
            self.game.taskmenu[self.game.task_dict[min_task]] = min_task.task
            self.alltaskcount[min_task.task] += 1
        else:
            self.game.NOWCOIN += 1  # 代表之前倒计时重新了
    
    def _update_coin_display(self):
        """更新金币显示"""
        th, hu, te, on = digitize(self.game.NOWCOIN)
        self.game.num1.set_num(th)
        self.game.num2.set_num(hu)
        self.game.num3.set_num(te)
        self.game.num4.set_num(on)
    
    def calculate_reward(self) -> Tuple[float, float]:
        """优化后的奖励计算函数"""
        tasksequence = []
        sparse_reward = 0.0
        shaped_reward = 0.0
        
        self.game.task_sprites.update(self.timercount)
        
        events = pygame.event.get()
        
        taskfinished = [False for _ in range(len(self.game.task_sprites))]  # 当前这个时间步，有没有任务被完成了
        for event in events:
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
                task_sparse_reward, task_shaped_reward = self._handle_task_finish_event(event)
                sparse_reward += task_sparse_reward
                shaped_reward += task_shaped_reward
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
                    if event.item in task.task:  # TODO: 目前是简单的字符串判定逻辑，例如：AClemon in AClemoncookedfish,后期考虑修改
                        shaped_reward += self.reward_shaping_params['get_need_cutting']
                        self.get_need_cutting += 1
                        tasksequence.append(f"cut the {event.item} completely")
                        if self.debug:
                            print(f"把{event.item}切好了，奖励{shaped_reward}")
                    else:
                        shaped_reward -= self.reward_shaping_params['get_need_cutting'] # 切出当前任务中不需要的东西，惩罚
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
                    else:
                        shaped_reward -= self.reward_shaping_params['get_need_cooking'] # 烹饪出当前任务中不需要的东西，惩罚

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
        
        # 优化数字显示更新
        self._update_coin_display()

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
    

    def get_obs(self) -> List[np.ndarray]:
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

        # def concat_dicts(a:Dict, b:Dict)->Dict:
        #     return {**a, **b}
        
        def ManhattanDis(relativepos:Tuple[int, int])->int:
            return abs(relativepos[0]) + abs(relativepos[1])
        
        def rel_xy(p:pygame.sprite.Sprite, 
                   obj:pygame.sprite.Sprite)-> List[int]:
            return [(p.rect.x-obj.rect.x)//80, 
                    (p.rect.y-obj.rect.y)//80]  # 使用整除代替除法
        
        players:List[pygame.sprite.Sprite] = self.game.playergroup

        # 预先计算玩家位置、方向和物品，避免重复计算
        player_positions = [(p.rect.x, p.rect.y) for p in players]
        player_directions = [p.direction for p in players]
        player_items = [p.item for p in players]
        player_dishes = [p.dish for p in players]

        # 初始化数据结构
        playab_cointables = [[] for _ in range(self.n_agents)]
        playab_closet_item_pos = [[10e9] * (len(self.itemdict)) * 2 for _ in range(self.n_agents)]
        
        # 计算与cointable的相对距离并处理玩家手持物品
        for i, player in enumerate(players):
            playab_cointables[i] = rel_xy(player, self.game.Cointable)
            
            if player_items[i]:
                itemindex = (self.itemdict[player_items[i]] - 1) * 2
                playab_closet_item_pos[i][itemindex:itemindex + 2] = [0, 0]
            elif player_dishes[i]:# TODO：这里逻辑有问题
                dishindex = (self.itemdict[player.dish] - 1) * 2
                playab_closet_item_pos[i][dishindex:dishindex + 2] = [0, 0]
        # 处理桌子和空桌子
        emptypalcenum = 0
        tablenum = 0
        playab_tables = [[] for _ in range(self.n_agents)]
        playab_closet_empty_table_pos = [[10e9, 10e9] for _ in range(self.n_agents)]
        
        # 批量处理tables
        for temp in self.game.tables:
            if temp.item:
                item_index = (self.itemdict[temp.item] - 1) * 2
                for i, player in enumerate(players):
                    temppos = rel_xy(player, temp)
                    if ManhattanDis(temppos) < ManhattanDis(playab_closet_item_pos[i][item_index:item_index + 2]):
                        playab_closet_item_pos[i][item_index:item_index + 2] = temppos
            
            if isinstance(temp, Table):
                if temp.dish: # 在考虑dish距离，和上一段是一致的，只是dish单独有个属性
                    dish_index = (self.itemdict[temp.dish] - 1) * 2
                    for i, player in enumerate(players):
                        temppos = rel_xy(player, temp)
                        if ManhattanDis(temppos) < ManhattanDis(playab_closet_item_pos[i][dish_index:dish_index + 2]):
                            playab_closet_item_pos[i][dish_index:dish_index + 2] = temppos
                if not temp.item:
                    emptypalcenum += 1
                    for i, player in enumerate(players):
                        temppos = rel_xy(player, temp)
                        if ManhattanDis(temppos) < ManhattanDis(playab_closet_empty_table_pos[i][0:2]):
                            playab_closet_empty_table_pos[i][0:2] = temppos
                continue
                
            for i, player in enumerate(players):
                tablenum += 1
                playab_tables[i] += rel_xy(player, temp)
        
        # 处理无效值
        for i in range(self.n_agents):
            if playab_closet_empty_table_pos[i][0] > 100000:
                playab_closet_empty_table_pos[i] = [0, 0]

            for j in range(len(playab_closet_item_pos[i])):
                if playab_closet_item_pos[i][j] > 100000:
                    playab_closet_item_pos[i][j] = 0

        # 批量处理锅和案板
        playab_pots_pos = [[10e9, 10e9]*len(self.game.pots) for _ in range(self.n_agents)]
        pots_state = [0, 0]*len(self.game.pots)
        pots_remaining_time = [-1]*len(self.game.pots)
        pots_lang = []
        
        # 优化锅的处理
        for p, pot in enumerate(self.game.pots):
            pot_index = p*2
            for i, player in enumerate(players):
                potpos = rel_xy(player, pot)
                playab_pots_pos[i][pot_index:pot_index+2] = potpos
            
            # 批量设置锅状态
            if pot.is_empty:
                pots_state[pot_index:pot_index+2] = [0, 0]
                pots_lang.append(f"pot{p} is empty")
            elif pot.is_cooking:
                pots_state[pot_index:pot_index+2] = [0, 1]
                pot_item = pot.item.replace("raw", "cooked")
                pots_lang.append(f"pot{p} is cooking, the {pot_item} will be ready in {pot.remaining_time} timesteps")
            elif pot.is_ready:
                pots_state[pot_index:pot_index+2] = [1, 0]
                pots_lang.append(f"{pot.item} in pot{p} is ready")

            pots_remaining_time[p] = pot.remaining_time // 10 if pot.remaining_time >=0 else -1

        self.state.update({"pot": pots_lang})

        # 优化案板处理
        cuttingtables_lang = []
        playab_cuttingtables_pos = [[10e9, 10e9]*len(self.game.cuttingtables) for _ in range(self.n_agents)]
        cuttingtables_state = [0,0]*len(self.game.cuttingtables)
        
        for p, cuttingtable in enumerate(self.game.cuttingtables):
            ct_index = p*2
            for i, player in enumerate(players):
                cuttingtablepos = rel_xy(player, cuttingtable)
                playab_cuttingtables_pos[i][ct_index:ct_index+2] = cuttingtablepos
            
            # 批量设置案板状态
            if cuttingtable.is_empty:
                cuttingtables_state[ct_index:ct_index+2] = [0, 0]
                cuttingtables_lang.append(f"cutting_table{p} is empty")
            elif cuttingtable.is_cutting:
                cuttingtables_state[ct_index:ct_index+2] = [0, 1]
                cuttingtables_lang.append(f"cutting_table{p} is occupied by {cuttingtable.item}")
            elif cuttingtable.is_ready:
                cuttingtables_state[ct_index:ct_index+2] = [1, 0]
                cuttingtables_lang.append(f"{cuttingtable.item} on cutting_table{p} is ready")

        self.state.update({"cutting_table": cuttingtables_lang})

        # 简化任务目标和特征处理
        nowtime = self.timercount
        self.state.update({"timestep": nowtime})

        tasks = []
        task_name = []
        tasktime = []
        for task in self.game.task_sprites:
            tasks.append(task)
            task_name.append(task.task)
            tasktime.append(task.timer - (nowtime - task.start_time))
            
        self.state.update({"task": task_name, "tasktime": tasktime})

        # 优化任务目标处理
        current_goal = [np.eye(len(self.taskdict))[self.taskdict[task.task]].tolist() for task in tasks]
        
        # 简化任务特征处理
        if len(tasks) > 1:
            flatencurrent_goal = [item for sublist in current_goal for item in sublist]
            task_feature = [i // 100 for i in tasktime] + flatencurrent_goal  # 使用整除
        else:
            task_feature = []

        # 优化玩家特征处理
        player_absolute_positions = []
        player_relative_positions = []
        player_features = []
        hold_objects = defaultdict(list)
        ori = []
        pos = []
        
        # 预先创建碰撞检测精灵
        rect_sprite = pygame.sprite.Sprite()
        
        for i, player in enumerate(players):
            # 玩家朝向 - 使用预计算的方向
            orientation_idx = Direction.DIRECTION2INDEX[player_directions[i]]
            orientation = np.eye(len(Direction.DIRECTION2INDEX))[orientation_idx].tolist()
            ori.append(tuple(player_directions[i]))
            
            # 玩家手持物品 - 使用预计算的物品
            if player_items[i]:
                hold_obj = np.eye(len(self.itemdict))[self.itemdict[player_items[i]]-1].tolist()
                hold_objects[i].append(player_items[i])
                if player_dishes[i]:
                    hold_obj[self.itemdict[player_items[i]]-1] = 1.0
                    hold_obj[self.itemdict["dish"]-1] = 1.0
                    hold_objects[i].append("dish")
            else:
                if player_dishes[i]:
                    hold_obj = np.eye(len(self.itemdict))[self.itemdict["dish"]-1].tolist()
                    hold_objects[i].append("dish")
                else:
                    hold_obj = np.zeros(len(self.itemdict)).tolist()
                    hold_objects[i].append("nothing")

            # 优化碰撞检测
            collide = []
            for direction in list(Direction.directions.values()):
                rect_sprite.rect = player.rect.move(direction[0] * ONEBLOCK // 2, direction[1] * ONEBLOCK // 2)
                is_block = pygame.sprite.spritecollide(rect_sprite, self.game.tables, False)
                collide.append(int(bool(is_block)))
            if self.debug:
                print('*' * 10)
                print(f'玩家{i} player_feature:')
                print(f'面向：{orientation} {self.directionzhongwen[str(player.direction)]}')
                print(f'手持物品: {hold_obj}', player.item if player.item else '空手')
                print(f'四面是否有墙壁：{collide}')
                print(f'和coin table的相对距离：{playab_cointables[i]}')
                print(f'和pots的相对距离：{playab_pots_pos[i]}')
                print(f'和cuttingtables的相对距离：{playab_pots_pos[i]}')
                print(f'和每个功能桌子（非table、cointable）的相对距离：{playab_tables[i]}')
                print("\n")
            # 合并特征
            player_features.append(
                np.array(orientation + hold_obj + collide + playab_cointables[i] + 
                        playab_pots_pos[i] + playab_cuttingtables_pos[i] + 
                        playab_closet_item_pos[i])
            )

            # 计算位置
            player_x, player_y = player_positions[i]
            player_absolute_positions.append(np.array([player_x // 80, player_y // 80]))
            pos.append((player_x // 80, player_y // 80))
            
            # 计算相对位置
            tempdis = []
            for j in range(self.n_agents):
                if j != i:
                    tempdis += rel_xy(players[i], self.game.playergroup[j])
            player_relative_positions.append(np.array(tempdis))

        # 更新状态
        self.state.update({"player": hold_objects})
        self.state.update({"players_pos_and_or": tuple(zip(*[pos, ori]))})
        
        # 构建最终特征
        ordered_features = []
        for i in range(self.n_agents):
            # 预先计算其他玩家特征
            other_player_features = np.concatenate([player_features[j] for j in range(self.n_agents) if j != i])
            
            # 一次性合并所有特征
            player_i_ordered_features = np.squeeze(np.concatenate([
                player_features[i],
                other_player_features,
                player_relative_positions[i],
                player_absolute_positions[i],
                pots_state,
                pots_remaining_time,
                cuttingtables_state,
                task_feature
            ]))
            
            ordered_features.append(player_i_ordered_features)
            # if self.debug:
            #     print(f'玩家{i} 的相对距离:{player_i_rel_pos}')
            #     print(f'玩家{i} 的绝对位置:{player_i_abs_pos}')

        if self.debug and len(tasks) > 1:
            print(f'当前任务特征: 任务{flatencurrent_goal},任务剩余时间{tasktime}')

        # return sparse_r, shaped_r, tasksequence, self.timercount > self.episode_limit, ordered_features, avaliable_actions
        return ordered_features

    def get_obs_grid(self) -> List[np.ndarray]:
        """
        生成h*w*n_channels格式的观测，其中h是地图高度，w是地图宽度，n_channels是特征通道数
        
        Returns:
            grid_obs (list[np.ndarray]): 每个玩家的网格观测，形状为 (h, w, n_channels)
            
        特征通道包括:
        channel1：矩阵中，所有锅的位置为1，其余位置为0 
        channel2：矩阵中，所有案板的位置为1，其余位置为0 
        channel3：矩阵中，所有coin table的位置为1，其余位置为0 
        channel4：矩阵中，所有垃圾桶的位置为1，其余位置为0 
        channel5：矩阵中，所有盘子提供处的位置为1，其余位置为0 
        channel6：矩阵中，所有生鱼提供处的位置为1，其余位置为0 
        channel7：矩阵中，所有未切柠檬提供处的位置为1，其余位置为0 
        channel8：矩阵中，所有汉堡提供处的位置为1，其余位置为0 
        channel9：矩阵中，所有生肉提供处的位置为1，其余位置为0 
        channel10：矩阵中，玩家1当前位置为1，其余位置为0 
        channel11：矩阵中，玩家2当前位置为1，其余位置为0 
        channel12：矩阵中，玩家1面朝方向分别为0，1，2，3，其余位置为0 
        channel13：矩阵中，玩家2面朝方向分别为0，1，2，3，其余位置为0 
        channel14-15：锅的状态，只用编码is_cooking和is_ready两种，如果是则为1，其余位置为0 
        channel16: 锅中烹饪的剩余时间，没有就是-1， 
        channel17-18：案板的状态，只用编码is_cutting和is_ready两种，如果是则为1，其余位置为0
        """
        # 固定为18个通道
        n_channels = 18
        
        # 准备当前状态
        state = {
            "tables": self.game.tables,
            "pots": self.game.pots,
            "cuttingtables": self.game.cuttingtables,
            "players": self.game.playergroup,
            "cointable": self.game.Cointable
        }
        
        # 获取MDP特征
        mdp_features = self.mdp.get_mdp_features(state)
        
        # 预先获取所有位置信息，避免重复调用
        pot_locations = self.mdp.get_pot_locations()
        cutting_table_locations = self.mdp.get_cutting_table_locations()
        trash_locations = self.mdp.get_trash_locations()
        dish_dispenser_locations = self.mdp.get_dish_dispenser_locations()
        rawfish_locations = self.mdp.get_rawfish_dispenser_locations()
        lemon_locations = self.mdp.get_lemon_dispenser_locations()
        hamburger_locations = self.mdp.get_hamburger_dispenser_locations()
        rawbeef_locations = self.mdp.get_rawbeef_dispenser_locations()
        player_positions = self.mdp.get_player_positions(state)
        player_directions = self.mdp.get_player_directions(state)
        
        # 预先计算锅和案板的位置和状态
        pot_positions = [(pot.rect.x // 80, pot.rect.y // 80) for pot in self.game.pots]
        pot_cooking_states = [pot.is_cooking for pot in self.game.pots]
        pot_ready_states = [pot.is_ready for pot in self.game.pots]
        pot_remaining_times = [pot.remaining_time if hasattr(pot, 'remaining_time') else -1 for pot in self.game.pots]
        pot_cooking_times = [pot.cookingtime if hasattr(pot, 'cookingtime') else 1 for pot in self.game.pots]
        
        cutting_table_positions = [(ct.rect.x // 80, ct.rect.y // 80) for ct in self.game.cuttingtables]
        cutting_table_cutting_states = [ct.is_cutting for ct in self.game.cuttingtables]
        cutting_table_ready_states = [ct.is_ready for ct in self.game.cuttingtables]
        
        # 预先计算coin table位置
        coin_x, coin_y = self.game.Cointable.rect.x // 80, self.game.Cointable.rect.y // 80
        
        grid_obs = []
        
        for agent_id in range(self.n_agents):
            # 初始化网格观测 - 一次性分配内存
            obs = np.zeros((self.height, self.width, n_channels), dtype=np.float32)
            
            # 批量填充静态位置信息
            self._fill_locations(obs, pot_locations, 0)
            self._fill_locations(obs, cutting_table_locations, 1)
            
            # 填充coin table位置
            if 0 <= coin_y < self.height and 0 <= coin_x < self.width:
                obs[coin_y, coin_x, 2] = 1.0
            
            # 批量填充其他静态位置
            self._fill_locations(obs, trash_locations, 3)
            self._fill_locations(obs, dish_dispenser_locations, 4)
            self._fill_locations(obs, rawfish_locations, 5)
            self._fill_locations(obs, lemon_locations, 6)
            self._fill_locations(obs, hamburger_locations, 7)
            self._fill_locations(obs, rawbeef_locations, 8)
            
            # 填充玩家位置和方向 - 只处理前两个玩家
            for player_id, player_pos in enumerate(player_positions):
                if player_id < 2:  # 只处理前两个玩家
                    player_x, player_y = player_pos
                    if 0 <= player_y < self.height and 0 <= player_x < self.width:
                        # 玩家位置
                        obs[player_y, player_x, 9 + player_id] = 1.0
                        # 玩家方向
                        direction_idx = Direction.DIRECTION2INDEX[player_directions[player_id]]
                        obs[player_y, player_x, 11 + player_id] = direction_idx
            
            # 批量处理锅的状态
            for i, (pot_x, pot_y) in enumerate(pot_positions):
                if 0 <= pot_y < self.height and 0 <= pot_x < self.width:
                    # 一次性设置锅的所有状态
                    if pot_cooking_states[i]:
                        obs[pot_y, pot_x, 13] = 1.0
                    if pot_ready_states[i]:
                        obs[pot_y, pot_x, 14] = 1.0
                    
                    # 设置剩余时间
                    remaining_time = pot_remaining_times[i]
                    cooking_time = pot_cooking_times[i]
                    if pot_cooking_states[i] and remaining_time > 0:
                        obs[pot_y, pot_x, 15] = remaining_time / cooking_time
                    else:
                        obs[pot_y, pot_x, 15] = -1
            
            # 批量处理案板状态
            for i, (ct_x, ct_y) in enumerate(cutting_table_positions):
                if 0 <= ct_y < self.height and 0 <= ct_x < self.width:
                    # 一次性设置案板的所有状态
                    if cutting_table_cutting_states[i]:
                        obs[ct_y, ct_x, 16] = 1.0
                    if cutting_table_ready_states[i]:
                        obs[ct_y, ct_x, 17] = 1.0
            
            grid_obs.append(obs)
        
        return grid_obs
        
    def _fill_locations(self, obs, locations, channel):
        """辅助方法：批量填充网格位置"""
        for x, y in locations:
            if 0 <= y < self.height and 0 <= x < self.width:
                obs[y, x, channel] = 1.0

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
        
    def close(self):
        pygame.quit()

    def seed(self, seed=None):
        # 设置随机数生成器的种子
        return np.random(seed=seed)