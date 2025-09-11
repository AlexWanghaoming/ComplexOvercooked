import itertools, copy
import numpy as np
from functools import reduce
from collections import defaultdict
import os
import json
import pygame
from typing import List, Tuple, Optional, Dict, Union
from envs.overcook_class import Table, Direction, Action, TrashTable, SupplyTable, Pot

# 获取当前文件的目录
current_dir = os.path.dirname(__file__)
maps_path = os.path.join(current_dir, 'maps.json')
with open(maps_path, 'r', encoding='utf-8') as file:
    maps = json.load(file)

# 合成菜品的得分
TASK_VALUE = {'AClemoncookedfish': defaultdict(float,{'AClemoncookedfish': 7, 
                                                      'AClemon': 0, 
                                                      'cookedfish': 0, 
                                                      'BClemon': 0,
                                                      'rawfish': 0}),
              'cookedfish': defaultdict(float, {'cookedfish': 3, 
                                                'rawfish': 0}),
              'ACtomatocookedbeefhamburger': defaultdict(float, {'ACtomatocookedbeefhamburger': 8.0,
                                                                 'ACtomatohamburger': 0,
                                                                 'BCtomato': 0,
                                                                 'ACtomato': 0,
                                                                 'cookedbeefhamburger': 0,
                                                                 'cookedbeef': 0,
                                                                 'rawbeef': 0,
                                                                 'hamburger': 0}),
              'cookedbeefhamburger': defaultdict(float, {'cookedbeefhamburger': 5.0, 
                                                         'cookedbeef': 0, 
                                                         'rawbeef': 0,
                                                         'hamburger': 0})}

class ComplexOvercookedGridworld(object):
    """
    ComplexOvercookedGridworld类定义了Overcooked游戏的核心MDP逻辑，包括地形、物品位置、交互逻辑等。
    该类负责管理游戏状态和提供访问游戏元素的方法，但不直接处理渲染和环境接口。
    """

    def __init__(self, map_name):
        """
        初始化ComplexOvercookedGridworld对象
        
        Args:
            map_name (str): 地图名称，用于加载对应的地图配置
        """
        self.map_name = map_name
        self.terrain = maps[map_name]['layout']
        self.terrain_mtx = self.convert_layout_to_2d_list()
        self.height = len(self.terrain_mtx)
        self.width = len(self.terrain_mtx[0])
        self.shape = (self.width, self.height)
        self.terrain_pos_dict = self._get_terrain_type_pos_dict()
        self.TASK_MENU = maps[map_name]['task']
        self.n_agents = maps[map_name]['players']
        self.TASKNUM = maps[map_name]['tasknum']
        self.ITEMS = maps[map_name]['items']
    # X: 桌子， F：生鱼供应处，B：生牛肉供应处，H：汉堡包供应处，M：番茄供应处，D：盘子供应处，L：柠檬供应处，T：垃圾桶，E：送菜口，C：锅，U：案板
    def convert_layout_to_2d_list(self) -> List[List[str]]:
        """
        将地图布局转换为二维列表
        
        Returns:
            List[List[str]]: 二维地图布局，每个元素是一个字符表示地形类型
        """
        ignore_chars = ['_']
        terrain_mtx = [[char for char in row if char not in  ignore_chars] for row in self.terrain]
        terrain_mtx = [row for row in terrain_mtx if row]  # Remove empty rows
        for y, row in enumerate(terrain_mtx):
            for x, c in enumerate(row):
                if c in ['1', '2', '3', '4']:  # 玩家起始位置标记为空地
                    terrain_mtx[y][x] = ' '

        return terrain_mtx
    
    def _get_terrain_type_pos_dict(self) -> Dict[str, List[Tuple[int, int]]]:
        """
        获取地图中各种地形类型的位置字典
        
        Returns:
            Dict[str, List[Tuple[int, int]]]: 地形类型到位置列表的映射
        """
        pos_dict = defaultdict(list)
        for y, terrain_row in enumerate(self.terrain_mtx):
            for x, terrain_type in enumerate(terrain_row):
                pos_dict[terrain_type].append((x, y))
        return pos_dict
    
    def get_player_positions(self, state) -> List[Tuple[int, int]]:
        """
        获取所有玩家的位置
        
        Args:
            state: 游戏当前状态
            
        Returns:
            List[Tuple[int, int]]: 玩家位置列表
        """
        positions = []
        for player in state["players"]:
            positions.append((player.rect.x // 80, player.rect.y // 80 - 1))
        return positions
    
    def get_player_directions(self, state) -> List[Tuple[int, int]]:
        """
        获取所有玩家的朝向
        
        Args:
            state: 游戏当前状态
            
        Returns:
            List[Tuple[int, int]]: 玩家朝向列表
        """
        directions = []
        for player in state["players"]:
            directions.append(player.direction)
        return directions
    
    def get_player_hold_objects(self, state) -> List[str]:
        """
        获取所有玩家手持的物品
        
        Args:
            state: 游戏当前状态
            
        Returns:
            List[str]: 玩家手持物品列表，如果没有则为None
        """
        hold_objects = []
        for player in state["players"]:
            hold_objects.append(player.item)
        return hold_objects

    def get_counter_objects_dict(self, state) -> Dict[str, List[Tuple[int, int]]]:
        """
        获取桌子上的物品及其位置

        Args:
            state: 游戏当前状态
            
        Returns:
            Dict[str, List[Tuple[int, int]]]: 物品类型到位置列表的映射
        """
        counter_objects = defaultdict(list)
        for table in state["tables"]:
            if isinstance(table, Table) and table.item:
                counter_objects[table.item].append((table.rect.x//80, table.rect.y//80-1))
        return counter_objects
    
    def get_counter_objects_pos(self, state) -> Dict[Tuple[int, int], str]:
        """
        获取桌子上的物品位置及其类型

        Args:
            state: 游戏当前状态
            
        Returns:
            Dict[Tuple[int, int], str]: 位置到物品类型的映射
        """
        counter_objects_pos = {}
        for table in state["tables"]:
            if isinstance(table, Table) and table.item:
                counter_objects_pos[(table.rect.x//80, table.rect.y//80-1)] = table.item
        return counter_objects_pos
    
    def get_counter_objects(self, state) -> Tuple[List[str], Dict[Tuple[int, int], str]]:
        """
        获取桌子上的物品及其位置

        Args:
            state: 游戏当前状态
            
        Returns:
            Tuple[List[str], Dict[Tuple[int, int], str]]: 物品类型列表和位置到物品类型的映射
        """
        counter_objects_pos = {}
        for table in state["tables"]:
            if isinstance(table, Table) and table.item:
                counter_objects_pos[(table.rect.x//80, table.rect.y//80-1)] = table.item
        return list(counter_objects_pos.keys()), counter_objects_pos
        
    def get_actions(self, state) -> List[str]:
        """
        获取当前状态下可执行的动作列表
        
        Args:
            state: 游戏当前状态
            
        Returns:
            List[str]: 可执行的动作列表
        """
        # 这里返回所有可能的动作，具体的可行性检查在环境中进行
        return list(Action.ACTION2INDEX.keys())
    
    def get_state_transition(self, state, joint_action):
        """
        获取状态转移函数
        
        Args:
            state: 当前状态
            joint_action: 联合动作
            
        Returns:
            next_state: 下一个状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        # 这个函数在实际环境中实现，MDP类只提供接口
        raise NotImplementedError("状态转移函数应在环境类中实现")
    
    def get_mdp_features(self, state):
        """
        获取MDP特征
        
        Args:
            state: 当前状态
            
        Returns:
            Dict: MDP特征字典
        """
        features = {
            "pot_locations": self.get_pot_locations(),
            "counter_locations": self.get_counter_locations(),
            "cutting_table_locations": self.get_cutting_table_locations(),
            "serving_locations": self.get_serving_locations(),
            "dish_dispenser_locations": self.get_dish_dispenser_locations(),
            "terrain_mtx": self.terrain_mtx,
            "width": self.width,
            "height": self.height
        }
        return features
    
    def get_table_item(self, state) -> Dict[str, List[Tuple[int, int]]]:
        """
        查看桌子上已经有哪些item和dish

        Args:
            state: 游戏当前状态
            
        Returns:
            Dict[str, List[Tuple[int, int]]]: 物品类型到位置列表的映射
        """
        counter_objects = defaultdict(list)
        for table in state["tables"]:
            if isinstance(table, Table) and table.item:
                counter_objects[table.item].append((table.rect.x//80, table.rect.y//80-1))
        return counter_objects
    
    def get_interaction_pos_and_dire(self, goal_pos:List[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        获取与每个物体交互的位置和玩家面朝方向。

        Args:
            goal_pos: 目标物体位置列表
            
        Returns:
            List[Tuple[Tuple[int, int], Tuple[int, int]]]: 交互位置和面朝方向的元组列表
        """
        directions = Direction.directions
        interaction_positions_and_directions = []

        for pos in goal_pos:
            for _, direction_vector in directions.items():
                interaction_pos = (pos[0] - direction_vector[0], pos[1] - direction_vector[1])
                # 检查交互位置是否在地图范围内
                if 0 <= interaction_pos[0] < self.width and 0 <= interaction_pos[1] < self.height:
                    # 检查交互位置是否可到达（不是障碍物）
                    terrain_type = self.get_terrain_type_at_pos(interaction_pos)
                    if terrain_type == ' ':  # 假设 ' ' 表示可到达区域
                        interaction_positions_and_directions.append((interaction_pos, direction_vector))
        return interaction_positions_and_directions

    def get_terrain_type_at_pos(self, pos):
        """
        获取指定位置的地形类型
        
        Args:
            pos: 位置坐标 (x, y)
            
        Returns:
            str: 地形类型字符
        """
        x, y = pos
        return self.terrain_mtx[y][x]
    
    # 以下方法获取各种游戏元素的位置
    
    def get_dish_dispenser_locations(self):
        """
        获取盘子提供处的位置列表
        
        Returns:
            List[Tuple[int, int]]: 盘子提供处位置列表
        """
        return list(self.terrain_pos_dict['D'])

    def get_lemon_dispenser_locations(self):
        """
        获取柠檬提供处的位置列表
        
        Returns:
            List[Tuple[int, int]]: 柠檬提供处位置列表
        """
        return list(self.terrain_pos_dict['L'])
    
    def get_tomato_dispenser_locations(self):
        """
        获取番茄提供处的位置列表
        
        Returns:
            List[Tuple[int, int]]: 番茄提供处位置列表
        """
        return list(self.terrain_pos_dict['M'])
    
    def get_rawfish_dispenser_locations(self):
        """
        获取生鱼提供处的位置列表
        
        Returns:
            List[Tuple[int, int]]: 生鱼提供处位置列表
        """
        return list(self.terrain_pos_dict['F'])
    
    def get_rawbeef_dispenser_locations(self):
        """
        获取生牛肉提供处的位置列表
        
        Returns:
            List[Tuple[int, int]]: 生牛肉提供处位置列表
        """
        return list(self.terrain_pos_dict['B'])
    
    def get_hamburger_dispenser_locations(self):
        """
        获取汉堡提供处的位置列表
        
        Returns:
            List[Tuple[int, int]]: 汉堡提供处位置列表
        """
        return list(self.terrain_pos_dict['H'])
    
    def get_cutting_table_locations(self):
        """
        获取切菜台的位置列表
        
        Returns:
            List[Tuple[int, int]]: 切菜台位置列表
        """
        return list(self.terrain_pos_dict['U'])
    
    def get_serving_locations(self):
        """
        获取送餐口的位置列表
        
        Returns:
            List[Tuple[int, int]]: 送餐口位置列表
        """
        return list(self.terrain_pos_dict['E'])

    def get_pot_locations(self):
        """
        获取锅的位置列表
        
        Returns:
            List[Tuple[int, int]]: 锅位置列表
        """
        return list(self.terrain_pos_dict['C'])

    def get_counter_locations(self):
        """
        获取普通桌子的位置列表
        
        Returns:
            List[Tuple[int, int]]: 普通桌子位置列表
        """
        return list(self.terrain_pos_dict['X'])
        
    def get_trash_locations(self):
        """
        获取垃圾桶的位置列表
        
        Returns:
            List[Tuple[int, int]]: 垃圾桶位置列表
        """
        return list(self.terrain_pos_dict['T'])
        
    def get_actions(self):
        """
        获取可用动作列表
        
        Returns:
            List[int]: 可用动作ID列表
        """
        return [Action.STAY, Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT, Action.INTERACT]
    
    def get_state_transition(self, state, actions):
        """
        根据当前状态和动作计算下一个状态
        
        Args:
            state (Dict): 当前状态
            actions (List[int]): 玩家动作列表
            
        Returns:
            Dict: 下一个状态
        """
        # 这个方法在环境类中实现，这里只是提供接口
        # 实际的状态转换逻辑在环境类的step方法中
        return state
    
    def get_mdp_features(self, state):
        """
        从状态中提取MDP特征
        
        Args:
            state (Dict): 当前状态
            
        Returns:
            Dict: MDP特征
        """
        features = {
            # 玩家相关特征
            "player_positions": self.get_player_positions(state),
            "player_directions": self.get_player_directions(state),
            "player_hold_objects": self.get_player_hold_objects(state),
            
            # 桌子和物品相关特征
            "counter_objects": self.get_counter_objects_dict(state),
            
            # 地形相关特征
            "pot_locations": self.get_pot_locations(),
            "cutting_board_locations": self.get_cutting_table_locations(),
            "serving_locations": self.get_serving_locations(),
            "counter_locations": self.get_counter_locations(),
            "trash_locations": self.get_trash_locations(),
            
            # 供应处相关特征
            "dish_supply_locations": self.get_dish_dispenser_locations(),
            "lemon_supply_locations": self.get_lemon_dispenser_locations(),
            "fish_supply_locations": self.get_rawfish_dispenser_locations(),
            "tomato_supply_locations": self.get_tomato_dispenser_locations(),
            "beef_supply_locations": self.get_rawbeef_dispenser_locations(),
            "hamburger_supply_locations": self.get_hamburger_dispenser_locations(),
            
            # 任务相关特征
            "task_menu": self.TASK_MENU
        }
        
        return features
