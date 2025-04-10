import itertools, copy
import numpy as np
from functools import reduce
from collections import defaultdict
from utils import load_from_json
import os
import json
from overcook_gym_env import OvercookPygameEnv
from overcook_gym_class import Table, Direction, Action
from typing import List, Tuple, Optional, Dict, Union

# 获取当前文件的目录
current_dir = os.path.dirname(__file__)
maps_path = os.path.join(current_dir, 'maps.json')
with open(maps_path, 'r', encoding='utf-8') as file:
    maps = json.load(file)


class ComplexOvercookedGridworld(object):

    def __init__(self, env: OvercookPygameEnv):
        self.env = env 
        self.terrain = maps[env.map_name]['layout']
        self.terrain_mtx = self.convert_layout_to_2d_list()
        self.height = len(self.terrain_mtx)
        self.width = len(self.terrain_mtx[0])
        self.shape = (self.width, self.height)
        self.terrain_pos_dict = self._get_terrain_type_pos_dict()

        # self.start_player_positions = start_player_positions
        # self.num_players = len(start_player_positions)
        # self.start_order_list = start_order_list
        # self.soup_cooking_time = cook_time
        # self.num_items_for_soup = num_items_for_soup
        # self.delivery_reward = delivery_reward
        # self.reward_shaping_params = NO_REW_SHAPING_PARAMS if rew_shaping_params is None else rew_shaping_params
        # self.layout_name = layout_name
    # X: 桌子， F：生鱼供应处，B：生牛肉供应处，H：汉堡包供应处，M：番茄供应处，D：盘子供应处，L：柠檬供应处，T：垃圾桶，E：送菜口，C：锅，U：案板
    def convert_layout_to_2d_list(self) -> List[List[str]]:

        ignore_chars = ['_']
        terrain_mtx = [[char for char in row if char not in  ignore_chars] for row in self.terrain]
        terrain_mtx = [row for row in terrain_mtx if row]  # Remove empty rows
        for y, row in enumerate(terrain_mtx):
            for x, c in enumerate(row):
                if c in ['1', '2', '3', '4']:
                    terrain_mtx[y][x] = ' '

        return terrain_mtx
    
    def _get_terrain_type_pos_dict(self) -> Dict[str, List[Tuple[int, int]]]:
        pos_dict = defaultdict(list)
        for y, terrain_row in enumerate(self.terrain_mtx):
            for x, terrain_type in enumerate(terrain_row):
                pos_dict[terrain_type].append((x, y))
        return pos_dict
    
    def get_counter_objects_dict(self) -> Dict[str, List[Tuple[int, int]]]:
        """计算每个桌子X上物体的位置

        Returns:
            {item: [pos1, pos2]}
        """
        counter_objects = defaultdict(list)
        for table in self.env.game.tables:
            if isinstance(table, Table) and table.item:
                counter_objects[table.item].append((table.rect.x//80, table.rect.y//80-1))
    
        return counter_objects
    
    def get_interaction_pos_and_dire(self, goal_pos:List[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        获取与每个物体交互的位置和玩家面朝方向。

        :param terrain_posiobject_postions: list 物体位置
        :return: list，包含每个物体交互位置和面朝方向的元组 [(交互位置, 面朝方向)]。
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
        x, y = pos
        return self.terrain_mtx[y][x]

    def get_dish_dispenser_locations(self):
        return list(self.terrain_pos_dict['D'])

    def get_lemon_dispenser_locations(self):
        return list(self.terrain_pos_dict['L'])
    
    def get_tomato_dispenser_locations(self):
        return list(self.terrain_pos_dict['M'])
    
    def get_rawfish_dispenser_locations(self):
        return list(self.terrain_pos_dict['F'])
    
    def get_rawbeef_dispenser_locations(self):
        return list(self.terrain_pos_dict['B'])
    
    def get_hamburger_dispenser_locations(self):
        return list(self.terrain_pos_dict['H'])
    
    def get_cutting_table_locations(self):
        return list(self.terrain_pos_dict['U'])
    
    def get_serving_locations(self):
        return list(self.terrain_pos_dict['E'])

    def get_pot_locations(self):
        return list(self.terrain_pos_dict['C'])

    def get_counter_locations(self):
        return list(self.terrain_pos_dict['X'])
