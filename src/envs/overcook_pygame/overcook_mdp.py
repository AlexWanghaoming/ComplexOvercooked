import itertools, copy
import numpy as np
from functools import reduce
from collections import defaultdict
from utils import load_from_json
import os
import json
from overcook_gym_env import OvercookPygameEnv
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
    def convert_layout_to_2d_list(self):
        ignore_chars = ['_']
        
        terrain_mtx = [[char for char in row if char not in  ignore_chars] for row in self.terrain]
        terrain_mtx = [row for row in terrain_mtx if row]  # Remove empty rows
        for y, row in enumerate(terrain_mtx):
            for x, c in enumerate(row):
                if c in ['1', '2', '3', '4']:
                    terrain_mtx[y][x] = ' '

        return terrain_mtx
    
    def _get_terrain_type_pos_dict(self):
        pos_dict = defaultdict(list)
        for y, terrain_row in enumerate(self.terrain_mtx):
            for x, terrain_type in enumerate(terrain_row):
                pos_dict[terrain_type].append((x, y))
        return pos_dict
    
    def get_counter_objects_dict(self):
        counter_objects = defaultdict(list)

        # 遍历所有桌子的位置
        for pos in self.get_counter_locations():
            # 获取当前位置的物体类型
            terrain_type = self.get_terrain_type_at_pos(pos)
            if terrain_type != 'X': 
                continue
            # 检查桌子上是否有物体
            table_object = self.env.game.get_object_at(pos)
            if table_object:
                counter_objects[table_object].append(pos)

        return dict(counter_objects)
    
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
