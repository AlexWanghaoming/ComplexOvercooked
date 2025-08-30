import pygame
import os, sys
from pygame.transform import scale as picscale
from pygame.image import load as picload
from pygame.event import Event as Event
from pygame.sprite import Group as Group
from envs.overcook_class import ONEBLOCK, ONEBLOCKSIZE, Player, TaskBoard, TimerTable, Wall, SupplyTable, Pot,\
    Table, CoinTable, TrashTable, DigitDisplay, Picshow, CuttingTable, TASK_FINISH_EVENT
import json

# 获取当前文件的目录
current_dir = os.path.dirname(__file__)
# 构建maps.json的路径
maps_path = os.path.join(current_dir, 'maps.json')

with open(maps_path, 'r', encoding='utf-8') as file:
    maps = json.load(file)

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

class MainGame(object):

    def __init__(self, map_name, ifrender=False):
        self.LINES = maps[map_name]['layout']
        self.TASK_MENU = maps[map_name]['task']
        self.TASKNUM = maps[map_name]['tasknum']
        self.PLAYERNUM = maps[map_name]['players']
        self.NOWCOIN = 0
        window_width = ONEBLOCK * (len(self.LINES[0]) + 2)
        window_height = ONEBLOCK * (len(self.LINES) + 1)
        
        pygame.init()

        self.ifrender = ifrender
        # 只在需要渲染时初始化pygame显示
        if ifrender:
            self.window = pygame.display.set_mode((window_width, window_height)) # 加载窗口
            pygame.display.set_caption('ComplexOvercooked Game')
        else:
            self.window = pygame.display.set_mode((window_width, window_height), pygame.HIDDEN)

        self.load_pics()
        
        self.init_maps()
        
        self.init_tasks()

        # 初始化墙壁
        wall_1 = Wall(-20, 0, 20, window_height)
        wall_2 = Wall(0, -20, window_width, 20)
        wall_3 = Wall(window_width, 0, 20, window_height)
        wall_4 = Wall(0, window_height, window_width, 20)
        walls = Group(wall_1, 
                                    wall_2, 
                                    wall_3, 
                                    wall_4)
        walls.add(self.tables)
        walls.add(self.tables, self.Cointable)
        self.walls = walls
        self.num1 = DigitDisplay(ONEBLOCK / 2, ONEBLOCK / 10)
        self.num2 = DigitDisplay(ONEBLOCK / 2 + 5 * 5, ONEBLOCK / 10)
        self.num3 = DigitDisplay(ONEBLOCK / 2 + 5 * 5 * 2, ONEBLOCK / 10)
        self.num4 = DigitDisplay(ONEBLOCK / 2 + 5 * 5 * 3, ONEBLOCK / 10)
                    
        font = pygame.font.SysFont('arial', ONEBLOCK)
        self.timercount = TimerTable(window_width - ONEBLOCK, 0, font, 600, self.ifrender)
        self.Coin = Picshow(0, 0, os.path.join(current_dir, f'assets/font/coin.png'))
        
        self.init_all_sprites()

        # 在创建其他显示组件后添加rewards显示
        self.rewards_text = pygame.font.SysFont('arial', 24)
        self.current_reward = 0.0

    def load_pics(self):
        # 加载player面朝各个方向的图片
        self.picplayerlist = [{
            '(0, 1)_': picscale(picload(os.path.join(current_dir, f'assets/chef{i}/f1.png')).convert_alpha(),
                                ONEBLOCKSIZE),
            '(-1, 0)_': picscale(picload(os.path.join(current_dir, f'assets/chef{i}/l1.png')).convert_alpha(),
                                 ONEBLOCKSIZE),
            '(1, 0)_': picscale(picload(os.path.join(current_dir, f'assets/chef{i}/r1.png')).convert_alpha(),
                                ONEBLOCKSIZE),
            '(0, -1)_': picscale(picload(os.path.join(current_dir, f'assets/chef{i}/b1.png')).convert_alpha(),
                                 ONEBLOCKSIZE),
            '(0, 1)': picscale(picload(os.path.join(current_dir, f'assets/chef{i}/f0.png')).convert_alpha(),
                               ONEBLOCKSIZE),
            '(-1, 0)': picscale(picload(os.path.join(current_dir, f'assets/chef{i}/l0.png')).convert_alpha(),
                                ONEBLOCKSIZE),
            '(1, 0)': picscale(picload(os.path.join(current_dir, f'assets/chef{i}/r0.png')).convert_alpha(),
                               ONEBLOCKSIZE),
            '(0, -1)': picscale(picload(os.path.join(current_dir, f'assets/chef{i}/b0.png')).convert_alpha(),
                                ONEBLOCKSIZE),
        } for i in range(1, self.PLAYERNUM + 1)]

        itempicslist = ['dish', 'BClemon', 'AClemon', 'rawfish',
                        'AClemoncookedfish', 'cookedfish', 'pot',
                        'ACtomatocookedbeefhamburger', 'cookedbeefhamburger','hamburger', 'ACtomato', 'BCtomato', 'rawbeef', 'cookedbeef', 'ACtomatohamburger']
        # 加载item的图片
        self.itempics = {
            key: picscale(picload(os.path.join(current_dir, f'assets/items/{key}.png')).convert_alpha(), ONEBLOCKSIZE) for
            key in
            itempicslist}

        supplylist = ['dish', 'BClemon', 'BCtomato', 'rawfish', 'rawbeef', 'hamburger']
        # 加载supply的图片
        self.supplyimg = {
            key: picscale(picload(os.path.join(current_dir, f'assets/table/{key}.png')).convert_alpha(), ONEBLOCKSIZE) for
            key in
            supplylist}
    
    def init_maps(self):
        self.playergroup = []
        self.tables = Group()  # 需要交互的
        self.pots = Group() 
        self.cuttingtables = Group()  
        # 创建地图
        for i, line in enumerate(self.LINES):
            for j, char in enumerate(line):
                x = j * ONEBLOCK
                y = (i + 1) * ONEBLOCK
                if char == 'X':
                    temp = Table(x, y, None, None, self.itempics, self.ifrender)
                    self.tables.add(temp)
                elif char == 'F':
                    temp = SupplyTable(x, y, 'rawfish', self.supplyimg, self.ifrender)
                    self.tables.add(temp)
                elif char == 'B':
                    temp = SupplyTable(x, y, 'rawbeef', self.supplyimg, self.ifrender)
                    self.tables.add(temp)
                elif char == 'H':
                    temp = SupplyTable(x, y, 'hamburger', self.supplyimg, self.ifrender)
                    self.tables.add(temp)
                elif char == 'M':
                    temp = SupplyTable(x, y, 'BCtomato', self.supplyimg, self.ifrender)
                    self.tables.add(temp)
                elif char == 'D':
                    temp = SupplyTable(x, y, 'dish', self.supplyimg, self.ifrender)
                    self.tables.add(temp)
                elif char == 'L':
                    temp = SupplyTable(x, y, 'BClemon', self.supplyimg, self.ifrender)
                    self.tables.add(temp)
                elif char == 'T':
                    temp = TrashTable(x, y, self.ifrender)
                    self.tables.add(temp)
                elif char == 'E':
                    self.Cointable = CoinTable(x, y, self.ifrender)
                elif char == 'C':
                    temp = Pot(x, y, self.itempics, self.ifrender)
                    self.pots.add(temp)
                    self.tables.add(temp)
                elif char == 'U':
                    temp = CuttingTable(x, y, self.itempics, self.ifrender)
                    self.cuttingtables.add(temp)
                    self.tables.add(temp)
                    
                # 初始化player的初始位置 (x,y)
                elif char == '1':
                    self.playergroup.append(Player('a', x, y, self.picplayerlist[0], self.itempics, self.ifrender))
                elif char == '2':
                    self.playergroup.append(Player('b', x, y, self.picplayerlist[1], self.itempics, self.ifrender))
                elif char == '3':
                    self.playergroup.append(Player('c', x, y, self.picplayerlist[2], self.itempics, self.ifrender))
                elif char == '4':
                    self.playergroup.append(Player('d', x, y, self.picplayerlist[3], self.itempics, self.ifrender))
    
    def init_tasks(self):
        # 创建任务精灵组
        self.task_sprites = Group()
        self.task_sprites.add(TaskBoard((1.75) * ONEBLOCK, 0, self.TASK_MENU, self.ifrender))
        # randome choice TASKNUM task from task menu
        if self.TASKNUM>=2:
            for i in range(1, self.TASKNUM):
                self.task_sprites.add(TaskBoard((3*(i-1)) * ONEBLOCK, len(self.LINES)*ONEBLOCK, self.TASK_MENU, self.ifrender))
                
        self.task_dict = {}
        self.taskmenu = []
        for i,task in enumerate(self.task_sprites):
            self.task_dict[task] = i
            self.taskmenu.append(task.task)
        # print(self.task_dict)
    
    def init_all_sprites(self):
        self.all_sprites = Group(self.walls, 
                                self.num1, 
                                self.num2, 
                                self.num3,
                                self.num4, 
                                self.Coin, 
                                self.task_sprites,
                                self.Cointable, 
                                self.timercount)
        for i in range(self.PLAYERNUM):
            self.all_sprites.add(self.playergroup[i])
            
    def update_reward(self, reward):
        """更新当前reward值"""
        self.current_reward = reward
        
    def draw_reward(self):
        """在界面上绘制reward"""
        if hasattr(self, 'window'):
            reward_surface = self.rewards_text.render(f'Reward: {self.current_reward:.2f}', True, (0, 0, 0))
            self.window.blit(reward_surface, (ONEBLOCK * 5, ONEBLOCK / 10))



if __name__ == '__main__':
    mainwindows = MainGame(map_name="supereasy", 
                           ifrender=True)
