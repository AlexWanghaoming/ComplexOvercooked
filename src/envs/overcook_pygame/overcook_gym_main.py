import pygame
import os, sys
from pygame.transform import scale as picscale
from pygame.image import load as picload
from overcook_gym_class import ONEBLOCK, ONEBLOCKSIZE, Player, TaskBoard, TimerTable, Wall, SupplyTable, Pot,\
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

    def __init__(self,
                  map_name, 
                  ifrender=False):
        
        LINES = maps[map_name]['layout']
        self.TASK_MENU = maps[map_name]['task']
        TASKNUM = maps[map_name]['tasknum']
        PLAYERNUM = maps[map_name]['players']
        # 定义主函数#替换成类方法，方便后续更新

        # 初始化pygame
        self.done = False
        self.NOWCOIN = 100

        pygame.init()
        # pygame.mixer.music.load('overcook_pygame/music/background.mp3')
        # pygame.mixer.music.set_volume(0.2)
        # pygame.mixer.music.play()

        """
        X: 桌子， F：生鱼供应处，B：生牛肉供应处，H：汉堡包供应处，M：番茄供应处，D：盘子供应处，L：柠檬供应处，
        T：垃圾桶，E：送菜口，C：锅，U：案板
        """

        lines = LINES
        # window_width = 80 * ONEBLOCK * (len(lines[0]) + 2)
        # window_height = 80 * ONEBLOCK * (len(lines) + 1)
        window_width = ONEBLOCK * (len(lines[0]) + 2)
        window_height = ONEBLOCK * (len(lines) + 1)
        if ifrender:
            self.window = pygame.display.set_mode((window_width, window_height))
            pygame.display.set_caption('Overcooked Game')
        else:
            self.window = pygame.display.set_mode((window_width, window_height), pygame.HIDDEN)

        # 设置窗口大小和标题
        picplayerlist = [{
            '[0, 1]_': picscale(picload(os.path.join(current_dir, f'assets/chef{i}/f1.png')).convert_alpha(),
                                ONEBLOCKSIZE),
            '[-1, 0]_': picscale(picload(os.path.join(current_dir, f'assets/chef{i}/l1.png')).convert_alpha(),
                                 ONEBLOCKSIZE),
            '[1, 0]_': picscale(picload(os.path.join(current_dir, f'assets/chef{i}/r1.png')).convert_alpha(),
                                ONEBLOCKSIZE),
            '[0, -1]_': picscale(picload(os.path.join(current_dir, f'assets/chef{i}/b1.png')).convert_alpha(),
                                 ONEBLOCKSIZE),
            '[0, 1]': picscale(picload(os.path.join(current_dir, f'assets/chef{i}/f0.png')).convert_alpha(),
                               ONEBLOCKSIZE),
            '[-1, 0]': picscale(picload(os.path.join(current_dir, f'assets/chef{i}/l0.png')).convert_alpha(),
                                ONEBLOCKSIZE),
            '[1, 0]': picscale(picload(os.path.join(current_dir, f'assets/chef{i}/r0.png')).convert_alpha(),
                               ONEBLOCKSIZE),
            '[0, -1]': picscale(picload(os.path.join(current_dir, f'assets/chef{i}/b0.png')).convert_alpha(),
                                ONEBLOCKSIZE),
        } for i in range(1, PLAYERNUM + 1)]

        itempicslist = ['dish', 'BClemon', 'AClemon', 'rawfish',
                        'AClemoncookedfish', 'cookedfish', 'pot',
                        'ACtomatocookedbeefhamburger', 'cookedbeefhamburger',
                        'hamburger', 'ACtomato', 'BCtomato', 'rawbeef', 'cookedbeef', 'ACtomatohamburger']
        itempics = {
            key: picscale(picload(os.path.join(current_dir, f'assets/items/{key}.png')).convert_alpha(), ONEBLOCKSIZE) for
            key in
            itempicslist}

        supplylist = ['dish', 'BClemon', 'BCtomato', 'rawfish', 'rawbeef', 'hamburger']
        
        supplyimg = {
            key: picscale(picload(os.path.join(current_dir, f'assets/table/{key}.png')).convert_alpha(), ONEBLOCKSIZE) for
            key in
            supplylist}

        wall_1 = Wall(-20, 0, 20, window_height)
        wall_2 = Wall(0, -20, window_width, 20)
        wall_3 = Wall(window_width, 0, 20, window_height)
        wall_4 = Wall(0, window_height, window_width, 20)
        walls = pygame.sprite.Group(wall_1, 
                                    wall_2, 
                                    wall_3, 
                                    wall_4)
        self.playergroup = []

        self.tables = pygame.sprite.Group()  # 需要交互的
        self.pots = pygame.sprite.Group() 
        self.cuttingtables = pygame.sprite.Group()  

        # 创建布局
        for i, line in enumerate(lines):
            for j, char in enumerate(line):
                x = j * ONEBLOCK
                y = (i + 1) * ONEBLOCK
                if char == 'X':
                    temp = Table(x, y, None, None, itempics)
                    self.tables.add(temp)
                elif char == 'F':
                    temp = SupplyTable(x, y, 'rawfish', supplyimg)
                    self.tables.add(temp)
                elif char == 'B':
                    temp = SupplyTable(x, y, 'rawbeef', supplyimg)
                    self.tables.add(temp)
                elif char == 'H':
                    temp = SupplyTable(x, y, 'hamburger', supplyimg)
                    self.tables.add(temp)
                elif char == 'M':
                    temp = SupplyTable(x, y, 'BCtomato', supplyimg)
                    self.tables.add(temp)
                elif char == 'D':
                    temp = SupplyTable(x, y, 'dish', supplyimg)
                    self.tables.add(temp)
                elif char == 'L':
                    temp = SupplyTable(x, y, 'BClemon', supplyimg)
                    self.tables.add(temp)
                elif char == 'T':
                    temp = TrashTable(x, y)
                    self.tables.add(temp)
                elif char == 'E':
                    Cointable = CoinTable(x, y)
                elif char == 'C':
                    temp = Pot(x, y, itempics)
                    self.pots.add(temp)
                    self.tables.add(temp)
                elif char == 'U':
                    temp = CuttingTable(x, y, itempics)
                    self.cuttingtables.add(temp)
                    self.tables.add(temp)

                # initilize players at the start point (x,y)
                elif char == '1':
                    self.playergroup.append(Player('a', x, y, picplayerlist[0], itempics))
                elif char == '2':
                    self.playergroup.append(Player('b', x, y, picplayerlist[1], itempics))
                elif char == '3':
                    self.playergroup.append(Player('c', x, y, picplayerlist[2], itempics))
                elif char == '4':
                    self.playergroup.append(Player('d', x, y, picplayerlist[3], itempics))

        # 创建墙体
        font = pygame.font.SysFont('arial', ONEBLOCK)

        self.font = pygame.font.SysFont('arial', 24)

        self.timercount = TimerTable(window_width - ONEBLOCK, 
                                     0, 
                                     font, 
                                     600)
        walls.add(self.tables)
        walls.add(self.tables, Cointable)
        self.walls = walls
        self.num1 = DigitDisplay(ONEBLOCK / 2, ONEBLOCK / 10)
        self.num2 = DigitDisplay(ONEBLOCK / 2 + 5 * 5, ONEBLOCK / 10)
        self.num3 = DigitDisplay(ONEBLOCK / 2 + 5 * 5 * 2, ONEBLOCK / 10)
        self.num4 = DigitDisplay(ONEBLOCK / 2 + 5 * 5 * 3, ONEBLOCK / 10)
        
        Coin = Picshow(0, 0, os.path.join(current_dir, f'assets/font/coin.png'))
        self.task_sprites = pygame.sprite.Group()
        self.task_sprites.add(TaskBoard((1.75) * ONEBLOCK, 0, self.TASK_MENU))
        
        # randome choice TASKNUM task from task menu
        if TASKNUM>=2:
            for i in range(1, TASKNUM):
                self.task_sprites.add(TaskBoard((3*(i-1)) * ONEBLOCK, len(lines)*ONEBLOCK, self.TASK_MENU))
        self.task_dict = {}
        
        for i,task in enumerate(self.task_sprites):
            self.task_dict[task] = i
        # print(self.task_dict)

        # task1 = taskBoard(1.75 * ONEBLOCK, 0,TASK_MENU)
        # task2 = taskBoard((1.75 + 3) * ONEBLOCK, 0)
        # self.task_sprites = pygame.sprite.Group(task1)
        # 创建精灵组
        self.all_sprites = pygame.sprite.Group(walls, 
                                               self.num1, 
                                               self.num2, 
                                               self.num3,
                                               self.num4, 
                                               Coin, 
                                               self.task_sprites,
                                               Cointable, 
                                               self.timercount)
        for i in range(PLAYERNUM):
            self.all_sprites.add(self.playergroup[i])
        self.Cointable = Cointable

        self.taskmenu = []
        for task in self.task_sprites:
            self.taskmenu.append(task.task)

        # 在创建其他显示组件后添加rewards显示
        self.rewards_text = pygame.font.SysFont('arial', 24)
        self.current_reward = 0.0

    def update_reward(self, reward):
        """更新当前reward值"""
        self.current_reward = reward
        
    def draw_reward(self):
        """在界面上绘制reward"""
        if hasattr(self, 'window'):
            reward_surface = self.rewards_text.render(f'Reward: {self.current_reward:.2f}', True, (0, 0, 0))
            self.window.blit(reward_surface, (ONEBLOCK * 5, ONEBLOCK / 10))

    def mainloop(self):
        pass


if __name__ == '__main__':
    mainwindows = MainGame(map_name="supereasy", 
                           ifrender=True)
    # mainwindows.mainloop()