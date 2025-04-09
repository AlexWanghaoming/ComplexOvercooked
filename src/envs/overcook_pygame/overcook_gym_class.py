import pygame
from pygame.transform import scale as picscale
from pygame.image import load as picload
import random
import os
from typing import List, Dict, Tuple, Any, Union
import ipdb

current_dir = os.path.dirname(__file__)

globalscale = 5
ONEBLOCK = 16 * globalscale
# ONEBLOCK = 1
ONEBLOCKSIZE = (ONEBLOCK, ONEBLOCK)
DISHSIZE = (0.75*ONEBLOCK, 
            0.75*ONEBLOCK)
ITEMSIZE =  (ONEBLOCK/2, ONEBLOCK/2)

TASK_FINISH_EVENT = pygame.USEREVENT + 1
OUT_SUPPLY_EVENT = pygame.USEREVENT + 2
OUT_DISH_EVENT = pygame.USEREVENT + 3
GET_MATERIAL_EVENT = pygame.USEREVENT + 4
GET_DISH_EVENT = pygame.USEREVENT + 5
MADE_NEWTHING_EVENT = pygame.USEREVENT+6
BEGINCUTTING_EVENT = pygame.USEREVENT+7
CUTTINGDOWN_EVENT = pygame.USEREVENT+8
BEGINCOOKING_EVENT = pygame.USEREVENT+9
COOKINGDOWN_EVENT = pygame.USEREVENT+10
COOKINGOUT_EVENT = pygame.USEREVENT +11
TRY_NEWTHING_EVENT = pygame.USEREVENT+12
PUTTHING_DISH_EVENT = pygame.USEREVENT+13
TRASH_EVENT = pygame.USEREVENT+14


RECIPE = {
    frozenset(['AClemon', 'cookedfish']): 'AClemoncookedfish',
    frozenset(['ACtomatohamburger', 'cookedbeef']): 'ACtomatocookedbeefhamburger',
    frozenset(['ACtomato', 'cookedbeefhamburger']): 'ACtomatocookedbeefhamburger',
    frozenset(['cookedbeef', 'hamburger']): 'cookedbeefhamburger',
    frozenset(['ACtomato', 'hamburger']): 'ACtomatohamburger',
}


class Action():
    STAY = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    LEFT = 4
    INTERACT = 5
    # to do 可以以table类为基础做基类

        # self.control = {0:[0,0],
        #                 1:[0,1],
        #                 2:[1,0],
        #                 3:[0,-1],
        #                 4:[-1,0]}
                
class Direction():

    directions = {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0)
        }
    
    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST = (1, 0)
    WEST = (-1, 0)
    
    DIRECTION2INDEX = {
                        NORTH: 0, 
                        EAST: 1, 
                        WEST: 2, 
                        SOUTH: 3
                        }
        
    DIRECTION2CAR = {
                    NORTH: '↑', 
                    EAST: '→', 
                    WEST: '←', 
                    SOUTH: '↓'
                    }

class TaskBoard(pygame.sprite.Sprite):
    def __init__(self, x, y, taskmenu):
        super().__init__()
        self.menu = taskmenu
        # 根据概率选择任务
        task_items = list(self.menu.keys())
        if isinstance(self.menu[task_items[0]], list):  # 检查是否是新格式 [分数, 概率]
            probs = [self.menu[item][1] for item in task_items]
            self.task = random.choices(task_items, weights=probs, k=1)[0]
        else:  # 兼容旧格式
            self.task = random.choice(task_items)
        
        self.image = picscale(picload(os.path.join(current_dir, f'assets/TASK/{self.task}.png')).convert_alpha(), (3 * ONEBLOCK, ONEBLOCK))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.timer = 200
        self.remaining_time = self.timer
        self.start_time = 0

    def update(self,nowtime) -> None:
        if self.timer > 0:
            elapsed_time = nowtime - self.start_time
            self.remaining_time = self.timer - elapsed_time
            if self.remaining_time <= 0:
                oldtask = self.task
                self.newtask(nowtime)
                pygame.event.post(pygame.event.Event(pygame.USEREVENT, {'action': 'notfinished',
                                                                        'newtask':self.task,
                                                                        'oldtask':oldtask,
                                                                        'taskclass':self}))

            else:
                progress_bar_rect = pygame.Rect(0, ONEBLOCK*7/8-3, 3*ONEBLOCK * (1-self.remaining_time / self.timer), ONEBLOCK / 8)
                progress_bar_surface = pygame.Surface(progress_bar_rect.size)
                progress_bar_surface.fill((128, 200, 0))
                self.image.blit(progress_bar_surface, progress_bar_rect)

    def newtask(self, nowtime):
        # 根据概率选择新任务
        task_items = list(self.menu.keys())
        if isinstance(self.menu[task_items[0]], list):  # 检查是否是新格式
            probs = [self.menu[item][1] for item in task_items]
            self.task = random.choices(task_items, weights=probs, k=1)[0]
        else:  # 兼容旧格式
            self.task = random.choice(task_items)
            
        self.start_time = nowtime
        self.remaining_time = self.timer
        self.image = picscale(picload(os.path.join(current_dir, f'assets/TASK/{self.task}.png')).convert_alpha(), (3 * ONEBLOCK, ONEBLOCK))
        return self.task


class DigitDisplay(pygame.sprite.Sprite):
    def __init__(self,x,y):
        super().__init__()
        # 加载数字精灵表
        self.digit_sprite_sheet = picload(os.path.join(current_dir, f'assets/font/number.png')).convert_alpha()
        # 确定每个数字的位置和大小
        self.digit_rects = [
            pygame.Rect(i * 5, 0, 4, 5) for i in range(10)
        ]
        # 初始化显示数字为0
        self.num = 0
        self.image = picscale(self.digit_sprite_sheet.subsurface(self.digit_rects[self.num]), (4*5,5*5))
        self.rect = self.image.get_rect()  # 获取图片的矩形区域
        self.rect.x = x  # 设置矩形区域的位置
        self.rect.y = y

    def set_num(self, num):
        # 设置显示的数字，并更新图像
        self.num = num
        self.image = picscale(self.digit_sprite_sheet.subsurface(self.digit_rects[self.num]), (4*5,5*5))


class Picshow(pygame.sprite.Sprite):
    def __init__(self, x, y, picdir):
        super().__init__()
        self.image = picscale(picload(picdir).convert_alpha(),(ONEBLOCK/2,ONEBLOCK/2))
        self.rect = self.image.get_rect()  # 获取图片的矩形区域
        self.rect.x = x  # 设置矩形区域的位置
        self.rect.y = y


class SupplyTable(pygame.sprite.Sprite):
    def __init__(self, x, y, item, itempics):
        super().__init__()
        self.item = item  # 设置物品属性，默认为 None
        self.pics = itempics
        self.updateimg()
        self.rect = self.image.get_rect()  # 获取图片的矩形区域
        self.rect.x = x  # 设置矩形区域的位置
        self.rect.y = y

    def isnewthing(self,item):
        pygame.event.post(pygame.event.Event(TRY_NEWTHING_EVENT))
        if frozenset([self.item,item]) in RECIPE:
            pygame.event.post(pygame.event.Event(MADE_NEWTHING_EVENT,{'newitem':RECIPE[frozenset([self.item,item])]}))
            return RECIPE[frozenset([self.item,item])]
        
    def updateimg(self, ):
        self.image = picscale(picload(os.path.join(current_dir, f'assets/table/table.png')).convert_alpha(), ONEBLOCKSIZE)  #
        item_rect = pygame.Rect(0,0, ONEBLOCK, ONEBLOCK)
        item_surface = picscale(self.pics[self.item], (ONEBLOCK, ONEBLOCK))
        self.image.blit(item_surface, item_rect)   

    def update(self, player:pygame.sprite.Sprite, keys:bool, nowtime) -> None:
        if keys:
            if player.rect.move(player.direction[0] * ONEBLOCK / 2,
                                player.direction[1] * ONEBLOCK / 2).colliderect(self.rect):
                #
                if self.item=='dish':
                    if not player.dish:
                        player.dish = 'dish'
                        pygame.event.post(pygame.event.Event(OUT_DISH_EVENT))
                        if player.item:
                            pygame.event.post(pygame.event.Event(PUTTHING_DISH_EVENT, {'item': player.item}))
                else:
                    if not player.item:


                        player.item = self.item
                        pygame.event.post(pygame.event.Event(OUT_SUPPLY_EVENT, {'item': self.item}))
                    elif player.dish and self.item:
                        tmp = self.isnewthing(player.item)
                        if tmp:
                            pygame.event.post(pygame.event.Event(PUTTHING_DISH_EVENT, {'item': tmp}))
                            player.item = tmp
                            player.updateimg()
                player.updateimg()

    def availbeinter(self,player):
        if player.rect.move(player.direction[0] * ONEBLOCK / 2,
                            player.direction[1] * ONEBLOCK / 2).colliderect(self.rect):
            if self.item == 'dish':
                if not player.dish:
                    return True
            else:
                if not player.item:
                    return True
                elif player.dish and self.item:
                    if frozenset([self.item,player.item]) in RECIPE:
                        return True
        return False


class TimerTable(pygame.sprite.Sprite):
    def __init__(self, x, y, font, time):
        super().__init__()
        self.timer = time
        self.image = font.render(str(self.timer),True,(0,0,0))
        self.font = font
        self.rect = self.image.get_rect()  # 获取图片的矩形区域
        self.rect.x = x  # 设置矩形区域的位置
        self.rect.y = y

    def update(self,nowtime) -> None:
        if self.timer - nowtime<=0:
            pygame.event.post(pygame.event.Event(pygame.USEREVENT, {'action': 'countdown_finished'}))
        self.image = self.font.render(str((self.timer - nowtime)//10),True,(0,0,0))


class CoinTable(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.updateimg()
        self.rect = self.image.get_rect()  # 获取图片的矩形区域
        self.rect.x = x  # 设置矩形区域的位置
        self.rect.y = y

    def updateimg(self, ):
        self.image = picscale(picload(os.path.join(current_dir, f'assets/table/table.png')).convert_alpha(), ONEBLOCKSIZE)  #
        item_rect = pygame.Rect(0,0, ONEBLOCK, ONEBLOCK)
        item_surface = picscale(picload(os.path.join(current_dir, f'assets/table/cointable.png')), (ONEBLOCK, ONEBLOCK))
        self.image.blit(item_surface, item_rect)    

    def update(self, player:pygame.sprite.Sprite, keys:bool, taskmenu) -> None:
        if keys:
            if player.rect.move(player.direction[0] * ONEBLOCK / 2,
                                player.direction[1] * ONEBLOCK / 2).colliderect(self.rect):
                if player.dish and player.item in taskmenu:
                    # score = taskmenu[player.item][0] if isinstance(taskmenu[player.item], list) else taskmenu[player.item]
                    # pygame.event.post(pygame.event.Event(TASK_FINISH_EVENT, {'action': player.item, 'player': player.name, 'score': score}))
                    pygame.event.post(pygame.event.Event(TASK_FINISH_EVENT, {'action': player.item, 'player': player.name}))
                    player.dish = None
                    player.item = None
                    player.updateimg()

    def availbeinter(self,player):
        if player.rect.move(player.direction[0] * ONEBLOCK / 2,
                            player.direction[1] * ONEBLOCK / 2).colliderect(self.rect):
            if player.dish and player.item:
                return True
        return False


class TrashTable(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.updateimg()
        self.rect = self.image.get_rect()  # 获取图片的矩形区域
        self.rect.x = x  # 设置矩形区域的位置
        self.rect.y = y
        self.item = None

    def updateimg(self, ):
        self.image = picscale(picload(os.path.join(current_dir, f'assets/table/trashbin.png')).convert_alpha(), ONEBLOCKSIZE)  #

    def update(self, player:pygame.sprite.Sprite, keys:bool, nowtime) -> None:
        if keys:
            if player.rect.move(player.direction[0] * ONEBLOCK / 2,
                                player.direction[1] * ONEBLOCK / 2).colliderect(self.rect):
                #
                pygame.event.post(pygame.event.Event(TRASH_EVENT, {'item': player.item}))
                player.dish = None
                player.item = None
                player.updateimg()
    def availbeinter(self,player):
        if player.rect.move(player.direction[0] * ONEBLOCK / 2,
                            player.direction[1] * ONEBLOCK / 2).colliderect(self.rect):
            if player.dish or player.item:
                return True
        return False


class Table(pygame.sprite.Sprite):
    def __init__(self, x, y, item, dish, tablepics):
        super().__init__()
        self.item = item  # 设置物品属性，默认为 None
        self.dish = dish
        self.pics = tablepics
        self.updateimg()
        self.rect = self.image.get_rect()  # 获取图片的矩形区域
        self.rect.x = x  # 设置矩形区域的位置
        self.rect.y = y

    def isnewthing(self,item):
        pygame.event.post(pygame.event.Event(TRY_NEWTHING_EVENT))
        if frozenset([self.item,item]) in RECIPE:
            pygame.event.post(pygame.event.Event(MADE_NEWTHING_EVENT,{'newitem':RECIPE[frozenset([self.item,item])]}))
            return RECIPE[frozenset([self.item,item])]
        
    def updateimg(self, ):
        self.image = picscale(picload(os.path.join(current_dir, f'assets/table/table.png')).convert_alpha(), ONEBLOCKSIZE)  #
        if self.dish:
            item_rect = pygame.Rect(ONEBLOCK / 8, ONEBLOCK / 8, 0.75 * ONEBLOCK, 0.75 * ONEBLOCK)
            item_surface = picscale(self.pics[self.dish], (0.75 * ONEBLOCK, 0.75 * ONEBLOCK))
            self.image.blit(item_surface, item_rect)
        if self.item:
            item_rect = pygame.Rect(ONEBLOCK / 4, ONEBLOCK / 4, ONEBLOCK / 2, ONEBLOCK / 2)
            item_surface = picscale(self.pics[self.item], (ONEBLOCK / 2, ONEBLOCK / 2))
            self.image.blit(item_surface, item_rect)

    def update(self, player:pygame.sprite.Sprite, keys:bool, nowtime) -> None:
        if keys:
            if player.rect.move(player.direction[0] * ONEBLOCK / 2,
                                player.direction[1] * ONEBLOCK / 2).colliderect(self.rect):
                #
                if self.dish:  # 桌上有盘子
                    if self.item:  # 盘子里有东西
                        if not player.item and not player.dish:
                            player.exchangedish(self)
                            player.exchangeitem(self)

                        elif not player.item and player.dish:  # 手上有盘子

                            pygame.event.post(pygame.event.Event(PUTTHING_DISH_EVENT, {'item': self.item}))
                            player.exchangeitem(self)
                            #pygame.event.post(pygame.event.Event(PUTTHING_DISH_EVENT, {'item': self.item}))交换，因此不变情况

                        else:#手上有东西
                            tmp = self.isnewthing(player.item)
                            if tmp:
                                pygame.event.post(pygame.event.Event(PUTTHING_DISH_EVENT, {'item': tmp}))
                                self.item = tmp
                                player.item = None
                        # 手上有盘有东西或手上无盘有东西没法拿了
                    else:#盘子里没东西
                        if player.item and player.dish:#把手上的盘子里的东西放桌上的盘子里
                            
                            self.item, player.item =player.item,self.item
                            #pygame.event.post(pygame.event.Event(PUTTHING_DISH_EVENT, {'item': self.item}))放置，场上情况也不变
                        elif player.item and not player.dish:#把东西放盘子里
                            self.item, player.item =player.item,self.item
                            pygame.event.post(pygame.event.Event(PUTTHING_DISH_EVENT, {'item': self.item}))


                        else:
                            player.dish, self.dish =self.dish,player.dish#拿起盘子或交换盘子（无事发生）

                else:#桌上没盘子
                    if self.item:  # 桌上有东西
                        if not player.item:#玩家空手,东西# 玩家手上有盘子，东西装盘子里
                            self.item, player.item =player.item,self.item
                            if player.dish:
                                pygame.event.post(pygame.event.Event(PUTTHING_DISH_EVENT, {'item': player.item}))

                        elif player.dish:
                            tmp = self.isnewthing(player.item)
                            if tmp:
                                pygame.event.post(pygame.event.Event(PUTTHING_DISH_EVENT, {'item': tmp}))
                                self.item = tmp
                                self.dish,player.dish = player.dish,self.dish
                                player.item = None


                        # 手上有盘有东西或手上无盘有东西没法拿了
                    else:  # 桌上没东西
                        if player.item:  # 把手上的盘子里的东西放桌上的盘子里
                            self.item, player.item =player.item,self.item


                        if player.dish:
                            player.dish, self.dish =self.dish,player.dish
                self.updateimg()
                player.updateimg()

    def availbeinter(self,player):
        if player.rect.move(player.direction[0] * ONEBLOCK / 2,
                            player.direction[1] * ONEBLOCK / 2).colliderect(self.rect):
            if self.dish:  # 桌上有盘子
                if self.item:  # 盘子里有东西
                    if not player.item and not player.dish:
                        return True

                    elif not player.item and player.dish:  # 手上有盘子
                        return
                    else:  # 手上有东西
                        if frozenset([self.item,player.item]) in RECIPE:
                            return True
                        else:
                            return False
                    # 手上有盘有东西或手上无盘有东西没法拿了
                else:  # 盘子里没东西
                    if player.item and player.dish:  # 把手上的盘子里的东西放桌上的盘子里

                        return True
                    elif player.item and not player.dish:  # 把东西放盘子里
                        return True

                    else:
                        return False  # 拿起盘子或交换盘子（无事发生）

            else:  # 桌上没盘子
                if self.item:  # 桌上有东西
                    if not player.item:  # 玩家空手,东西# 玩家手上有盘子，东西装盘子里
                        return True
                    elif player.dish:
                        if frozenset([self.item,player.item]) in RECIPE:
                            return True

                    # 手上有盘有东西或手上无盘有东西没法拿了
                else:  # 桌上没东西
                    if not player.item and not player.dish:  # 把手上的盘子里的东西放桌上的盘子里
                        return False
                    return True
        return False


class Player(pygame.sprite.Sprite):
    def __init__(self,name, x, y, playerpic, itempic):

        super().__init__()
        self.name = name
        self.playerpic = playerpic
        self.itempic = itempic
        self.direction = (0, 1)
        self.item = None
        self.dish = None
        self.image = playerpic[str(self.direction)]
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        # TODO: 目前layout矩阵中x轴是从0开始，y轴从1开始
        self.control = {Action.STAY: (0,0),
                        Action.DOWN: (0,1),
                        Action.RIGHT: (1,0),
                        Action.UP: (0,-1),
                        Action.LEFT: (-1,0)}
                
        self.speed = ONEBLOCK
        self.cutting=False

    def exchangedish(self,table):
        self.dish,table.dish = table.dish,self.dish
        pygame.event.post(pygame.event.Event(GET_DISH_EVENT, 
                                             {'player': self.name,'olditem':self.dish,
                                              'newitem':table.dish}))

    def exchangeitem(self,table):
        self.item,table.item = table.item,self.item
        pygame.event.post(pygame.event.Event(GET_MATERIAL_EVENT, 
                                             {'player': self.name,'olditem':self.item,
                                              'newitem':table.item}))        
    
    def updateimg(self):
        if self.dish or self.item:
            self.image = self.playerpic[str(self.direction) + '_'].copy()
            if self.direction != (0, -1):
                if self.dish:
                        item_rect = pygame.Rect((1 + self.direction[0]) * ONEBLOCK / 8, ONEBLOCK / 4, DISHSIZE[0],DISHSIZE[0])
                        item_surface = picscale(self.itempic[self.dish], DISHSIZE)
                        self.image.blit(item_surface, item_rect)
                if self.item:
                        item_rect = pygame.Rect((1 + self.direction[0]) * ONEBLOCK / 4, ONEBLOCK / 2, ONEBLOCK / 2,
                                                ONEBLOCK / 2)
                        item_surface = picscale(self.itempic[self.item], (ONEBLOCK / 2, ONEBLOCK / 2))
                        self.image.blit(item_surface, item_rect)
        else:
            self.image = self.playerpic[str(self.direction)]

    def availaction(self, 
                    action:int,
                    pengzhuang:pygame.sprite.Group, 
                    player:List) -> bool:
        
        move_vector = self.control[action]
        if move_vector == self.direction: # 如果移动的方向和面朝的方向一致
            pengzhuangflag = True
            for wall in pengzhuang:
                if self.rect.move(move_vector[0] * self.speed,
                             move_vector[1] * self.speed).colliderect(wall.rect):
                    pengzhuangflag = False
                    break
            if pengzhuangflag:
                if isinstance(player, list):
                    for play in player:
                        if self.rect.move(move_vector[0] * self.speed,
                            move_vector[1] * self.speed).colliderect(play.rect):
                            pengzhuangflag = False
                            break
                else:
                    if self.rect.move(move_vector[0] * self.speed,
                            move_vector[1] * self.speed).colliderect(player.rect):
                        pengzhuangflag = False
            return pengzhuangflag
        else:
            # # 如果移动的方向和面朝的方向不一致，则改变方向，action一定可行
            return True

    def update(self, action, pengzhuang, player):
        if self.cutting:
            return
        tempx = self.rect.x
        tempy = self.rect.y
        move_vector = self.control[action]
        if move_vector == self.direction:
            self.rect.x += move_vector[0] * self.speed
            self.rect.y += move_vector[1] * self.speed
        elif sum(move_vector) == 0:
            return
        else:
            self.direction = move_vector
            self.updateimg()
            return True#改变方向是available的
        pengzhuangflag = True

        # 碰撞检测
        for wall in pengzhuang:
            if self.rect.colliderect(wall.rect):
                self.rect.x = tempx
                self.rect.y = tempy
                pengzhuangflag = False
                break

        if pengzhuangflag:
            if isinstance(player,list):
                for play in player:
                    if self.rect.colliderect(play.rect):
                        self.rect.x = tempx
                        self.rect.y = tempy
                        pengzhuangflag = False
                        break
            else:
                if self.rect.colliderect(player.rect):
                    self.rect.x = tempx
                    self.rect.y = tempy
                    pengzhuangflag = False

        return pengzhuangflag
    

class CuttingTable(pygame.sprite.Sprite):

    def __init__(self, x, y,  itempics):
        super().__init__()
        self.item = None  # 设置物品属性，默认为 None
        self.pics = itempics
        self.cuttingtime = 6
        self.timer = 0
        self.start_time = 0
        self.updateimg()
        self.rect = self.image.get_rect()  # 获取图片的矩形区域
        self.rect.x = x  # 设置矩形区域的位置
        self.rect.y = y
        gif = picload(os.path.join(current_dir, f'assets/table/cutting.png')).convert_alpha()
        self.frames = []
        self.e_time=0
        self.cuttingplayer = None
        for i in range(3):
            self.frames.append(gif.subsurface(pygame.Rect(i * 32, 0, 32, 32)))
    
    @property
    def is_cutting(self):
        return self.item is not None and self.timer > 0
    
    @property
    def is_ready(self):
        return self.item is not None and 'raw' not in self.item
    
    @property
    def is_empty(self):
        return self.item is None
    
    def isnewthing(self,item):
        pygame.event.post(pygame.event.Event(TRY_NEWTHING_EVENT))
        if frozenset([self.item,item]) in RECIPE:
            pygame.event.post(pygame.event.Event(MADE_NEWTHING_EVENT,{'newitem':RECIPE[frozenset([self.item,item])]}))
            return RECIPE[frozenset([self.item,item])]
    
    def updateimg(self, ):
        if self.item:
            self.image = picscale(picload(os.path.join(current_dir, f'assets/table/cuttingboard-pixilart.png')).convert_alpha(), ONEBLOCKSIZE)
            item_rect = pygame.Rect(ONEBLOCK / 4, ONEBLOCK / 4, ONEBLOCK / 2, ONEBLOCK / 2)
            item_surface = picscale(self.pics[self.item], (ONEBLOCK / 2, ONEBLOCK / 2))
            self.image.blit(item_surface, item_rect)
            cut_rect = pygame.Rect(0,ONEBLOCK/4, ONEBLOCK, ONEBLOCK )
            cut_surface = picscale(self.frames[self.e_time%3], (ONEBLOCK , ONEBLOCK ))
            self.image.blit(cut_surface,cut_rect)
        else:
            self.image = picscale(picload(os.path.join(current_dir, f'assets/table/cuttingtable.png')).convert_alpha(), ONEBLOCKSIZE)
    
    def update(self, player, keys, nowtime) -> None:
        if keys:
            if player.rect.move(player.direction[0] * ONEBLOCK / 2,
                                player.direction[1] * ONEBLOCK / 2).colliderect(self.rect):             # 如果玩家手中有物品
                if player.item:
                    if 'BC' in player.item:
                    # 如果玩家有可以切的东西，就放进去切
                        if self.item is None:
                            pygame.event.post(pygame.event.Event(BEGINCUTTING_EVENT,{'item':player.item}))
                            self.item, player.item = player.item, self.item
                            player.cutting = True
                            self.cuttingplayer = player
                            player.updateimg()
                            self.updateimg()

                            self.start_time = nowtime
                            self.timer = self.cuttingtime
                    elif player.dish and self.item:
                        tmp = self.isnewthing(player.item)
                        if tmp:
                            pygame.event.post(pygame.event.Event(PUTTHING_DISH_EVENT, {'item': tmp}))
                            player.item = tmp
                            self.item = None
                            player.updateimg()
                            self.updateimg()
                else:
                    #没东西可以把东西拿出来
                    if not player.cutting:
                        if self.item:
                            if player.dish:
                                pygame.event.post(pygame.event.Event(PUTTHING_DISH_EVENT, {'item': self.item}))                            
                            player.exchangeitem(self)

                            #self.item, player.item = player.item, self.item
                        
                            player.updateimg()
                            self.updateimg()

        if self.item and self.timer > 0:
            elapsed_time = nowtime - self.start_time
            self.e_time = nowtime %3
            remaining_time = self.timer - elapsed_time

            if remaining_time <= 0:
                self.cuttingplayer.cutting = False
                if 'BC' in self.item:
                    self.item=self.item.replace('BC','AC')
                    
                    pygame.event.post(pygame.event.Event(CUTTINGDOWN_EVENT,{'item':self.item}))
                    self.image = picscale(picload(os.path.join(current_dir, f'assets/table/cuttingboard-pixilart.png')).convert_alpha(),
                                          ONEBLOCKSIZE)
                    item_rect = pygame.Rect(ONEBLOCK / 4, 
                                            ONEBLOCK / 4, 
                                            ONEBLOCK / 2, 
                                            ONEBLOCK / 2)
                    item_surface = picscale(self.pics[self.item], (ONEBLOCK / 2, ONEBLOCK / 2))
                    self.image.blit(item_surface, item_rect)
                self.timer=0

            else:
                self.updateimg()
                progress_bar_rect = pygame.Rect(0, 0, ONEBLOCK * (1 - remaining_time / self.timer), ONEBLOCK / 8)
                progress_bar_surface = pygame.Surface(progress_bar_rect.size)
                progress_bar_surface.fill((255, 0, 0))
                self.image.blit(progress_bar_surface, progress_bar_rect)
    
    def availbeinter(self,player):
        if player.rect.move(player.direction[0] * ONEBLOCK / 2,
                            player.direction[1] * ONEBLOCK / 2).colliderect(self.rect):  # 如果玩家手中有物品
            if player.item:
                if 'BC' in player.item:
                    # 如果玩家有可以切的东西，就放进去切
                    return True
                elif player.dish and self.item:
                    if frozenset([self.item,player.item]) in RECIPE:
                        return True
            else:
                # 没东西可以把东西拿出来
                if not player.cutting:
                    if self.item:
                        return True
        return False


class Pot(pygame.sprite.Sprite):

    def __init__(self, x, y,  itempics):
        super().__init__()
        self.item = None  # 设置物品属性，默认为 None
        self.pics = itempics
        self.timer = 0
        self.remaining_time = -1
        self.start_time = 0
        self.updateimg()
        self.rect = self.image.get_rect()  # 获取图片的矩形区域
        self.rect.x = x  # 设置矩形区域的位置
        self.rect.y = y
        self.cookingtime = 20

    @property
    def is_empty(self):
        return self.item is None
    
    @property
    def is_ready(self):
        return self.item is not None and 'raw' not in self.item

    @property
    def is_cooking(self):
        # print('cooking!')
        return self.item is not None and 'raw' in self.item and self.timer > 0
    
    def updateimg(self):
        self.image = picscale(picload(os.path.join(current_dir, f'assets/cook/cook.png')).convert_alpha(), ONEBLOCKSIZE)  #
        if self.item:
            item_rect = pygame.Rect(ONEBLOCK / 4, ONEBLOCK / 4, ONEBLOCK / 2, ONEBLOCK / 2)
            item_surface = picscale(self.pics[self.item], (ONEBLOCK / 2, ONEBLOCK / 2))
            self.image.blit(item_surface, item_rect)

    def isnewthing(self,item):
        pygame.event.post(pygame.event.Event(TRY_NEWTHING_EVENT))
        if frozenset([self.item,item]) in RECIPE:
            pygame.event.post(pygame.event.Event(MADE_NEWTHING_EVENT,{'newitem':RECIPE[frozenset([self.item,item])]}))
            return RECIPE[frozenset([self.item,item])]
        
    def update(self, player, keys, nowtime) -> None:
        if keys:
            if player.rect.move(player.direction[0] * ONEBLOCK / 2,player.direction[1] * ONEBLOCK / 2).colliderect(self.rect):             # 如果玩家手中有物品
                if player.item:
                    if 'raw' in player.item:
                    # 如果玩家有可以烧的东西，就放进去烧
                        if self.item is None:
                            pygame.event.post(pygame.event.Event(BEGINCOOKING_EVENT,{'item':player.item}))
                            self.item, player.item = player.item, self.item
                            player.updateimg()
                            self.updateimg()
                            self.start_time = nowtime
                            self.timer = self.cookingtime
                    elif player.dish and self.item:
                        tmp = self.isnewthing(player.item)
                        if tmp:
                            pygame.event.post(pygame.event.Event(PUTTHING_DISH_EVENT, {'item': tmp}))
                            player.item = tmp
                            self.item = None
                            player.updateimg()
                            self.updateimg()
                else:
                    #没东西可以把东西拿出来#7.18修改如果没熟不能拿出来
                    if self.item and 'raw' not in self.item:
                        if player.dish:
                            pygame.event.post(pygame.event.Event(PUTTHING_DISH_EVENT, {'item': self.item}))
                        pygame.event.post(pygame.event.Event(COOKINGOUT_EVENT,{'item':self.item}))
                        self.item,player.item = player.item,self.item
                        
                        player.updateimg()
                        self.updateimg()

        if self.item and self.timer > 0:

            elapsed_time = nowtime - self.start_time
            self.remaining_time = self.timer - elapsed_time
            if self.remaining_time <= 0:
                if 'raw' in self.item:
                    self.item=self.item.replace('raw','cooked')
                    pygame.event.post(pygame.event.Event(COOKINGDOWN_EVENT,{'item':self.item}))
                    self.updateimg()
                self.timer=0
                self.remaining_time=-1
            else:
                progress_bar_rect = pygame.Rect(0, 0, ONEBLOCK * (1 - self.remaining_time / self.timer), ONEBLOCK / 8)
                progress_bar_surface = pygame.Surface(progress_bar_rect.size)
                progress_bar_surface.fill((255, 0, 0))
                self.image.blit(progress_bar_surface, progress_bar_rect)

    def availbeinter(self,player):
        if player.rect.move(player.direction[0] * ONEBLOCK / 2,
                            player.direction[1] * ONEBLOCK / 2).colliderect(self.rect):  # 如果玩家手中有物品
            if player.item:
                if 'raw' in player.item:
                    # 如果玩家有可以烧的东西，就放进去烧
                    return True
                elif player.dish and self.item:
                    if frozenset([self.item,player.item]) in RECIPE:
                        return True

            else:
                # 没东西可以把东西拿出来#7.18修改如果没熟不能拿出来
                if self.item and 'raw' not in self.item:
                    return True
        return False


class Wall(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height):
        super().__init__()
        self.image = pygame.Surface((width, height))
        self.image.fill((0, 0, 0))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
