# ComplexOvercooked 🍳

🌐 [中文](/readme.ch.md) | [English](/readme.md) 🌐

### 简介 🎮:

-----

嗨👋，目前的开源的overcook项目是基于前后端编写的h5游戏🕹️。因为只会python🐍，考察了一下pygame，发现写这样一个游戏很简单🎉，并且可以更加还原游戏原本的一些环境🌍，更加适合python同学👥。以下是游戏场景的动态展示👀：

<p align="center">
  <img src="showpic/2player.gif" width="33%" />
  <img src="showpic/2playerhuman.gif" width="33%" />
  <img src="showpic/4player.gif" width="33%" />
</p>
<p align="center">
  <span style="display: inline-block; width: 33%; text-align: left;">2player_overcooked</span>
  <span style="display: inline-block; width: 33%; text-align: center;">2player_agentwithhuan_overcooked</span>
  <span style="display: inline-block; width: 33%; text-align: right;">4player_overcooked</span>
</p>
为了更加适配强化学习🧠，我们做了以下的改进🔧：

首先是游戏逻辑相比简化的Overcooked_AI更贴近真实的overcooked 2👨‍🍳，真实的overcooked游戏实际上是一个多人多任务的合作游戏👫👬，玩家数量最多是4个人🎮，并且同一时间智能体需要烹饪的菜品可能是多种🍲，多个智能体在完成任务时需要具有分工意识和任务分配意识🤝，这些方面的能力在简化的overcooked环境中体现得不够充分。

此外，简化的overcooked的任务步骤较为简单，例如完成洋葱汤只需要将三个洋葱放入锅中，然后烹饪，送菜🍜，这使得智能体之间的合作模型较为单一。多样的合作模式更加考验智能体之间的配合和协调🤖，也有利于训练合作能力更出色的智能体。

在工程上🔨，原始的overcooked环境的渲染使用的是前端界面，许多方法是自己重定义的，比较冗长，阅读性不佳📚。我们使用pygame实现了基本的游戏逻辑（可能在结构上并不一致，具体可以参考后文的代码结构）👨‍💻。

### 特性 🌟

----

- 原版游戏不同菜品的合成🍽️，如 切好的 柠檬+ 煎好的 鱼+盘子=香煎精鱼🐟，牛肉+番茄+汉堡胚+盘子=牛肉汉堡🍔！
- 步骤任意性🔄，我们像原版游戏一样，提供多路径同终点的合成规则设置，比如番茄汉堡+牛肉和牛肉汉堡+番茄都能合成番茄牛肉汉堡🍅🍔！
- 还原切菜动作✂️，相当于为环境增加了更复杂的中间步骤，切菜时不可移动，移动则暂停切菜！
- 菜单滚动条倒计时⏲️，支持用户自己定义，增加图片和合成菜谱即可！
- 垃圾桶🗑️，原版游戏因为有"煮糊了"这个设定，是需要提供灭火器和垃圾桶来消除烧焦物的影响的。但我们在自然人与agent的合作中发现，如果有时候想要提前备菜，容易遇到做错菜导致桌子不够用的情况，这个时候训练好的智能体就丧失了行动能力。因此还是添加了垃圾桶的设置，希望智能体可以做到有纠错能力。
- 支持用户定义游戏场景🌆，仅需要在maps.json中配置你想要的地图人数和菜谱，你就可以得到全新的环境！
- 提供多功能的强化学习接口🔌，智能体与智能体，人与智能体以及LLM控制智能体！
- 提供了一些基本的绘图接口🎨，如绘制agents的送菜成功率或者每个智能体的移动轨迹热力图等等。
### 安装 🛠️

---

你可以克隆本项目📁：
```bash
git clone https://anonymous.4open.science/r/ComplexOvercooked-1D82/readme.md
pip install -r requirements.txt
```
### 训练 🚀
举个例子, 在2playerhard中训练ippo智能体: 
```Bash
python src/main.py --config=ippo --env-config=overcooked2 with env_args.map_name=2playerhard
```
使用不同的算法在不同的游戏地图中批量训练多个随机种子：
```Bash
./runalgo.sh
```
###  ComplexOvercooked环境代码结构 📐

```
ComplexOvercooked/
├── assets/  # 游戏元素的图片资源
├── src/  # 核心源代码目录
│   ├── components/  # 基础组件
│   │   ├── action_selectors.py  # 动作选择器(epsilon-greedy等)
│   │   ├── episode_buffer.py    # 经验回放缓冲区
│   │   ├── epsilon_schedules.py # epsilon衰减调度器
│   │   ├── standarize_stream.py # 数据标准化
│   │   └── transforms.py        # 数据转换工具
│   ├── config/  # 配置文件目录
│   │   └── algs/  # 算法配置
│   │       ├── coma.yaml        # COMA算法配置
│   │       ├── ia2c.yaml        # IA2C算法配置  
│   │       ├── ippo.yaml        # IPPO算法配置
│   │       ├── iql.yaml         # IQL算法配置
│   │       └── *_ns.yaml        # 非共享参数版本的算法配置
│   └── envs/  # 环境实现
│       └── overcook_pygame/  # Overcooked环境
│           ├── overcook_gym_class.py  # 游戏基础类(桌子、锅、案板等)
│           ├── overcook_gym_env.py    # Gym环境封装
│           └── overcook_gym_main.py   # 环境主入口
├── maps.json  # 地图配置文件
├── requirements.txt  # 项目依赖
└── runalgo.sh  # 批量训练脚本
```

主要目录说明:
- `src/components/`: 包含强化学习算法的基础组件实现，如动作选择、经验回放等
- `src/config/algs/`: 包含各种算法(COMA、IA2C、IPPO、IQL等)的配置文件
- `src/envs/`: 包含Overcooked环境的具体实现
- `assets/`: 存放游戏素材资源
- `maps.json`: 用于配置游戏地图、玩家数量和任务类型
- `runalgo.sh`: 用于批量训练不同算法、不同地图的脚本
