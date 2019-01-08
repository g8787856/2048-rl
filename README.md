# 2048-rl

我的AI課程期末專題報告，用強化學習做一個2048遊戲代理人  
可以選擇用Qlearning或者A2C兩種算法來執行  
在終端下指令：`python main.py`就可以開始執行，預設的指令式使用Qlearning，並且載入已經訓練好的q_table

可以使用 ***-algorithm*** 來決定算法， ***-load*** 來決定是否要載入模型  
例如：  
`python main.py -algorithm a2c -load False`  
就會使用A2C算法，並且重新訓練一個模型

---

My AI course final project, I want to study reinforcement-learning, so I did a game agent research.

I choose 2048 game as environment, which based on [2048-python](https://github.com/yangshun/2048-python) by yangshun, and I try to edit it to make it suit for RL env.

I refer to [Reinforcement-learning-with-tensorflow](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow) by MorvanZhou as the Q-learning agent, and [reinforcement-learning](https://github.com/rlcode/reinforcement-learning) by rlcode as the A2C agent.

You can run `python main.py` to start the game.

## Default
It's will use Q-learning as algorithm and will load the previous model.
you can use ***-algorithm*** to decide algorithm and use ***-load*** to decide load model or not
Example:
`python main.py -algorithm a2c -load False`
It's will use A2C algorithm and retrain a model 

## 參考（References）
[1] yangshun, [2048-python](https://github.com/yangshun/2048-python)(2013).

[2] MorvanZhou, [Reinforcement-learning-with-tensorflow](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow)(2018).

[3] rlcode, [reinforcement-learning](https://github.com/rlcode/reinforcement-learning)(2017).
