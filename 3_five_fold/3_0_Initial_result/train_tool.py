import numpy as np
import os

def CountScore(ScoreList):
    '''
    獲得一個list中的平均值、最大值、最小值
    '''
    avr_Score = sum(ScoreList) / len(ScoreList)
    highest_Score = max(ScoreList)
    lowest_Score = min(ScoreList)
    avr_highest_lowest_score = [avr_Score, highest_Score, lowest_Score]
    
    return avr_highest_lowest_score

def Format_2f(my_list):
    '''
    list不能直接在輸出的時候轉format小數點後2位
    所以值算好後先一併轉
    '''
    formatted_list = ["{:.2f}".format(round(item+0.001, 2)) for item in my_list]

    return formatted_list