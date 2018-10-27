import os
import re
import json
import itertools
import numpy as np
import pandas as pd
from datetime import datetime

def getFacebookData():
    responseDictionary = dict()
    fbFile = open('conversationData.txt', 'r') 
    nobitaMessage, doraemonMessage = "",""
    for line1,line2 in itertools.izip_longest(*[fbFile]*2):
        # luu cac cau noi cua nobita o dong le 
        # (boi vi nobita luon bat dau cuoc noi chuyen truoc nen cau noi se ow cac dong 1, 3, 5, ...)
        # muc dich la tao mot python dict co dang
        # {
        #   nobitaMessage : doraemonMessage,
        #   nobitaMessage : doraemonMessage,
        #   nobitaMessage : doraemonMessage,
        #   ....
        # }
        colon1 = line1.find(':')
        nobitaMessage = line1.split(':')[1]
        # print nobitaMessage

        # cau noi cua doraemon o dong chan
        colon2 = line2.find(':')
        doraemonMessage = line2.split(':')[1]
        # print doraemonMessage

        responseDictionary[nobitaMessage] = doraemonMessage
    return responseDictionary

dictData = getFacebookData()
np.save('/home/q/Downloads/workPlaceBackUp/doraemonBot/conversationDictionary.npy',dictData)
