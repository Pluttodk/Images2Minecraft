import cv2
import numpy as np
import os
import urllib.request
from collections import Counter
import pandas as pd
import pydirectinput
from pywinauto import keyboard, mouse
import time
import tqdm
import win32api
from pynput.mouse import Listener
import pyperclip
import random
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp

# PARAMETERS! CHANGE AS YOU LIKE!!!
#######################################################
# The scale in minecraft blocks
minecraft_scale = (50,50)
# The depth of the house
depth = 10
# How many different blocks you want to use
number_of_blocks = 2
# Location of the image you want to convert
img_loc = "target/house2.png"
#######################################################

def read_block_names():
    return pd.read_csv("blocks.txt", sep="|")

def convert_name(name):
    return int(name.split(".png")[0])

block_names = list(read_block_names()["Block_names"])
mapping_blocks = pd.read_csv("blocks_id.csv")

block_ids = []
avoid_id = []
for i, name in enumerate(block_names):
    id = mapping_blocks["Name"] == name.strip()
    try:
        block_ids.append(list(mapping_blocks[id]["Item ID"])[0])
    except:
        random_block = random.choice(list(mapping_blocks["Item ID"]))
        block_ids.append(random_block)



house_org = cv2.imread(img_loc)
house = cv2.resize(house_org, minecraft_scale)
original_scale = house.shape

blocks = {}
for file in os.listdir("img/"):
    img = cv2.imread(f"img/{file}")
    if img.shape == (48,48,3) and int(file[:-4]) not in avoid_id:
        blocks[file] = img
        
# K - means
Z = house.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)
blocks_shape = (48,48)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv2.kmeans(
    Z,
    number_of_blocks,
    None,
    criteria,
    10,
    cv2.KMEANS_RANDOM_CENTERS
)


# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((house.shape))

ideal_block = []
for color in center:
    best_block = None
    best = np.inf
    canvas = np.zeros((48,48,3), np.uint8)
    canvas[:] = color
    score = [rmse(block, canvas) for block in blocks.values()]
    best_block = list(blocks.items())[np.argmin(score)]
    ideal_block.append(best_block)

scale_blocks = 48*np.array(res2.shape[:2])
blank_canvas = np.zeros((scale_blocks[0],scale_blocks[1],3))
label.resize(res2.shape[:2])

commands = np.empty(res2.shape[:2][::-1], dtype=object)
for xi, x in enumerate(range(0, scale_blocks[0], 48)):
    for yi, y in enumerate(range(0,scale_blocks[1], 48)):
        block = ideal_block[label[xi,yi]]
        id_of_block = convert_name(block[0])
        commands[yi,xi] = block_ids[id_of_block]
        
        blank_canvas[x:x+48, y:y+48] = block[1]

# Show solution
ratio = blank_canvas.shape[0]/blank_canvas.shape[1]
image_size = (500, int(500*ratio))
cv2.imshow('K-means solution',house_org)
cv2.imshow('Minecraft_blocks',cv2.resize(np.uint8(blank_canvas), image_size, interpolation=cv2.INTER_NEAREST))
cv2.waitKey(0)
cv2.destroyAllWindows()

def get_keyboard_commands():
    cx,cy,cz = 5,-2,0
    x,y,z = cx,cy+original_scale[1],cz
    text_commands = []
    command_block_text = []
    avoid_block = commands[0,0]

    for xi, inner_commands in enumerate(commands):
        y = cy+original_scale[1]
        x += 1
        for yi, command in enumerate(inner_commands):
            y -= 1
            if command == avoid_block:
                continue
            text = f"/fill ~{x} ~{y} ~{z} ~{x} ~{y} ~{z+depth} {command} destroy"
            text_commands.append(
                text.replace(" ", "{SPACE}").replace("~", "{~}")
            )
            command_block_text.append(text)

            checks = []
            if xi+1 < original_scale[1]:
                checks.append(commands[xi+1,yi] not in (avoid_block, "minecraft:air"))
            if xi-1 >= 0:
                checks.append(commands[xi-1,yi] not in (avoid_block,  "minecraft:air"))
            if yi+1 < original_scale[0]:
                checks.append(commands[xi,yi+1] not in (avoid_block,  "minecraft:air"))
            if yi-1 >= 0:
                checks.append(commands[xi,yi-1] not in (avoid_block,  "minecraft:air"))
            
            if all(checks) and len(checks) == 4:
                clear_text = f"/fill ~{x} ~{y} ~{z+1} ~{x} ~{y} ~{z+depth-1} minecraft:air destroy"
                text_commands.append(
                    clear_text.replace(" ", "{SPACE}").replace("~", "{~}")
                )
                command_block_text.append(clear_text)
        
    clear_blocks = "/kill @e[type=item]"
    text_commands.append(clear_blocks.replace(" ", "{SPACE}"))
    command_block_text.append(clear_blocks)

    command_texts = command_blocks(command_block_text)

    print("SWITCH TO MINECRAFT!")
    time.sleep(2)

    # Set rotation
    def deploy(command_text, is_first=False):
        pyperclip.copy(command_text)
        
        if is_first:
            tp_line = "/execute as @a at @s align xyz run teleport @s ~ ~ ~ -25 45".replace(" ", "{SPACE}").replace("~", "{~}")
            pydirectinput.press("t")
            keyboard.send_keys(tp_line)
            keyboard.send_keys("~")

        pydirectinput.press("1")
        # Build command block
        pydirectinput.rightClick()
        # mouse.press(button="right")


        pydirectinput.rightClick()

        pydirectinput.keyDown('ctrl')
        pydirectinput.press('v')
        pydirectinput.keyUp('ctrl')

        keyboard.send_keys("~")

        #build button
        pydirectinput.press("2")
        pydirectinput.keyDown('shift')
        pydirectinput.rightClick()
        pydirectinput.keyUp('shift')

        pydirectinput.rightClick()

        time.sleep(6)
        pydirectinput.leftClick()
        pydirectinput.leftClick()
    
    for i in tqdm.tqdm(range(len(command_texts))):
        deploy(command_texts[i], is_first=(i == 0))
        time.sleep(2)

    



    ## Naive solution:
    #----------------------------
    # # Types it to keyboard
    # print("Switch to minecraft NOW!!!!!")
    # time.sleep(5)
    # for i in tqdm.tqdm(range(len(text_commands))):
    #     line = text_commands[i]
    #     # Press t
    #     pydirectinput.press("t")
    #     # press multiple buttons equal to our line
    #     keyboard.send_keys(line)
    #     # press enter
    #     keyboard.send_keys('~')

def command_blocks(list_of_text):
    all_messages = []
    clear_blocks = "/kill @e[type=item]"
    start =  r"summon falling_block ~ ~1 ~ {Time:1,BlockState:{Name:activator_rail},Passengers:["
    end = "{id:command_block_minecart,Command:'/setblock ~0 ~0 ~ lava[level=6]'}]}"
    command_message = start
    for line in list_of_text:
        command_message += (
            r"{id:command_block_minecart,Command:'"
            + line
            + "'},"
        )
        if len(command_message) > 29_000:
            command_message += (
                r"{id:command_block_minecart,Command:'"
                + clear_blocks
                + "'},"
            )
            all_messages.append(command_message+end)
            command_message = start
    all_messages.append(command_message+end)
    return all_messages
get_keyboard_commands()

