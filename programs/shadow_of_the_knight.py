'''
Batman will look for the hostages on a given building by jumping from one window to another using his grapnel gun. Batman's goal is to jump to the window where the hostages are located in order to disarm the bombs. Unfortunately he has a limited number of jumps before the bombs go off...
 	Rules
Before each jump, the heat-signature device will provide Batman with the direction of the bombs based on Batman current position:
U (Up)
UR (Up-Right)
R (Right)
DR (Down-Right)
D (Down)
DL (Down-Left)
L (Left)
UL (Up-Left)

Your mission is to program the device so that it indicates the location of the next window Batman should jump to in order to reach the bombs' room as soon as possible.

Buildings are represented as a rectangular array of windows, the window in the top left corner of the building is at index (0,0).
'''
import sys
import math

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

# w: width of the building.
# h: height of the building.
w, h = [int(i) for i in input().split()]
n = int(input())  # maximum number of turns before game over.
x0, y0 = [int(i) for i in input().split()]

x1,y1=0,0
x2,y2=w-1,h-1
# game loop
while True:
    bomb_dir = input()  # the direction of the bombs from batman's current location (U, UR, R, DR, D, DL, L or UL)

    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr, flush=True)
    if bomb_dir.__contains__('U'):
        y2 = y0-1
    elif bomb_dir.__contains__('D'):
        y1 = y0+1
    if bomb_dir.__contains__('L'):
        x2=x0-1
    elif bomb_dir.__contains__('R'):
        x1=x0+1

    x0 = x1+(x2-x1)//2
    y0 = y1+(y2-y1)//2

    # the location of the next window Batman should jump to.
    print(x0,y0)
