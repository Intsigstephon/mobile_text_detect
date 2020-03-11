#coding=utf-8
#author: stephon
#time" 2020.03.06

"""
implement two line_point genernate algo
middle_line and BresenhamLine
"""

def MiddleLine(x1,y1,x2,y2):
    """
    Middle line algo
    """
    assert(x1 <= x2)
    result = []
    y = y1
    a = y1 - y2
    b = x2 - x1
    d0 = 2 * a + b
    d1 = 2 * a
    d2 = 2 * (a + b)
    for x in range(x1, x2 + 1):
        result.append([x,y])
        if(d0 <0 ):
            y +=1            
            d0 += d2
        else:
            d0 += d1
    return result

def BresenhamLine(x1,y1,x2,y2):
    """
    Bresenham algorithm
    """
    assert(x1 <= x2)
    result = []
    y = y1
    dx = x2 - x1    
    dy = y2 - y1     
    d = 2 * dy - dx
    for x in range(x1, x2 + 1):
        result.append([x,y])
        if(d < 0):
            d += 2 * dy
        else:
            y += 1
            d += 2 * dy - 2 * dx
    return result