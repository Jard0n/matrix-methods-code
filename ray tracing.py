import pygame as pg
import numpy as np
import time
import math

pg.init()
screen = pg.display.set_mode((400,300))
clock = pg.time.Clock
running = True
fovx = math.pi / 2
fovy = math.pi *.375
indexAir = 1
indexLens = 10
#an array representing the screen of 12 pixels. Each pixel is a vector of length 6 representing xy angle (theta), z angle (phi), x, y, and z positions, and alpha insensity value
pixels = np.array([[[0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0]],
                   [[0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0]],
                   [[0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0]]], dtype = float)

collisions = np.array([[[-1],[-1],[-1],[-1]],
                       [[-1],[-1],[-1],[-1]],
                       [[-1],[-1],[-1],[-1]]], dtype = list)

done = np.array([[0,0,0,0],
                [0,0,0,0],
                [0,0,0,0]], dtype = bool)

intensities = np.array([[255,255,255,255],
                [255,255,255,255],
                [255,255,255,255]], dtype = int)
#a = np.array([[1,1],[3,1]])
#b = np.array([[1,0],[1,0]])
#c = np.matmul(a,b)
#print(c)

#theta phi, x, y, z, size, type. the last 6 entries are the hard-coded walls, assumed to be 100% reflective
objects = np.array([[1,1,0,0,0,0,0], [0,0,0,0,0,0,0], [0,0,0,0,0,0,0], [0,0,0,0,0,0,0],
[0,math.pi/2,0,-1,0,300,4],[0,math.pi/2,0,400,0,300,4],[math.pi/2,math.pi/2,150,0,0,400,4],[math.pi/2,math.pi/2,-150,0,0,400,4],[0,0,0,0,113,400,4],[0,0,0,0,-113,400,4]],dtype=float)

def wallCollision(ray, object):
    '''vec1 = np.array([ray[2] - object[2],ray[0]+object[0]+math.pi/2])
    wallMatrix = np.array([[1,0],[0,1]])
    vec2 = np.matmul(wallMatrix,vec1)
    #if(object[1] != 0):
    #    ray[0] = 2 * object[0] - ray[0]
    #ray[1] = 2 * object[1] - ray[1]
    ray[0] += vec2[1]
    ray[2] = vec2[0]'''
    if object[1] == 0:
        zvec1 = np.array([ray[4], - ray[1] + 2 * object[1]])
        wallMatrix = np.array([[1,0],[0,1]])
        zvec2 = np.matmul(wallMatrix,zvec1)
        ray[1] = zvec2[1]
        ray[4] = zvec2[0]
    else: #object[0] == 0:
        yvec1 = np.array([ray[3], -1 * ray[0] + math.pi + 2 * object[0]])
        #print (yvec1)
        wallMatrix = np.array([[1,0],[0,1]])
        yvec2 = np.matmul(wallMatrix,yvec1)
        ray[0] = yvec2[1]
        ray[3] = yvec2[0]
    return ray

def lensCollision(ray, object):
    xvec1 = np.array([object[2] - ray[2], ray[0]])
    zvec1 = np.array([object[4] - ray[4], ray[1]])
    matrix1 = np.array([[1,0],[(indexLens-indexAir)/(indexAir * object[5]),indexLens/indexAir]])
    matrix2 = np.array([[1,4],[0,1]])
    matrix3 = np.array([[1,0],[(indexAir-indexLens)/(indexLens * object[5]),indexAir/indexLens]])
    lensMatrix = np.matmul(matrix1,matrix2)
    lensMatrix = np.matmul(lensMatrix, matrix3)
    print(lensMatrix)
    print(xvec1)
    xvec2 = np.matmul(lensMatrix, xvec1)
    print(xvec2[1])
    zvec2 = np.matmul(lensMatrix, zvec1)
    if ray[3] < object[3]:
        ray[3] += object[5] * 0.3
    else:
        ray[3] -= object[5] * 0.3
    ray[2] = xvec2[0] + object[2]
    ray[4] = zvec2[0] + object[4]
    ray[0] = xvec2[1]
    ray[1] = zvec2[1]
    return ray

#sets the correct inital angle and position values for each pixel depending on position on the screen
for i in range(pixels.shape[0]):
    for j in range(pixels.shape[1]):
        pixels[i, j, 0] = (-1 * fovx / 2) + (j + 1) * (fovx / 5)
        pixels[i, j, 1] = (-1 * fovy / 2) + (i + 1) * (fovy / 4)
        #pixels[i, j, 2] = -150 + 100 * j
        #pixels[i, j, 4] = 100 - 100 * i
'''
for i in range(pixels.shape[0]):
    for j in range(pixels.shape[1]):
        pixels[i, j, 0] = 0
        pixels[i, j, 1] = 0
        pixels[i,j,2] = -15 + 10 * i
'''
def dotProduct(vec1, vec2):
    return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]
def crossProduct(vec1, vec2):
    output = np.array([0,0,0])
    output[0] = vec1[1]*vec2[2] - vec1[2]*vec2[1]
    output[1] = vec1[0]*vec2[2] - vec1[2]*vec2[0]
    output[2] = vec1[0]*vec2[1] - vec1[1]*vec2[0]
    return output

def normalize(vec):
    output = np.array([0,0,0], dtype = float)
    for i in range(3):
        if(math.sqrt(math.pow(vec[0], 2) + math.pow(vec[1], 2) + math.pow(vec[2], 2))) != 0:
            output[i] = vec[i] / (math.sqrt(math.pow(vec[0], 2) + math.pow(vec[1], 2) + math.pow(vec[2], 2)))
    return output
    

def shift(object):
    #x1, y1, z1, a
    output = np.array([0,0,0,0,0,0,0], dtype = float)
    if object[6] == 0:
        for i in range(4):
            output[i+2] = object[i+2]
        output[3] = object[3]+ 20
        '''vec1 = np.array([math.cos(object[0]) * math.cos(object[1]), math.sin(object[0]) * math.cos(object[1]), math.sin(object[1])])
        vec2 = np.array([math.cos(object[0] + math.pi / 2) * math.cos(object[1]), math.sin(object[0]) * math.cos(object[1]), 0])
        vec3 = crossProduct(vec1, vec2)
        for i in range(3):
            output[0, i] = normalize(vec3)[i]
            output[1, i] = normalize(vec3)[i]
            output[2, i] = normalize(vec1)[i]
            output[3, i] = normalize(vec1)[i]
            output[4, i] = normalize(vec2)[i]
            output[5, i] = normalize(vec2)[i]
        output[0, 3] = -0.05 * object[5]
        output[1, 3] = 0.05 * object[5]
        output[2, 3] = -1 * object[5]
        output[3, 3] = object[5]
        output[4, 3] = -1 * object[5]
        output[5, 3] = object[5]'''
    return output
        

def collision(pixels, object):
    output = np.array([[-1,-1,-1,-1],
                     [-1,-1,-1,-1],
                     [-1,-1,-1,-1]])
    coords = shift(object)
    if object[5] <= 0:
        return output
    if object[6]< 0.1:
        for i in range(pixels.shape[0]):
            for j in range(pixels.shape[1]):
                '''if i == 1 and j == 0 and object[1] == 1 and (math.pow(object[2] - pixels[i,j,2],2) + math.pow(object[4] - pixels[i,j,4], 2)) - (object[5] * (pixels[i,j,3] - object[3] + 1)) >= 0:
                    print(object)
                    print(pixels[i,j])
                    print((math.pow(object[2] - pixels[i,j,2],2) + math.pow(object[4] - pixels[i,j,4],2)) + object[5] * (pixels[i,j,3] - object[3] - 1))
                    time.sleep(0.1)
'''
                if ((math.pow(object[2] - pixels[i,j,2],2) + math.pow(object[4] - pixels[i,j,4], 2)) + math.pow(object[3] - pixels[i,j,3] + 0.87 * object[5],2)) <= math.pow(object[5],2) and (math.pow(object[2] - pixels[i,j,2],2) + math.pow(object[4] - pixels[i,j,4], 2)) + math.pow(object[3] - pixels[i,j,3] - 0.87 * object[5],2) <= math.pow(object[5],2) :
                    output[i,j] = 0
                    #time.sleep(0.1)
    elif object[6] == 1:
        print ("box collision")
    elif object[6] == 2:
        print ("sphere collision")
    elif object[6] == 3:
        for i in range(pixels.shape[0]):
            for j in range(pixels.shape[1]):
                if math.pow(object[2]-pixels[i,j,2],2) + math.pow(object[3]-pixels[i,j,3],2) + math.pow(object[4]-pixels[i,j,4],2) <= math.pow(object[5],2):
                    output[i,j] = 3
                    #print ("light collision")
    elif object[6] == 4:
        for i in range(pixels.shape[0]):
            for j in range(pixels.shape[1]):
                if object[2] > 0.5:
                    if (pixels[i,j,2]) > object[2]:
                        #print("a")
                        output[i,j] = 4
                if object[2] < -0.5:
                    if (pixels[i,j,2]) < object[2]:
                        #print("b")
                        output[i,j] = 4
                if object[4] > 0.5:
                    if (pixels[i,j,4]) > object[4]:
                        #print("c")
                        output[i,j] = 4
                if object[4] < -0.5:
                    if (pixels[i,j,4]) < object[4]:
                        #print("d")
                        output[i,j] = 4
                if object[3] > 0.5:
                    if (pixels[i,j,3]) > object[3]:
                        output[i,j] = 4
                if object[3] < -0.5:
                    if (pixels[i,j,3]) < object[3]:
                        output[i,j] = 4

    return output

def step(pixel):
    x = pixel[0]
    y = pixel[1]
    pixels[x, y, 2] += .1 * (math.sin(pixels[x, y, 0]) * (1 + math.cos(pixels[x, y, 1])))
    pixels[x, y, 3] += .1 * (math.cos(pixels[x, y, 0]) * (1 + math.cos(pixels[x, y, 1])))
    pixels[x, y, 4] += .1 * (math.sin(pixels[x, y, 1]))

def drawPixel(pixel):
    y = pixel[0]
    x = pixel[1]
    left = 100 * x
    top = y * 100
    width = 100
    height = 100
    return [left, top, width, height]

def drawAtMouse(obj_type, size):
    global placed_objects
    global objects
    pos = pg.mouse.get_pos()
    if int(height) >= -255 and int(height) <= 255:
        color = pg.Color(0, 255-int((255 + height)/2), int((height + 255)/2))
    elif int(height) < 0:
        color = pg.Color(0, 255, 0)
    else:
        color = pg.Color(255,0,0)
    if pos[1] > (30 + size/2):
        font = pg.font.SysFont("Arial", size)
        if(obj_type == 0):
            pg.draw.arc(screen,color,(pos[0] - .8 * size, pos[1] - size/2,size,size),-math.pi/4,math.pi/4,1)
            pg.draw.arc(screen,color,(pos[0] - .2 * size, pos[1] - size/2,size,size),3 * math.pi/4, 5 * math.pi/4,1)
            #img = pg.font.Font.render(font, "()", 1, color)
            #screen.blit(img, (pos[0] - size/4, pos[1] - size * 0.6))
        elif(obj_type == 1):
            img = pg.font.Font.render(font, "□", 1, color)
            screen.blit(img, (pos[0] - size/2, pos[1] - size))
        elif(obj_type == 2):
            img = pg.font.Font.render(font, "○", 1, color)
            screen.blit(img, (pos[0] - size/2, pos[1] - size))
        elif(obj_type == 3):
            img = pg.font.Font.render(font, "☼", 1, color)
            screen.blit(img, (pos[0] - size/2, pos[1] - size*.6))
        #img = pg.transform.rotate(img, angle)
        
def drawObjects():
    for i in range(objects.shape[0]):
        if int(objects[i,4]) >= -255 and int(objects[i,4]) <= 255:
            color = pg.Color(0, 255-int((255 + objects[i,4])/2), int((objects[i,4] + 255)/2))
        elif int(objects[i,4]) < 0:
            color = pg.Color(0, 255, 0)
        else:
            color = pg.Color(255,0,0)
        if objects[i][5] == 0:
            continue
        elif objects[i][6] == 4:
            continue
        else:
            font = pg.font.SysFont("Arial", int(2 * objects[i][5]))
        
        if objects[i][6] == 0:
            #img = pg.font.Font.render(font, "()", 1, color)
            #screen.blit(img, (objects[i][3] - objects[i,5]/2, objects[i][2] - objects[i,5]*1.2 + 150))
            pg.draw.arc(screen,color,(objects[i][3] - .8 * objects[i,5], objects[i][2] - objects[i,5]/2 + 150,objects[i,5],objects[i,5]),-math.pi/4,math.pi/4,1)
            pg.draw.arc(screen,color,(objects[i][3] - .2 * objects[i,5], objects[i][2] - objects[i,5]/2 + 150,objects[i,5],objects[i,5]),3 * math.pi/4, 5 * math.pi/4,1)
            pg.display.flip()
        elif objects[i][6] == 1:
            img = pg.font.Font.render(font, "□", 1, color)
        elif objects[i][6] == 2:
            img = pg.font.Font.render(font, "○", 1, color)
        elif objects[i][6] == 3:
            img = pg.font.Font.render(font, "☼", 1, color)
            screen.blit(img, (objects[i][3] - objects[i,5], objects[i][2] - objects[i,5]*1.2 + 150))
        #img = pg.transform.rotate(img, objects[i, 0])
        #pg.draw.line(screen,"black",(0,objects[i,2] + 150),(400,objects[i,2] + 150),2)
        #pg.draw.line(screen,"black",(objects[i,3],0),(objects[i,3],300),2)
        #pg.draw.line(screen,"black",(0,objects[i,2] - objects[i,5] + 150),(400,objects[i,2] - objects[i,5] + 150),2)
        #pg.display.flip()
        #screen.blit(img, (objects[i][3] - objects[i,5]/2, objects[i][2] - objects[i,5]*1.2 + 150))
        

def drawRays(previous):
    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            if int(pixels[i][j][4]) >= -255 and int(pixels[i][j][4]) <= 255:
                color = pg.Color(0, 255-int((255 + pixels[i][j][4])/2), int((pixels[i][j][4] + 255)/2))
            elif int(pixels[i][j][4]) < 0:
                color = pg.Color(0, 255, 0)
            else:
                color = pg.Color(255,0,0)
            #print(color)
            pg.draw.line(screen, color, (previous[i][j][3], previous[i][j][2] + 150), (pixels[i][j][3], pixels[i][j][2] + 150), 2)
    pg.display.flip()

state = "setup"
size = 30
height = 0
angle = 0
placed_objects = 0
obj_type = 0
t=0
colliding = False

while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
    if (state == "setup"):
        screen.fill("white")
        mousepos = (pg.mouse.get_pos())
        mousex = mousepos[0]
        mousey = mousepos[1]
        pg.font.init()
        txt_font = pg.font.SysFont("Arial", 15)
        txt = pg.font.Font.render(txt_font, "Instructions                Lens         Box         Sphere         Light               Done", 1, "black")
        screen.blit(txt, (10, 10))
        pg.draw.line(screen, "black", (0,35),(400,35), 2)
        pg.draw.line(screen, "black", (105,0),(105,35), 2)
        pg.draw.line(screen, "black", (160,0),(160,35), 2)
        pg.draw.line(screen, "black", (210,0),(210,35), 2)
        pg.draw.line(screen, "black", (275,0),(275,35), 2)
        pg.draw.line(screen, "black", (330,0),(330,35), 2)
        for event in pg.event.get():
            pos = pg.mouse.get_pos()
            if event.type == pg.MOUSEWHEEL:
                if (size + event.y) > 3:
                    size += event.y
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_DOWN and height > -110:
                    height -= 30
                elif event.key == pg.K_UP and height < 110:
                    height += 30
                elif event.key == pg.K_LEFT:
                    angle += 5
                elif event.key == pg.K_RIGHT:
                    angle -= 5
            elif pg.mouse.get_pressed(num_buttons=3)[0] == True:
                time.sleep(0.1)
                if pos[1] < 35:
                    if pos[0] < 105:
                        time.sleep(.1)
                        pg.event.get()
                        while pg.mouse.get_pressed(num_buttons=3)[0] == False:
                            screen.fill("white")
                            txt_font = pg.font.SysFont("Arial", 19)
                            line0 = pg.font.Font.render(txt_font, "Place up to 4 objects then hit done to begin simulation", 1, "black")
                            line1 = pg.font.Font.render(txt_font, "Click buttons on top of screen to change object type.", 1, "black")
                            line2 = pg.font.Font.render(txt_font, "Resize objects with scroll wheel.", 1, "black")
                            line3 = pg.font.Font.render(txt_font, "Raise/lower objects with U/D arrow keys.", 1, "black")
                            line4 = pg.font.Font.render(txt_font, "Once simulation is done, click anywhere to view results.", 1, "black")
                            line5 = pg.font.Font.render(txt_font, "It is recommended to make light large to ensure rays hit", 1, "black")
                            line6 = pg.font.Font.render(txt_font, "Sphere and Box not Implemented", 1, "black")
                            line7 = pg.font.Font.render(txt_font, "Click anywhere to close this message.", 1, "black")
                            screen.blit(line0, (10, 10))
                            screen.blit(line1, (10, 30))
                            screen.blit(line2, (10, 50))
                            screen.blit(line3, (10, 70))
                            screen.blit(line4, (10, 90))
                            screen.blit(line5, (10, 110))
                            screen.blit(line6, (10, 130))
                            screen.blit(line7, (10, 150))
                            pg.event.get()
                            pg.display.flip()
                        time.sleep(0.1)
                        continue
                    #lens, box, sphere, light
                    elif pos[0] > 105 and pos[0] < 160:
                        obj_type = 0
                    elif pos[0] > 160 and pos[0] < 210:
                        obj_type = 1
                    elif pos[0] > 210 and pos[0] < 275:
                        obj_type = 2
                    elif pos[0] > 275 and pos[0] < 330:
                        obj_type = 3
                    else:
                        screen.fill("white")
                        drawObjects()
                        state = "computing"
                elif (placed_objects < 4):
                    objects[placed_objects][0] = angle
                    objects[placed_objects][3] = pos[0]
                    objects[placed_objects][2] = pos[1] - 150 
                    
                    #objects[placed_objects][2] = pos[0] - size/3
                    #objects[placed_objects][3] = pos[1] - size
                    objects[placed_objects][4] = height
                    if obj_type == 0:
                        objects[placed_objects][5] = size
                    elif obj_type == 3:
                        objects[placed_objects][5] = size / 2
                    objects[placed_objects][6] = obj_type
                    placed_objects += 1
                
        drawObjects()
        if (placed_objects < 4):
           drawAtMouse(obj_type, size)
    elif state == "computing":
        t += 1
        for i in range(pixels.shape[0]):
            for j in range(pixels.shape[1]):
                for k in range(objects.shape[0]):
                    #print(i,j,k)
                    if (collision(pixels,objects[k]))[i][j] >=0 and t!=0:
                        #print (collision(pixels,objects[k]))
                        arrayLength = (collisions.shape[2]) - 1
                        if collisions[i,j,arrayLength] != -1:
                            collisions2 = np.zeros((3,4,arrayLength + 2))
                            arrayLength = (collisions2.shape[2]) - 1
                            for a in range(3):
                                for b in range(4):
                                    collisions2[a,b,arrayLength] -= 1
                                    for c in range(arrayLength):
                                        if c <= arrayLength:
                                            collisions2[a,b,c] = collisions[a,b,c]
                                        else:
                                            collisions2[a,b,c] -= 1
                            #print ("a")
                            #print (collisions2.shape)
                            collisions = collisions2

                        collisions[i,j,arrayLength] = k
                        #print(collisions[i,j])
                    if (collision(pixels,objects[k]))[i][j] == 0:
                        #pg.draw.line(screen,"black",(0,objects[k,2] + 150),(400,objects[k,2] + 150),2)
                        pg.display.flip()
                        pixels[i,j] = lensCollision(pixels[i,j], objects[k])
                    if collision(pixels,objects[k])[i][j] == 3:
                        done[i,j] = 1
                    if (collision(pixels,objects[k]))[i][j] == 4:
                        #colliding = True
                        pixels[i,j] = wallCollision(pixels[i,j], objects[k])
                        #print(pixels[i,j])
                if colliding == False:
                    previous = np.copy(pixels)
                    if done[i,j] == False:
                        step([i, j])
                        drawRays(previous)
        finished = True
        for i in range(pixels.shape[0]):
            for j in range(pixels.shape[1]):
                if done[i,j] == False:
                    finished = False
        if finished == True:
            state = "display"
    else:
        for i in range(pixels.shape[0]):
            for j in range(pixels.shape[1]):
                for k in range(collisions.shape[2]):
                    if collisions[i,j,k] == -1:
                        continue
                    elif collisions[i,j,k] == 0:
                        intensities[i,j] *= 0.5
                color = (255,255,255,intensities[i,j])
                pg.draw.rect(screen, color, drawPixel([i, j]))
                time.sleep(0.01)
                
        pg.draw.line(screen, "black", [200, 0], [200, 300], 1)
        pg.draw.line(screen, "black", [100, 0], [100, 300], 1)
        pg.draw.line(screen, "black", [300, 0], [300, 300], 1)
        pg.draw.line(screen, "black", [0, 100], [400, 100], 1)
        pg.draw.line(screen, "black", [0, 200], [400, 200], 1)
    pg.display.flip()

#print(collisions)
print(done)
pg.quit()
