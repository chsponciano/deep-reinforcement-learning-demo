'''
Authors: Carlos Henrique Ponciano da Silva & Vinicius Luis da Silva
Implementation of the Reinforcement Learning course with Deep Learning, PyTorch and Python offered by Jones Granatyr
https://www.udemy.com/course/aprendizagem-reforco-deep-learning-pytorch-python/

Map and autonomous car creation
'''
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

# variable that stores the learning
brain = None

# Does not allow adding a red dot to the stage when it is clicked with the right mouse button
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# The variables last_x and last_y are used to keep the last point in memory when drawing sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# action = 0 => no rotation, action = 1 => rotates 20 degrees, action = 2 => rotates -20 degrees
action_rotation = list([0, 20, -20])

# initialization of the last reward
last_reward = 0

# initialization of the average value of the rewards (sliding window) with respect to time
scores = list()

# used to initialize the map only once
first_update = True

# Initialization of the last distance indicating the distance to the destination
last_distance = 0

# Map initialization
def init():
    # sand is represented by a vector that has the same number of pixels as the complete interface - 1 if there is sand and 0 if there is no sand
    global sand
    sand = np.zeros((longueur, largeur))

    # x coordinate of the objective (where the car is going, from the airport to the center or the other way around)
    global goal_x
    goal_x = 20

    # y coordinate of the objective (where the car is going, from the center to the airport or the other way around)
    global goal_y
    goal_y = largeur - 20
    
    global first_update 
    first_update = False

# Creation of the car class (to better understand "NumericProperty" and "ReferenceListProperty", see: https://kivy.org/docs/tutorials/pong.html)
class Car(Widget):
    # car angle boot
    angle = NumericProperty(0)

    # initialization of the last car rotation (after an action, the car rotates 0, 20 or -20 degrees)
    rotation = NumericProperty(0)

    # speed coordinate initialization
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    # initialization of the x coordinate of the first sensor (front)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)

    # initialization of the x coordinate of the second sensor (30 degrees to the left)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)

    # initialization of the x coordinate of the third sensor (30 degrees to the right)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)

    # initialization of the signal received by sensor 1, 2 and 3
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def get_sensor_value(self, _rate=0):
        _angle = self.angle if _rate == 0 else (self.angle + _rate) % 360
        return Vector(30, 0).rotate(_angle) + self.pos

    def get_signal_value(self, x, y):
        global sand
        _x, _y = int(x), int(y)
        return int(np.sum(sand[_x-10:_x+10, _y-10:_y+10])) / 400.

    def validates_collision_area(self):
        if self.sensor1_x > longueur - 10 or self.sensor1_x < 10 or self.sensor1_y > largeur - 10 or self.sensor1_y < 10:
            self.signal1 = 1.
        if self.sensor2_x > longueur - 10 or self.sensor2_x < 10 or self.sensor2_y > largeur - 10 or self.sensor2_y < 10:
            self.signal2 = 1.
        if self.sensor3_x > longueur - 10 or self.sensor3_x < 10 or self.sensor3_y > largeur - 10 or self.sensor3_y < 10:
            self.signal3 = 1.

    def move(self, rotation):
        # updates the car's position according to its last position and speed
        self.pos = Vector(*self.velocity) + self.pos 
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        
        # updates the position of the sensors
        self.sensor1 = self.get_sensor_value()
        self.sensor2 = self.get_sensor_value(_rate=30)
        self.sensor3 = self.get_sensor_value(_rate=-30)

        # calculates the signal received from the sensor 1, 2 and 3
        self.signal1 = self.get_signal_value(self.sensor1_x, self.sensor1_y)
        self.signal2 = self.get_signal_value(self.sensor2_x, self.sensor2_y)
        self.signal3 = self.get_signal_value(self.sensor3_x, self.sensor3_y)

        self.validates_collision_area()

class Sensor1(Widget): # sensor 1 (veja https://kivy.org/docs/tutorials/pong.html)
    pass

class Sensor2(Widget): # sensor 2 (veja https://kivy.org/docs/tutorials/pong.html)
    pass

class Sensor3(Widget): # sensor 3 (veja kivy https://kivy.org/docs/tutorials/pong.html)
    pass

# Creation of the class to "play" (for "ObjectProperty" see kivy https://kivy.org/docs/tutorials/pong.html)
class Game(Widget):
    car = ObjectProperty(None)
    sensor1 = ObjectProperty(None)
    sensor2 = ObjectProperty(None)
    sensor3 = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt):
        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur

        longueur = self.width
        largeur = self.height

        if first_update:
            init()

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        action = brain.update(last_reward, last_signal)
        scores.append(brain.score())
        
        # converts the current action (0, 1 or 2) to the rotation angles (0 °, 20 ° or -20 °)
        rotation = action_rotation[action]
        self.car.move(rotation)
        
        # calculates the new distance between the car and the objective
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        
        self.sensor1.pos = self.car.sensor1
        self.sensor2.pos = self.car.sensor2
        self.sensor3.pos = self.car.sensor3

        if sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            last_reward = -1
        else:
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)
            last_reward = -0.2
            if distance < last_distance:
                last_reward = 0.1

        if self.car.x < 10:
            self.car.x = 10
            last_reward = -1
        if self.car.x > self.width - 10:
            self.car.x = self.width - 10
            last_reward = -1
        if self.car.y < 10:
            self.car.y = 10
            last_reward = -1
        if self.car.y > self.height - 10:
            self.car.y = self.height - 10
            last_reward = -1

        if distance < 100:
            goal_x = self.width-goal_x 
            goal_y = self.height-goal_y 
        
        last_distance = distance

# Graphical interface (see https://kivy.org/docs/tutorials/firstwidget.html)
class MyPaintWidget(Widget):
    # add sand when we left click
    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1

    # adds sand when we move the mouse while pressing
    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
            last_x = x
            last_y = y

# API and interface (see https://kivy.org/docs/tutorials/pong.html)
class CarApp(App):
    title = 'Autonomous car by reinforcement'

    # building the app
    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)

        text = Label(text='Draw with the click of the mouse on the', font_size='20sp', pos=(490, 20))
        parent.add_widget(text)  
        
        text = Label(text='sand fields scene so that the tick can dodge', font_size='20sp', pos=(490, 0))
        parent.add_widget(text)

        return parent

    # clear button
    def clear_canvas(self, obj): 
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur, largeur))

    # save button
    def save(self, obj): 
        print("saving brain...")
        global brain
        brain.save()
        plt.plot(scores)
        plt.show()

    # load button
    def load(self, obj): 
        print("loading last saved brain...")
        global brain
        brain.load()

    def add_brain(self, dqn):
        global brain
        brain = dqn
    