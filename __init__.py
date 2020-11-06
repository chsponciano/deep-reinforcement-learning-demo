'''
Authors: Carlos Henrique Ponciano da Silva & Vinicius Luis da Silva
Implementation of the Reinforcement Learning course with Deep Learning, PyTorch and Python offered by Jones Granatyr
https://www.udemy.com/course/aprendizagem-reforco-deep-learning-pytorch-python/
'''
from reinforcement import Dqn
from scenario import CarApp


n_inputs = 5 # 5 inputs (sensors + direction)
n_outputs = 3 # 3 outputs
gamma_value = 0.9 # gamma value

if __name__ == "__main__":
    autonomous_car = CarApp()
    autonomous_car.add_brain(Dqn(n_inputs, n_outputs, gamma_value))
    autonomous_car.run()