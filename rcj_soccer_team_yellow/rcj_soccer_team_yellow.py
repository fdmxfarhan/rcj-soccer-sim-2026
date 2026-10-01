from controller import Robot
from my_robot import MyRobot

robot = Robot()
name = robot.getName()
robot_number = int(name[1])

if robot_number == 1:
    robot_controller = MyRobot(robot)
elif robot_number == 2:
    robot_controller = MyRobot(robot)
else:
    robot_controller = MyRobot(robot)

robot_controller.run()
