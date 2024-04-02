# import necessary libraries
import codey, rocky

def get_rgb_values():
    red = rocky.color_ir_sensor.get_red()
    green = rocky.color_ir_sensor.get_green()
    blue = rocky.color_ir_sensor.get_blue()

    return (red, green, blue)

# continuously print out the RGB values detected by the sensor
while True:
    rgb_values = get_rgb_values()
    if(rgb_values[0]==255 and rgb_values[1] == 255 and rgb_values[2] == 255):
        print("white")
    elif(rgb_values[0]==0 and rgb_values[1] == 0 and rgb_values[2] == 0):
        print("black")
    else:
        print(rgb_values[0])
        print(rgb_values[1])
        print(rgb_values[2])