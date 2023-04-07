import phyre
import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

seed = 42
random.seed(seed)

eval_setup = "ball_within_template"
action_tier = phyre.eval_setup_to_action_tier(eval_setup)
train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, seed)

# Select a few tasks to get started
tasks = dev_tasks[:50]
simulator = phyre.initialize_simulator(tasks, action_tier)

# Show the first frame of the simulation for the first task
task_id = 0
initial_scene = simulator.initial_scenes[task_id]

# Show the features for the objects in the first frame
# 0 => x in pixels of center of mass divided by SCENE_WIDTH
# 1 => y in pixels of center of mass divided by SCENE_HEIGHT
# 2 => angle of the object between 0 and 2pi divided by 2pi
# 3 => diameter in pixels of object divided by SCENE_WIDTH
# 4-7 => one hot encoding of the object shape, according to order: ball, bar, jar, standing sticks
# 8-13 => one hot encoding of object color, according to order: red, green, blue, purple, gray, black
initial_featurized_objects = simulator.initial_featurized_objects[task_id]
print(initial_featurized_objects.features)

# Pick a random action after sampling 100 candidates from the uniform cube of possible actions
actions = simulator.build_discrete_action_space(max_actions=100)
action = random.choice(actions)
simulation = simulator.simulate_action(
    task_id, action, need_images=True, need_featurized_objects=True
)

# Visualize the simulation
fig, ax = plt.subplots()


def update(i):
    obs = simulation.images[i]
    img = phyre.observations_to_float_rgb(obs)
    ax.imshow(img)
    ax.set_axis_off()


def on_press(event):
    if event.key.isspace():
        if anim.running:
            anim.event_source.stop()
        else:
            anim.event_source.start()
        anim.running ^= True
    elif event.key == "left":
        anim.direction = -1
    elif event.key == "right":
        anim.direction = +1

    # Manually update the plot
    if event.key in ["left", "right"]:
        t = anim.frame_seq.next()
        update(t)
        plt.draw()


if len(simulation.images) > 1:
    fig.canvas.mpl_connect("key_press_event", on_press)
    anim = FuncAnimation(fig, update, frames=len(simulation.images), interval=50)
    plt.show()
else:
    print("Simulation failed, no images to show.")
