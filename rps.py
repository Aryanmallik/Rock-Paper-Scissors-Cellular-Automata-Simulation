import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

kernel_code = """
__kernel void rps_simulation(__global int* agents, __global int* positions, __global float* directions, int num_agents, int width, int height, __global float* wiggle) {
    int i = get_global_id(0);
    if (i >= num_agents || agents[i] == -1) return;

    float angle = directions[i];

    int dx = round(cos(angle));
    int dy = round(sin(angle));

    int new_x = (positions[2 * i] + dx + width) % width;
    int new_y = (positions[2 * i + 1] + dy + height) % height;

    positions[2 * i] = new_x;
    positions[2 * i + 1] = new_y;

    directions[i] += wiggle[i];

    int neighbors[4][2] = {
        {new_x, (new_y + 1) % height},
        {new_x, (new_y - 1 + height) % height},
        {(new_x + 1) % width, new_y},
        {(new_x - 1 + width) % width, new_y}
    };

    for (int j = 0; j < num_agents; j++) {
        if (j == i || agents[j] == -1) continue;
        int other_x = positions[2 * j];
        int other_y = positions[2 * j + 1];
        int other_agent = agents[j];

        for (int k = 0; k < 4; k++) {
            if (other_x == neighbors[k][0] && other_y == neighbors[k][1]) {
                if ((agents[i] == 0 && other_agent == 2) || 
                    (agents[i] == 1 && other_agent == 0) || 
                    (agents[i] == 2 && other_agent == 1)) {
                    agents[j] = agents[i];
                    //agents[j]=-1;
                   
                }
            }
        }
    }
}
"""

platforms = cl.get_platforms()
gpu_devices = [device for platform in platforms for device in platform.get_devices(device_type=cl.device_type.GPU)]
device = gpu_devices[-1]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)
program = cl.Program(ctx, kernel_code).build()

width, height = 400,400
num_agents = 9000

types = np.array([0] * 3000+ [1] * 3000+ [2] * 3000, dtype=np.int32)
np.random.shuffle(types)

positions = np.random.choice(width * height, size=num_agents, replace=False)
positions = np.column_stack((positions % width, positions // width))

directions = np.random.uniform(0, 2 * np.pi, size=num_agents).astype(np.float32)
wiggle = np.random.uniform(-0.2, 0.2, size=num_agents).astype(np.float32)

agents_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=types)
positions_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=positions)
directions_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=directions)
wiggle_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=wiggle)

colors = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1], -1: [0, 0, 0]}

fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
ax.margins(0)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

img = ax.imshow(np.zeros((height, width, 3)), animated=True, interpolation='nearest')
text = ax.text(0.01, 0.99, "", fontsize=14, color="white", ha='left', va='top',
               transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.7, pad=5))

def update(frame):
    cl.enqueue_copy(queue, wiggle_buf, np.random.uniform(-0.2, 0.2, size=num_agents).astype(np.float32))
    program.rps_simulation(queue, (num_agents,), None, agents_buf, positions_buf, directions_buf, np.int32(num_agents), np.int32(width), np.int32(height), wiggle_buf)
    cl.enqueue_copy(queue, types, agents_buf)
    cl.enqueue_copy(queue, positions, positions_buf)
    cl.enqueue_copy(queue, directions, directions_buf)

    rock_count = np.sum(types == 0)
    paper_count = np.sum(types == 1)
    scissors_count = np.sum(types == 2)

    if paper_count== num_agents:
        print("Simulation over, paper wins")
        ani.event_source.stop()
    if scissors_count == num_agents:
        print("Simulation over, scissor wins")
        ani.event_source.stop()
    if rock_count == num_agents:
        print("Simulation over, rock wins")
        ani.event_source.stop()

    grid = np.zeros((height, width, 3))
    for i in range(num_agents):
        if types[i] != -1:
            x, y = positions[i]
            grid[y, x] = colors[types[i]]
    img.set_array(grid)

    text.set_text(f"rock(red) {rock_count}  paper(green) {paper_count}  scissor(blue) {scissors_count}")

    return [img, text]

ani = animation.FuncAnimation(fig, update, frames=144, interval=10, blit=False)
plt.show()

