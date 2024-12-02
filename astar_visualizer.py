import tkinter as tk
from tkinter import Scale
from PIL import Image, ImageTk
import numpy as np
from Astar_plus import astar

class AStarVisualizer:
    def __init__(self, master):
        self.master = master
        self.master.title("A* Pathfinding Visualizer")

        # 加载背景图片
        self.background_image = Image.open("images/background.png")  # 确保路径正确
        self.background_photo = ImageTk.PhotoImage(self.background_image)

        # 创建 Canvas，根据图片大小初始化
        self.canvas = tk.Canvas(master, width=self.background_photo.width(), height=self.background_photo.height())
        self.canvas.pack(fill=tk.BOTH, expand=True)  # 使 Canvas 填充整个窗口并可以伸缩
        self.image_on_canvas = self.canvas.create_image(0, 0, image=self.background_photo, anchor='nw')
        self.canvas.bind("<Button-1>", self.set_point)

        # 绑定窗口大小改变事件，以便更新背景图片的显示
        master.bind("<Configure>", self.resize_image)

        # 直接加载迷宫 mask 图片和景观掩模文件
        self.load_maze("images/saved_mask_white.png")
        self.load_landscape_mask("images/landscape_array.txt")
        self.start = None
        self.end = None

        self.alpha_scale = Scale(master, from_=0, to=1, resolution=0.1, orient=tk.HORIZONTAL, label="Alpha")
        self.alpha_scale.set(0.5)
        self.alpha_scale.pack()
        self.beta_scale = Scale(master, from_=0, to=1, resolution=0.1, orient=tk.HORIZONTAL, label="Beta")
        self.beta_scale.set(0.5)
        self.beta_scale.pack()

    def resize_image(self, event):
        # Resize the image, update the canvas size as well
        new_width = event.width
        new_height = event.height
        resized_image = self.background_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.background_photo = ImageTk.PhotoImage(resized_image)
        self.canvas.config(width=new_width, height=new_height)  # Update canvas size
        self.canvas.itemconfig(self.image_on_canvas, image=self.background_photo)

    def load_maze(self, file_path):
        image = Image.open(file_path)
        image_array = np.array(image.convert("L"))  # 转换为灰度
        self.maze = (image_array > 128).astype(int)  # 假设高于128的灰度值表示通行区域

    def load_landscape_mask(self, file_path):
        self.landscape_mask = np.loadtxt(file_path, dtype=np.float32)  # 从文本文件加载景观掩模

    def set_point(self, event):
        coord = (event.y, event.x)  # 调整为行和列
        if not self.start:
            self.start = coord
            self.canvas.create_oval(event.x - 5, event.y - 5, event.x + 5, event.y + 5, fill='green', outline='green')
        elif not self.end:
            self.end = coord
            self.canvas.create_oval(event.x - 5, event.y - 5, event.x + 5, event.y + 5, fill='red', outline='red')
            self.find_path()

    def find_path(self):
        if self.start and self.end:
            alpha = self.alpha_scale.get()
            beta = self.beta_scale.get()
            path = astar(self.maze, self.start, self.end, self.landscape_mask, alpha, beta)
            if path:
                self.draw_path(path)
            else:
                print("No path found")

    def draw_path(self, path):
        for i in range(len(path) - 1):
            self.canvas.create_line(path[i][1], path[i][0], path[i+1][1], path[i+1][0], fill='blue')

    def run(self):
        self.master.mainloop()

root = tk.Tk()
app = AStarVisualizer(root)
app.run()