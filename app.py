from flask import Flask, request, jsonify,render_template,abort
import numpy as np
from PIL import Image
from Astar_plus import astar
import cv2
import json

app = Flask(__name__)

# 从图像文件加载迷宫的函数
def load_maze(image_path):
    with Image.open(image_path) as img:
        img = img.convert('L')  # 将图像转换为灰度
        array = np.array(img)   # 将图像转换为 numpy 数组
        binary_array = (array == 255).astype(int)  # 将值转换为二进制（0 和 1）
        return binary_array

# 从文本文件加载景观掩膜的函数
def load_landscape_mask(file_path):
    return np.loadtxt(file_path, delimiter=' ')


@app.route('/')
def home():
    return render_template('home.html')  # 这里假设你的 HTML 文件名为 home.html


@app.route('/navigation')
def navigation():
    return render_template('navigation.html')



@app.route('/runAstar', methods=['POST'])
def runAstar():
    print("calling runAstar")
    try:
        data = request.get_json()
        print("Received data:", data)  # 打印接收到的数据
        if not data or 'start' not in data or 'end' not in data or 'alpha' not in data or 'beta' not in data:
            abort(400, description="Invalid data provided.")
        
        maze_file_path = "images/saved_mask_white.png"
        landscape_mask_file_path = "images/landscape_array.txt"
        
        maze = load_maze(maze_file_path)
        landscape_mask = load_landscape_mask(landscape_mask_file_path)
        
        start = tuple(data['start'])
        end = tuple(data['end'])
        alpha = float(data['alpha'])
        beta = float(data['beta'])
        
        # 对start 和 end 坐标的横纵坐标进行交换
        start = (start[1], start[0])
        end = (end[1], end[0])

        # 使用 A* 算法查找路径
        path = astar(maze, start, end, landscape_mask, alpha, beta)

        swapped_path = [(y, x) for x, y in path]
        # print("Return Path:", json.dumps(path, indent=4))
        return jsonify({'path': swapped_path})
    except FileNotFoundError:
        abort(404, description="File not found.")
    except Exception as e:
        abort(500, description=str(e))

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)