from main import *
from model_parameters import *
from PIL import ImageTk, Image
from tkinter import filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)

temp = np.array(0)
# temp2 = 0

def openfilename():
    filename = filedialog.askopenfilename(title ='"pen')
    return filename

def open_img():
    
    global temp
    x = openfilename()
 
    temp = loadImage(x)
    img = Image.fromarray(np.uint8(temp*255))
    
    plot(img, 2, 2)
    
    img = img.resize((250, 250), Image.ANTIALIAS)
 
    img = ImageTk.PhotoImage(img)
  
    panel = Label(root, image = img)
     
    panel.image = img
    panel.grid(row = 2)

def run_img():
#     temp2
    temp = temp.resize((28, 28), Image.ANTIALIAS)
    temp = preprocess(temp)
    temp2 = forward(temp)
    
    print(int(temp2))
    
root = Tk()
 
root.title("Image Loader")
 
root.geometry("1200x1200+500+500")
 
root.resizable(width = True, height = True)
 
btn = Button(root, text ='open image', command = open_img).grid(row = 1, columnspan = 4)

btn2 = Button(root, text ='run image', command = run_img).grid(row = 10, columnspan = 4)