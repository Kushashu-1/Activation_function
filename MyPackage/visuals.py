import matplotlib.pyplot as plt

def draw_graph(x,y,name):
    fig ,p= plt.subplots()
    p.plot(x,y)
    plt.figure(figsize=(10, 4))
    plt.xlabel(' X Axis',fontsize = 20)
    plt.ylabel(' Y Axis',fontsize = 20)
    return fig
    



