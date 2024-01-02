from PIL import Image
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sigma = 100

im = Image.open('tower.png').convert('L') #convert to grayscale
img = np.array(im)
h = im.height
w = im.width
tic = time.time()
#grayscale values; list of size h*w , where each values is integer in range [0,255]

intensity = list(im.getdata())
#%%
for iters in range(25):

    def idx(i,j):
        return w*(i-1) + j

    #coordinates
    coords = [(i,j) for i in range(1, h+1) for j in range(1, w+1)]

    #nodes
    nodes = [idx(i, j) for i,j in coords]

    t = h*w+1 # destination node with index h*w+1
    nodes.append(t)

    ngbrs = {idx(i,j):[idx(i+1,j2) for j2 in [j-1,j,j+1] if i+1 <= h and 1<=j2<=w]
              for (i,j) in coords}
    ngbrs[t] = []

    for j in range(1,w+1):
        ngbrs[idx(h,j)].append(t)

    edges = [(n1,n2) for n1 in nodes for n2 in ngbrs[n1]]

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    #Extract the incidence matrix
    A = nx.incidence_matrix(G, oriented=True)

    def get_weight(before_node, last_node):
        if last_node >= t:
            return 0
        else:
            return np.abs(intensity[before_node-1]-intensity[last_node-1])
        

    m = gp.Model("ImageTraversal")
    m.Params.LogToConsole = 0
    m.Params.Method = 0
    f = m.addMVar(shape=G.number_of_edges(), vtype=GRB.CONTINUOUS, lb=0, name="")
    b = np.zeros(G.number_of_nodes())
    b[idx(1,50)-1] = -1
    b[-1] = 1

    c = np.zeros(G.number_of_edges())
    for i in range(G.number_of_edges()):
        c[i] = get_weight(edges[i][0],edges[i][1])

    obj = c@f
    m.addConstr(A@f==b)
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()
    flows = m.getAttr("X", m.getVars())
    print("Primal objective: ", m.getObjective().getValue())

    flows = m.getAttr("X", m.getVars())

    to_remove = [] # get the indices of nodes (pixels) to be removed

    for i in range(G.number_of_edges()):

        if flows[i] >= 0.9:

           to_remove.append(edges[i][0]-1) #0 indexed

    intensity = [I for idx,I in enumerate(intensity) if idx not in to_remove]

    w = w - 1

    mm = gp.Model("Dual Image Traversal")
    mm.Params.LogToConsole = 0
    mm.Params.Method = 0
    p = mm.addMVar(shape=G.number_of_nodes(), vtype=GRB.CONTINUOUS, lb=0, name="z")
    obj = b@p
    mm.addConstr(A.transpose()@p <= c)
    mm.setObjective(obj, GRB.MAXIMIZE)
    mm.optimize()
    print("Dual objective: ", mm.getObjective().getValue())
    dual_vars = mm.getAttr("X", mm.getVars())

    #Check complementary slackness conditions
    cuts = dual_vars[:G.number_of_edges()]
    y = dual_vars[G.number_of_edges():]
    cmaxflows = A@flows - b
    print('Complementary slackness: ', cmaxflows@cuts)
    Aty_plus_z_min_r = c - A.transpose()@dual_vars
    print('Complementary slackness: ', Aty_plus_z_min_r@flows)

arr = np.reshape(intensity,(h,w)).astype('uint8')
final_image = Image.fromarray(arr)
final_image.save('final_image.png')
toc = time.time()
print ('elapsed ', toc - tic)
# %%
