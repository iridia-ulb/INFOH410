
class Graph:
    '''
    The Graph holds a list of all its vertices and edges.
    Upon creation, it propagates the Vertex instances to the Edge instances 
    connecting them and vice versa.
    '''
    def __init__(self, e, v):
        self.edges = e # list of Edge instances
        self.vertices = v # list of Vertex instances

        # Link edges to vertices
        verticesDict = {v.id: v for v in self.vertices}
        for edge in self.edges:
            verticesDict[edge.vertexAID].addEdge(edge)
            edge.vertexA = verticesDict[edge.vertexAID]
            verticesDict[edge.vertexBID].addEdge(edge)
            edge.vertexB = verticesDict[edge.vertexBID]

class Vertex:
    '''
    Each vertex has a unique ID given at creation and holds a list
    of Edge instances connecting it to other Vertex instances.
    '''
    def __init__(self, ID):
        self.id = ID
        self.edges = list()
        self.part = None # Partition label, A or B

    def addEdge(self, edge):
        '''
        Each edge should connect two same vertices only once.
        Skipping verification.
        '''
        self.edges.append(edge)

    def getEx(self):
        '''External cost'''
        Ex = 0
        for edge in self.edges:
            if edge.vertexA.id == self.id:
                otherVertex = edge.vertexB
            elif edge.vertexB.id == self.id:
                otherVertex = edge.vertexA
            if otherVertex.part != self.part:
                Ex += edge.weight
        return Ex

    def getIx(self):
        '''Internal cost'''
        Ix = 0
        for edge in self.edges:
            if edge.vertexA.id == self.id:
                otherVertex = edge.vertexB
            elif edge.vertexB.id == self.id:
                otherVertex = edge.vertexA
            if otherVertex.part == self.part:
                Ix += edge.weight
        return Ix

    def swapPart(self):
        if self.part == "A":
            self.part = "B"
        elif self.part == "B":
            self.part = "A"


class Edge:
    '''
    Each Edge has a weight given at creation and holds a pointer
    to the two Vertex instances it connects.
    '''
    def __init__(self, w, va, vb):
        self.weight = w
        self.vertexAID = va
        self.vertexBID = vb
        self.vertexA = None
        self.vertexB = None

def loadGraph(graphfile):
    '''
    Load a graph from graphile, each line formated with  as follows:
    <weight [float]> <vertex_A_ID [int]> <vertex_B_ID [int]>
    '''
    edges = list() # list of Edge instances
    vertices = list() # list of Vertex instances
    verticesID = list() # list of already created vertices ID

    with open(graphfile, 'r') as f:
        lines = f.readlines()
    for line in lines:
        weight = float(line.split()[0])
        vertexA = Vertex(int(line.split()[1]))
        vertexB = Vertex(int(line.split()[2]))
        edge = Edge(weight, vertexA.id, vertexB.id)

        edges.append(edge)

        if vertexA.id not in verticesID:
            vertices.append(vertexA)
            verticesID.append(vertexA.id)
        if vertexB.id not in verticesID:
            vertices.append(vertexB)
            verticesID.append(vertexB.id)

    return Graph(edges, vertices)