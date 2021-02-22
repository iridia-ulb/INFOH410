from graph import *

def kl(graph):
    # Initial "random" partitioning
    # First half is A, second is B
    for i in range(int(len(graph.vertices)/2)):
        graph.vertices[i].part = "A"
    for i in range(int(len(graph.vertices)/2), len(graph.vertices)):
        graph.vertices[i].part = "B"

    while True:
        # Reset the groups, "unlock" the vertices.
        groupA = list()
        groupB = list()

        for vertex in graph.vertices:
            if vertex.part == "A":
                groupA.append(vertex)
            elif vertex.part == "B":
                groupB.append(vertex)
        gains = list()
        swaps = list()

        # Compute the initial cutsize of the partition
        cutsize = 0
        for vertexA in groupA:
            for vertexB in groupB:
                for edge in vertexA.edges:
                        if edge.vertexA.id == vertexB.id or edge.vertexB.id == vertexB.id:
                            cutsize += edge.weight

        while len(groupA) > 0 and len(groupB) > 0:

            potentialGains = list() # list of gains
            potentialSwaps = list() # [[a,b], ...], same order as potentialGains

            # For each pair of vertices, compute the gain if they were swapped.
            for vertexA in groupA:
                for vertexB in groupB:
                    edgeAB = None
                    for edge in vertexA.edges:
                        if edge.vertexA.id == vertexB.id or edge.vertexB.id == vertexB.id:
                            edgeAB = edge
                            break
                    if edgeAB:
                        gain = (vertexA.getEx() - vertexA.getIx()) + (vertexB.getEx() - vertexB.getIx()) - (2*edgeAB.weight)
                        potentialGains.append(gain)
                        potentialSwaps.append([vertexA, vertexB])
            if len(potentialGains) == 0:
                break
            # Only keep the maximum gain, even if it's negative
            idx = potentialGains.index(max(potentialGains))
            gains.append(potentialGains[idx])
            swaps.append(potentialSwaps[idx])

            # Remove the vertices from their groups, they are now "locked".
            groupA.remove(swaps[-1][0])
            groupB.remove(swaps[-1][1])

        print("Initial cutsize: {}".format(cutsize))

        # Update the cutsize
        newCutsize = cutsize
        for i, gain in enumerate(gains):
            # If the cut is improved, aknowledge the gain and swap the vertices.
            if (newCutsize - gain) < newCutsize:
                newCutsize -= gain
                for vertex in swaps[i]:
                    vertex.swapPart()
            else:
                break
        print("New cutsize: {}\n".format(newCutsize))

        # If the cut is not improved, stop.
        if newCutsize == cutsize:
            break
    print("The best I cant do is {}".format(cutsize))








def mainKL():
    graph = loadGraph("large_graph_50380_539746.txt")
    kl(graph)


if __name__ == "__main__":
    mainKL()