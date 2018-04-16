""" A-Star Shortest Path.
Import library.
"""
from bisect import insort

def heuristic_value(path, finish, coordinate):
    """Heuristic Value.
    Function for searching heuristic value,
    based on coordinate, path, and finish target.
    """

    # result heuristic
    result_heuristic = pythagorean_distance_equation(coordinate[path], coordinate[finish])

    # return result
    return result_heuristic

def pythagorean_distance_equation(path1, path2):
    """Pythagorean Distance Equation.
    Function for counting distance, 
    derived from the Pythagorean theorem.
    """

    # point path dot X1
    dotX1 = path1[0]

    # point path dot X2
    dotX2 = path2[0]

    # point path dot Y1
    dotY1 = path1[1]

    # point path dot Y2
    dotY2 = path2[1]
    
    # result distance --> revise
    result_distance = ((((dotX2-dotX1)**2)+((dotY2-dotY1)**2))**0.5)

    # return result
    return result_distance

def shortest_path(M, start, goal):
    """Shortest Path A-Star Algorithm.
    Function for searching shortest path, 
    using A-Star algorithm.
    """

    # started
    print("A-Star Proccessing Started")

    # init result a-star variable
    result_a_star = []

    # initialization
    expanded_map = set()
    frontier_map = [start] 
    frontier_map_set = set([start])
    real_cost = {start:0} 
    real_predecessor = {start:None}
    heuristic_cost = {start:heuristic_value(start, goal, M.intersections)}

    # expanding map
    while frontier_map:
        # check emptyness --> revise
        if not frontier_map:
            print("failure: empty")
            break

        # choose best node
        best_path = frontier_map.pop(0)
        print("exploring", best_path)
        frontier_map_set.remove(best_path)
        expanded_map.add(best_path)

        # already goal --> revise
        if best_path == goal:
            print("finish", goal)
            break
        
        # updating expanded map from best path
        neighborhood = M.roads[best_path]
        for neighbors in neighborhood:
            neighbours_value = real_cost[best_path] + pythagorean_distance_equation(M.intersections[best_path], 
                                                                                    M.intersections[neighbors])
            if neighbors not in expanded_map:
                if neighbors not in frontier_map_set:
                    insort(frontier_map, neighbors) 
                    frontier_map_set.add(neighbors) 
                    real_cost[neighbors] = neighbours_value
                    heuristic_cost[neighbors] = heuristic_value(neighbors, goal, M.intersections)
                    real_predecessor[neighbors] = best_path
                    frontier_map.sort(key=lambda x:real_cost[x]+heuristic_cost[x])
                elif neighbours_value < real_cost[neighbors]:
                    real_cost[neighbors] = neighbours_value
                    heuristic_cost[neighbors] = heuristic_value(neighbors, goal, M.intersections)
                    real_predecessor[neighbors] = best_path
                    frontier_map.sort(key=lambda x:real_cost[x]+heuristic_cost[x])

    # getting goal, insert into path
    path = goal

    # insert a-star result into path
    while path != None:
        result_a_star.insert(0, path)
        path = real_predecessor[path]

    # finished
    print("A-Star Proccessing Finished")

    # return result
    return result_a_star