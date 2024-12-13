#!/usr/bin/python3
import sys
from tqdm import tqdm
def coord_2_id(x,y,z,a,b,c):
    def check_bounds(x, y, z, a, b, c):
        if not (0 <= x < 24):
            print(f"Error: x={x} is out of bounds (0-23)")
            return False
        if not (0 <= y < 23):
            print(f"Error: y={y} is out of bounds (0-22)")
            return False
        if not (0 <= z < 24):
            print(f"Error: z={z} is out of bounds (0-23)")
            return False
        if not (0 <= a < 2):
            print(f"Error: a={a} is out of bounds (0-1)")
            return False
        if not (0 <= b < 3):
            print(f"Error: b={b} is out of bounds (0-2)")
            return False
        if not (0 <= c < 2):
            print(f"Error: c={c} is out of bounds (0-1)")
            return False
        return True

    def coord_to_id(x,y,z,a,b,c):
        return x*12+y*288+z*6624+a+b*2+c*6

    if check_bounds(x, y, z, a, b, c):
        return coord_to_id(x,y,z,a,b,c)
    else: 
        return -1


def check():
    filename = "tofu.dot"
    connections = {}

    num_lines = sum(1 for _ in open(filename))

    with open(filename, 'r') as file:
        for line in tqdm(file,  total=num_lines,  desc="Processing lines"):
            if "--" in line:
                parts = line.split(" -- ")
                node1 = parts[0].strip().strip('"')
                node2 = parts[1].split()[0].strip().strip('"')
                # print(node1, node2)

                # Ensure node1 is in the dictionary
                if node1 not in connections:
                    connections[node1] = []
                # Ensure node2 is in the dictionary
                if node2 not in connections:
                    connections[node2] = []
            
                connections[node1].append(node2)
                connections[node2].append(node1)

    for z in tqdm(range(24),desc="Checking Formulas"):
        for y in range(23):
            for x in range(24):
                for c in range(2):
                    for b in range(3):
                        for a in range(2):
                            s = a+2*c+4*b #given by Jens
                            s_node = f"S<{x},{y},{z},{s}>"
                            t_node = next((t_node for t_node in connections[s_node] if t_node.startswith("T")), None)
                            id = coord_2_id(x,y,z,a,b,c)
                            t_try = f"T<{id}>"
                            # print (x,y,z,a,b,c,"~>",s_node,"->",t_node, "=>", t_try)
                            if t_try != t_node:
                                print ("Error in formula for:", s_node,"->",t_node, "=>", t_try)
                                quit()
        
    print ("No error detected in formula for converting coordinates to node ID"); 

if __name__ == "__main__":
    # Ensure there are exactly 6 arguments (excluding the script name)
    if len(sys.argv) != 7:  # sys.argv[0] is the script name
        print("Script to convert Fugaku 6D node coordinates into Node ID of the tofu.dot file")
        print(f"Usage: {sys.argv[0]} x y z a b c")
        print("Since there are not exactly 6 parameters given, checking the formula...")
        check()
        quit()

    # Assign the first 6 arguments to variables
    x, y, z, a, b, c = map(int, sys.argv[1:7])
    # print(x, y, z, a, b, c)
    node_id = coord_2_id(x,y,z,a,b,c)
    print(node_id)

