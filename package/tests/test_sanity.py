import graph_structure as gs 

def test_nearest_neighbors():
    import numpy as np
    A = np.diag([1,2,3,4])
    nn = gs.nearest_neighbors(A)
    res = np.array( [[1, 2, 3, 0],
                    [0, 2, 3, 1],
                    [0, 1, 3, 2],
                    [0, 1, 2, 3]])
    assert (nn==res).all()
    
def main():
    test_nearest_neighbors()
    
if __name__ == "__main__":
    main()