import cv2


def rect_contains(rect, point):
    if point[0] < rect[0] or point[0] > rect[0] + rect[2]:
        return False
    elif point[1] < rect[1] or point[1] > rect[1] + rect[3]:
        return False
    return True


def draw_point(img, p, color):
    cv2.circle(img, p, 10, color, cv2.FILLED, cv2.LINE_AA)


def draw_delaunay(img, subdiv, delaunay_color):
    triangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])
    for t in triangleList:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(
                r, pt3):
            cv2.line(img, pt1, pt2, delaunay_color, 3, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 3, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 3, cv2.LINE_AA, 0)


def draw_voronoi(img, subdiv):
    (facets, centers) = subdiv.getVoronoiFacetList([])
    for i in range(0, len(facets)):
        ifacet_arr = []
        for f in facets[i]:
            ifacet_arr.append(f)
        ifacet = np.array(ifacet_arr, np.int32)
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        cv2.fillConvexPoly(img, ifacet, color)
        ifacets = np.array([ifacet])
        cv2.polylines(img, ifacets, True, (0, 0, 0), 1)
        cv2.circle(img, (centers[i][0], centers[i][1]), 3, (0, 0, 0),
                   cv2.FILLED)
