import numpy as np
from shapely.geometry import Polygon


def intersection(g, p):
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter/union


def weighted_merge(g, p):
    g[:8] = (g[8] * g[:8] + p[8] * p[:8])/(g[8] + p[8])
    g[8] = (g[8]+p[8])
    return g

def merge(g, p):
    g = np.array(g)
    p = np.array(p)
    g[0] = np.minimum(g[0], p[0])
    g[1] = np.minimum(g[1], p[1])
    g[2] = np.maximum(g[2], p[2])
    g[3] = np.minimum(g[3], p[3])
    g[4] = np.maximum(g[4], p[4])
    g[5] = np.maximum(g[5], p[5])
    g[6] = np.minimum(g[6], p[6])
    g[7] = np.maximum(g[7], p[7])
    return g
# 修改nms
# def standard_nms(S, thres, nms_thres):
def standard_nms(S, thres):
    print(type(S))
    order = np.argsort(S[:, 8])[::-1]
    keep = []
    remain=[]
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = np.array([intersection(S[i], S[t]) for t in order[1:]])
        inds = np.where(ovr <= thres)[0]
        order = order[inds+1]

    # l = list(range(len(S[keep])))
    # while len(l) != 0:
    #     g = S[keep][l[0]]
    #     # print(len(temp))
    #     l.remove(l[0])
    #     # print(len(temp))
    #     if l != None:
    #         for i in l:
    #             q = S[keep][i]
    #             if intersection(g, q) > nms_thres:
    #                 g = merge(g, q)
    #                 del l[int(np.where(i)[0])]
    #         remain.append(g)
    #     else:
    #         remain.append(g)
    return S[keep]
    #return np.array(remain)

def nms(boxes, nms_thres):
    S = []
    q = None
    for g in boxes:
        if q is not None and intersection(g, q) > nms_thres:
            q = merge(g, q)
        else:
            if q is not None:
                S.append(q)
            q = g
    if q is not None:
        S.append(q)

    # order = np.argsort(boxes[:, 8])[::-1]
    # remain = []
    # temp_list = []
    # while order.size > 0:
    #     i = order[0]
    #     #remain.append(i)
    #     odr = np.array([intersection(boxes[i], boxes[t]) for t in order[1:]])
    #     inds = np.where(odr <= nms_thres)[0]
    #     temp_list = order[inds]
    #     for j in temp_list[1:]:
    #         boxes[i] = merge(boxes[i], boxes[j])
    #     remain.append(i)
    #     order = order[inds+1]

    # l = list(range(len(boxes)))
    # while len(l) != 0:
    #     g = boxes[l[0]]
    #     # print(len(temp))
    #     l.remove(l[0])
    #     # print(len(temp))
    #     if l != None:
    #         for i in l:
    #             q = boxes[i]
    #             if intersection(g, q) > nms_thres:
    #                 g = merge(g, q)
    #                 num = i
    #                 l.pop(num)
    #         remain.append(g)
    #     else:
    #         remain.append(g)
    # print(remain)
    return standard_nms(np.array(S), nms_thres)

def nms_locality_1(polys, thres, nms_thres):
    '''
    locality aware nms of EAST
    :param polys: a N*9 numpy array. first 8 coordinates, then prob
    :return: boxes after nms
    '''
    S = []
    temp = []
    p = None
    for g in polys:
        if p is not None and intersection(g, p) > thres:
            p = weighted_merge(g, p)
        else:
            if p is not None:
                temp.append(p)
            p = g
    if p is not None:
        temp.append(p)
    if len(temp) == 0:
        return np.array([])
    return standard_nms(np.array(temp), thres)

def nms_locality(polys, thres, nms_thres):
    '''
    locality aware nms of EAST
    :param polys: a N*9 numpy array. first 8 coordinates, then prob
    :return: boxes after nms
    '''
    S = []
    temp = []
    p = None
    for g in polys:
        if p is not None and intersection(g, p) > thres:
            p = weighted_merge(g, p)
        else:
            if p is not None:
                temp.append(p)
            p = g
    if p is not None:
        temp.append(p)
    # l = list(range(len(temp)))
    # while len(l) != 0:
    #     g = temp[l[0]]
    #     #print(len(temp))
    #     l.remove(l[0])
    #     #print(len(temp))
    #     if l != None:
    #         for i in l:
    #             q = temp[i]
    #             if intersection(g, q) > nms_thres:
    #                 g = merge(g, q)
    #                 del l[int(np.where(i)[0])]
    #         S.append(g)
    #     else:
    #         S.append(g)
    q = None
    for g in temp:
        if q is not None and intersection(g, q) > nms_thres:
            q = merge(g, q)
        else:
            if q is not None:
                S.append(q)
            q = g
    if q is not None:
        S.append(q)
    if len(temp) == 0:
        return np.array([])
    return standard_nms(np.array(S), nms_thres)
    #return nms(np.array(temp), thres)



if __name__ == '__main__':
    # 343,350,448,135,474,143,369,359
    print(Polygon(np.array([[343, 350], [448, 135],
                            [474, 143], [369, 359]])).area)