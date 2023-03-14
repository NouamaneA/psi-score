
def push_nf_fifo(i, A_t, b_i, eps: float =1e-4):
    p = {}
    n_msg = 0
    if i not in A_t:
        return p, n_msg
    if len(A_t[i]) == 0:
        return p, n_msg

    mark = {}
    FIFO = []
    for s in b_i:
        if b_i[s] >= eps:
            FIFO.append(s)
            mark[s] = True

    while (len(FIFO)>0):
        u = FIFO.pop(0)
        if u in p:
            p[u] = p[u]+b_i[u]
        else:
            p[u] = b_i[u]

        for v in A_t[u]:
            n_msg += 1
            if v in b_i:
                b_i[v] = b_i[v]+A_t[u][v]*b_i[u]
            else:
                b_i[v] = A_t[u][v]*b_i[u]
            if (b_i[v]>eps) and (v not in mark):
                FIFO.append(v)
                mark[v]=True
        b_i[u] = 0
        del b_i[u]
        del mark[u]
    return p, n_msg

def push_fifo(M, res, eps: float =1e-4):
    apr = {}
    n_msg = 0

    mark = {}
    FIFO = []
    for u in res:
        if res[u] >= eps:
            FIFO.append(u)
            mark[u] = True

    while (len(FIFO)>0):
        u = FIFO.pop(0)
        if u in apr:
            apr[u] = apr[u]+res[u]
        else:
            apr[u] = res[u]

        for v in M[u]:
            n_msg += 1
            if v in res:
                res[v] = res[v]+M[u][v]*res[u]
            else:
                res[v] = M[u][v]*res[u]
            if (res[v]>eps) and (v not in mark):
                FIFO.append(v)
                mark[v]=True
        res[u] = 0
        del res[u]
        del mark[u]
    return apr, n_msg