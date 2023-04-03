def append_function(x, list, idx):
    list.append([])
    for i in range(x.shape[0]):
        list[idx].append(x[i])

    return list
