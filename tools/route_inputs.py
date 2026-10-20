"""Run-length encoding for production-rate route inputs."""


def packed(inputs):
    result=[]
    for direction,power in inputs:
        if result and result[-1][0]==direction and not power and len(result[-1])==2:
            result[-1][1]+=1
        else:
            result.append([direction,1,True] if power else [direction,1])
    return result
