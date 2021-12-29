from memory_profiler import profile
import pandas as pd
import numpy as np
import time

np.random.seed(192304)
log_start = str(int(time.time()))

def timer(func):
    loops = 100
    def wrapper(*args, **kwargs):
        print(func.__name__)
        start = time.time()
        for i in range(loops):
            result = func(*args, **kwargs)
        elapsed = time.time()-start
        with open(f'logs/intercept_logs_{log_start}', 'a') as f:
            f.write(f"{func.__name__} ran in {elapsed} seconds for {loops} loops\n")
        return result
    return wrapper

@profile
def base():
    X = pd.DataFrame(np.random.randint(0,100,size=(100000, 100)))
    ones1 = np.ones((X.shape[0], 1))
    ones2 = np.ones(X.shape[0])
    nparr = np.array(X)
    return X
    # return nparr

X = base()

@profile
@timer
def first_method():
    tempX = np.ones((X.shape[0], X.shape[1] + 1))
    tempX[:,1:] = X
    return tempX

@profile
@timer
def second_method():
    tempX = pd.concat([pd.DataFrame(np.ones(X.shape[0])), X], axis=1)
    return tempX

@profile
@timer
def third_method():
    tempX = np.c_[np.ones(X.shape[0]), X] 
    return tempX

@profile
@timer
def fourth_method():
    tempX = np.hstack([np.ones((X.shape[0], 1)), X]) 
    return tempX

@profile
@timer
def fifth_method():
    tempX = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    return tempX

@profile
@timer
def sixth_method():
    tempX = np.insert(np.array(X), 0, 1, axis=1)
    # tempX = np.insert(X, 0, 1, axis=1)
    return tempX

@profile
@timer
# def seventh_method():
#     X['intercept'] = 1
#     columns = list(X.columns)
#     columns[0], columns[1:] = columns[-1], columns[0:-1]
#     X.reindex(columns=columns)
#     return X
def seventh_method(tempX):
    tempX['intercept'] = 1
    columns = list(tempX.columns)
    columns[0], columns[1:] = columns[-1], columns[0:-1]
    tempX.reindex(columns=columns)
    return tempX



if __name__ == '__main__':
    tempX = first_method()
    tempX = second_method()
    tempX = third_method()
    tempX = fourth_method()
    tempX = fifth_method()
    tempX = sixth_method()
    tempX = X.copy()
    tempX = seventh_method(tempX)
    # tempX = seventh_method()
