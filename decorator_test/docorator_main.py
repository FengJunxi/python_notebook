import time

def timmer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        # 将不定数量的参数传递给该函数
        func(*args, **kwargs)
        stop = time.perf_counter()
        time_used = stop - start
        print("%s time used is %fs" % (func.__name__, time_used))
    return wrapper

# here add decorator timmer
@timmer
def process_1(x, y):
    time.sleep(3)
    print('%d + %d = %d' % (x, y, x + y))


@timmer
def process_2(name):
    time.sleep(2)
    print('input is %s' % name)


@timmer
def process_3():
    time.sleep(2)
    print('process 3 test')


if __name__ == '__main__':
    process_1(x=1, y=2)
    process_2('test')
    process_3()
