# return x^n, recursively. n \in N
def recursivePow(x, n):
    if n == 0:  # not needed actually
        return 1
    if n == 1:
        return x
    else:
        if n % 2 == 0:
            r = recursivePow(x, n / 2)
            return r * r
        else:
            r = recursivePow(x, (n - 1) / 2)
            return x * r * r


x = 2
for n in range(1, 10):
    print(recursivePow(x, n))
