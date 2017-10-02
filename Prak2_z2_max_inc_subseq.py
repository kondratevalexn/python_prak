def lenMaxIncreasingSequence(list):
    max = 0
    curr_max = 1
    if len(list) == 0:
        return 0
    prev = list[0]
    for x in list:
        if not isinstance(x, int):
            raise TypeError
        if x > prev:
            curr_max += 1
        else:
            curr_max = 1
        if curr_max > max:
            max = curr_max
        prev = x
    return max


try:
    list = [7, 7, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 0, 1, 0, 1]
    print(lenMaxIncreasingSequence(list))
except TypeError:
    print("Not integer in list")
