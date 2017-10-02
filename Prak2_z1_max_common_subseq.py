def lenMaxCommonSubsequence(list1, list2):
    max = 0

    len1 = len(list1)
    len2 = len(list2)
    i = 0
    while i < len1 - max:
        curr_max = 0
        sub_i = i
        j = 0
        while j < len2 and sub_i < len1:
            if list1[sub_i] == list2[j]:
                curr_max += 1
                sub_i += 1
            else:
                sub_i = i
                curr_max = 0
            j += 1
            if (curr_max > max):
                max = curr_max
        i += 1
    return max


list1 = [6, 9, 3, 1, 4, 5, 6, 7, 9, 1]
list2 = [6, 9, 3, 4, 5, 6, 7, 9]

print(lenMaxCommonSubsequence(list1, list2))
