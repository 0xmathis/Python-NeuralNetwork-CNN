a = [[1, 2, 3],
     [4, 5, 6]]

print(max(a[i][j] for j in range(3) for i in range(2)))
