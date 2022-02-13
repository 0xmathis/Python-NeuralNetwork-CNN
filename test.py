from numpy import array, random

arrayTest = array(
    [
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]],

        [[10, 11, 12],
         [13, 14, 15],
         [16, 17, 18]]
        ]
    )

print(arrayTest)
print(arrayTest.reshape((3*3*2)))

# arrayTest2 = random.randint(-5, 5, (3, 3, 3))
# print(arrayTest2)

# for x in arrayTest2.tolist():
#     print(x)
#     print()
