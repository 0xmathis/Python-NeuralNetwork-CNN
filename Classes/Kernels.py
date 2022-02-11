# accentue les differences de pixels
sharpen = [[0, -1, 0],
           [-1, 5, -1],
           [0, -1, 0]]

# pixels adjacents de la meme couleur : pixel noir, couleurs differentes : pixel blanc
outline = [[-1, -1, -1],
           [-1, 8, -1],
           [-1, -1, -1]]
