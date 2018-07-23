from sneurons import SNeurons


sneurons = SNeurons([2,2], 20, 0.1)

sneurons.run(1000)

sneurons.plot()