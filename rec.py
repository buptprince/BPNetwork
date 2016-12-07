import matplotlib.pyplot as plt
fin = open("rec.txt")
js = []
for line in fin.readlines():
    sp = line.split(":")
    try:
        j = float(sp[2])
        js.append(j)
    except:
        pass

plt.plot(js)
plt.show()
