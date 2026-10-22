import matplotlib.pyplot as plt 

#print(plt.style.available)
#plt.style.use('bmh')


plt.xkcd()

ages_x = [25,34,46,57,67, 68,89]

dev_y = [1300,4300,500,5600,6700,7800,2300]

plt.plot(
    ages_x,
    dev_y, 
    color="#22210B", 
    linestyle='-',
    label="All devs")


py_dev_y =  [1000,4350,5000,5100,6700,7230,5300]

#plt.plot(ages_x, py_dev_y,color="#AA16AC",linestyle='-', label='All python')
plt.plot(ages_x, py_dev_y,label='All python')
java_dev = [1999, 2325, 3464, 45656, 23523, 23653, 23523]

#plt.plot(ages_x, java_dev, color="g", linewidth=3,linestyle="--", label="javascript")
plt.plot(ages_x, java_dev,label="javascript")


plt.xlabel("ages")
plt.ylabel("salary")
plt.title("Median salary by age")

plt.legend()

plt.grid(True)

plt.tight_layout()


plt.savefig('plot.png')
plt.show()

