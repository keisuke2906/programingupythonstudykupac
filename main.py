dict = {}
itiran = []
while 2 > 1:
    a = input("登録する概念の英字は？")
    dict[a] = input(a + "の意味は？")
    itiran.insert(0, a)
    b = input("登録を続ける？(yes/no)") 
    if b != "yes":
        break
print("登録された概念は" + itiran)
c = input("意味を知りたい概念の英字は？")
print(dict[c])