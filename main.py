def fizz_buzz(n,x,y):
    if n % x*y == 0:
        a ="FizzBuzz"
    elif n % x == 0:
        a = "Fizz"
    elif n % y == 0:
        a = "Buzz"
    else :
        a = str (n)
    return a
n=int(input ("判定する整数nは？"))
x=int(input ("判定する整数xは？"))
y=int(input ("判定する整数yは？"))
print (fizz_buzz(n,x,y))