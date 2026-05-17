def fizz_buzz(n):
    if n % 15 == 0:
        a ="FizzBuzz"
    elif n % 3 == 0:
        a = "Fizz"
    elif n % 5 == 0:
        a = "Buzz"
    else :
        a = str (n)
    return a
n=int(input ("判定する整数は？"))